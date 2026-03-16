"""
Step 3: Fine-tune a LeRobot policy with advantage-weighted regression.

The loss is scaled by per-step advantage weights before backprop, so the
policy learns more from high-advantage (high-progress) actions.

Usage:
    python -m awr.weighted_trainer \
        --dataset "your_username/so100_pick_cube" \
        --advantages awr/outputs/advantages/advantages.json \
        --checkpoint "your_username/act_so100_pick_cube" \
        --policy-type act \
        --steps 5000 \
        --output awr/outputs/checkpoints/

IMPORTANT NOTES (version-dependent):
  - LeRobot policy API changes between versions. Lines marked # ADAPT THIS
    may need adjustment for your installed lerobot version.
  - The loss returned by policy.forward() may be a scalar (already reduced)
    or per-sample. See _apply_weighted_loss() for how both are handled.
  - Check the dataset's field names with: python -c "from lerobot...; print(dataset[0].keys())"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from awr.config import (
    BATCH_SIZE,
    BASE_CHECKPOINT,
    DATASET_REPO_ID,
    DATASET_ROOT,
    DEVICE,
    LEARNING_RATE,
    NUM_TRAIN_STEPS,
    OUTPUT_DIR,
    POLICY_TYPE,
    SAVE_EVERY,
)


# ─── Dataset ─────────────────────────────────────────────────────────────────

def load_lerobot_dataset(repo_id: str | None, root: str | None):
    """Load a LeRobot dataset."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # ADAPT THIS
    except ImportError as e:
        raise ImportError("lerobot not installed. Run: pip install lerobot") from e

    if root:
        return LeRobotDataset(repo_id=repo_id or "local", root=root)
    return LeRobotDataset(repo_id=repo_id)


def build_weights_array(dataset, advantages_path: Path) -> np.ndarray:
    """
    Build a flat weight array aligned with dataset indices.

    weights[i] = advantage weight for dataset[i].
    Uses episode_data_index to map (episode, frame) → global index.
    """
    with open(advantages_path) as f:
        advantages = json.load(f)

    weights = np.ones(len(dataset), dtype=np.float32)
    episode_data_index = dataset.episode_data_index

    for ep_data in advantages["episodes"]:
        ep_id = ep_data["episode_id"]
        if ep_id >= len(episode_data_index["from"]):
            continue
        start = episode_data_index["from"][ep_id].item()
        end = episode_data_index["to"][ep_id].item()
        ep_weights = np.array(ep_data["weights"], dtype=np.float32)

        # Clip to actual episode length in case of off-by-one
        n = min(end - start, len(ep_weights))
        weights[start:start + n] = ep_weights[:n]

    return weights


# ─── Policy loading ───────────────────────────────────────────────────────────

def load_policy(policy_type: str, checkpoint: str, device: str):
    """
    Load a pretrained LeRobot policy.

    # ADAPT THIS: LeRobot's policy loading API varies by version.
    # The snippet below covers the most common patterns; adjust as needed.
    """
    policy_type = policy_type.lower()

    try:
        # Modern lerobot (>= 0.2): use from_pretrained
        if policy_type == "act":
            from lerobot.common.policies.act.modeling_act import ACTPolicy  # ADAPT THIS
            policy = ACTPolicy.from_pretrained(checkpoint)
        elif policy_type == "diffusion":
            from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy  # ADAPT THIS
            policy = DiffusionPolicy.from_pretrained(checkpoint)
        else:
            raise ValueError(f"Unknown policy type: {policy_type!r}. Expected 'act' or 'diffusion'.")
    except AttributeError:
        # Older lerobot: instantiate from config + load state dict manually
        raise RuntimeError(
            f"Could not load policy '{policy_type}' from '{checkpoint}'. "
            "Check your lerobot version and adjust the import in weighted_trainer.py."
        )

    policy = policy.to(device)
    policy.train()
    return policy


# ─── Loss ─────────────────────────────────────────────────────────────────────

def _apply_weighted_loss(
    policy,
    batch: dict,
    batch_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute weighted loss.

    LeRobot policies may return:
      (a) A scalar loss — we approximate weighting by multiplying by the batch mean weight.
      (b) A per-sample loss tensor — we weight each sample then take the mean.

    # ADAPT THIS: inspect policy.forward(batch) to see what it returns.
    """
    loss_dict = policy.forward(batch)  # ADAPT THIS — some versions use policy(batch)

    loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict

    if loss.dim() == 0:
        # Scalar loss: use mean batch weight as approximation
        weighted_loss = loss * batch_weights.mean()
    else:
        # Per-sample loss: shape (B,) or (B, T, ...)
        # Reduce spatial/temporal dims first, then weight by sample
        while loss.dim() > 1:
            loss = loss.mean(dim=-1)
        weighted_loss = (loss * batch_weights).mean()

    return weighted_loss


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(policy, output_dir: Path, step: int, final: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "final" if final else f"step_{step:06d}"
    ckpt_path = output_dir / tag
    ckpt_path.mkdir(exist_ok=True)

    try:
        policy.save_pretrained(str(ckpt_path))  # ADAPT THIS
    except AttributeError:
        # Fallback: save state dict
        torch.save(policy.state_dict(), ckpt_path / "policy.pt")

    print(f"  Saved checkpoint: {ckpt_path}")


# ─── Training loop ────────────────────────────────────────────────────────────

def train_awr(
    repo_id: str | None,
    root: str | None,
    advantages_path: Path,
    base_checkpoint: str,
    output_dir: Path,
    policy_type: str = POLICY_TYPE,
    num_steps: int = NUM_TRAIN_STEPS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device: str = DEVICE,
    save_every: int = SAVE_EVERY,
):
    print("Loading dataset...")
    dataset = load_lerobot_dataset(repo_id, root)
    print(f"  {len(dataset)} frames")

    print("Building weight array...")
    weights = build_weights_array(dataset, advantages_path)
    print(f"  Mean weight: {weights.mean():.3f}  Std: {weights.std():.3f}")

    print(f"Loading policy ({policy_type}) from {base_checkpoint}...")
    policy = load_policy(policy_type, base_checkpoint, device)
    print(f"  Policy loaded on {device}")

    # ADAPT THIS: LeRobot DataLoader construction varies by version.
    # Some versions require a specific collate_fn or sampler.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    log = []
    step = 0
    print(f"\nTraining for {num_steps} steps (batch_size={batch_size}, lr={lr})...")
    print()

    for epoch in range(9999):
        for batch in dataloader:
            if step >= num_steps:
                break

            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Retrieve advantage weights for this batch
            # ADAPT THIS: field name for global index may differ ("index", "frame_index", etc.)
            if "index" in batch:
                batch_idx = batch["index"].cpu().numpy()
            else:
                # Fallback: use uniform weights if index key not found
                batch_idx = None

            if batch_idx is not None:
                batch_weights = torch.tensor(
                    weights[batch_idx], device=device, dtype=torch.float32
                )
            else:
                batch_weights = torch.ones(batch_size, device=device)

            weighted_loss = _apply_weighted_loss(policy, batch, batch_weights)

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                mw = batch_weights.mean().item()
                print(
                    f"  Step {step:5d}/{num_steps} | "
                    f"loss={weighted_loss.item():.4f} | "
                    f"mean_weight={mw:.3f}"
                )

            log.append({
                "step": step,
                "loss": weighted_loss.item(),
                "mean_weight": batch_weights.mean().item(),
            })

            if step > 0 and step % save_every == 0:
                save_checkpoint(policy, output_dir, step)

            step += 1

        if step >= num_steps:
            break

    save_checkpoint(policy, output_dir, step, final=True)

    # Write training log
    log_path = output_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved to {log_path}")

    return policy


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AWR fine-tuning with advantage-weighted loss.")
    parser.add_argument("--dataset", default=DATASET_REPO_ID or None,
                        help="LeRobot dataset repo_id")
    parser.add_argument("--dataset-root", default=DATASET_ROOT,
                        help="Local dataset root path")
    parser.add_argument("--advantages", type=Path,
                        default=OUTPUT_DIR / "advantages" / "advantages.json",
                        help="Path to advantages.json")
    parser.add_argument("--checkpoint", default=BASE_CHECKPOINT,
                        help="Pretrained policy checkpoint (local path or HF repo)")
    parser.add_argument("--policy-type", default=POLICY_TYPE, choices=["act", "diffusion"],
                        help="Policy architecture")
    parser.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print("AWR Fine-Tuning")
    print("=" * 60)
    print(f"Dataset:     {args.dataset or args.dataset_root}")
    print(f"Advantages:  {args.advantages}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Policy:      {args.policy_type}")
    print(f"Steps:       {args.steps}")
    print(f"LR:          {args.lr}")
    print(f"Device:      {args.device}")
    print(f"Output:      {args.output}")
    print()

    train_awr(
        repo_id=args.dataset,
        root=args.dataset_root,
        advantages_path=args.advantages,
        base_checkpoint=args.checkpoint,
        output_dir=args.output,
        policy_type=args.policy_type,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
