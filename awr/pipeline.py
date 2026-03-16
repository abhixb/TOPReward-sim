"""
Full AWR pipeline: score → compute advantages → inspect → train.

Usage:
    python -m awr.pipeline \
        --dataset "your_username/so100_pick_cube" \
        --instruction "Pick up the red cube and place it in the box" \
        --checkpoint "your_username/act_so100_pick_cube" \
        --policy-type act
"""

import argparse
import json
import time
from pathlib import Path

from awr.config import (
    BASE_CHECKPOINT,
    BATCH_SIZE,
    CAMERA_KEY,
    DATASET_REPO_ID,
    DATASET_ROOT,
    DEVICE,
    INSTRUCTION,
    LEARNING_RATE,
    MAX_FRAMES_PER_PREFIX,
    NUM_EVAL_FRAMES,
    NUM_TRAIN_STEPS,
    OUTPUT_DIR,
    POLICY_TYPE,
    SAVE_EVERY,
)


def main():
    parser = argparse.ArgumentParser(description="End-to-end AWR fine-tuning pipeline.")
    parser.add_argument("--dataset", default=DATASET_REPO_ID or None,
                        help="LeRobot dataset repo_id")
    parser.add_argument("--dataset-root", default=DATASET_ROOT,
                        help="Local dataset root path")
    parser.add_argument("--instruction", default=INSTRUCTION,
                        help="Task instruction string")
    parser.add_argument("--camera-key", default=CAMERA_KEY)
    parser.add_argument("--checkpoint", default=BASE_CHECKPOINT,
                        help="Pretrained policy checkpoint")
    parser.add_argument("--policy-type", default=POLICY_TYPE, choices=["act", "diffusion"])
    parser.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--eval-frames", type=int, default=NUM_EVAL_FRAMES)
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy scorer (no model loading, for testing)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Run scoring + advantages + inspect only (no training)")
    args = parser.parse_args()

    output_dir = args.output_dir
    scores_dir = output_dir / "scores"
    advantages_path = output_dir / "advantages" / "advantages.json"
    plots_dir = output_dir / "plots"
    checkpoint_dir = output_dir / "checkpoints" / "awr_finetuned"

    print("=" * 60)
    print("AWR Fine-Tuning with TOPReward")
    print("=" * 60)
    print(f"Dataset:     {args.dataset or args.dataset_root}")
    print(f"Instruction: {args.instruction}")
    print(f"Policy:      {args.policy_type}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Steps:       {args.steps}")
    print(f"Output dir:  {output_dir}")
    print()

    pipeline_start = time.time()
    run_log = {
        "dataset": args.dataset or args.dataset_root,
        "instruction": args.instruction,
        "policy_type": args.policy_type,
        "base_checkpoint": args.checkpoint,
        "num_steps": args.steps,
        "steps": {},
    }

    # ── Step 1: Score dataset ────────────────────────────────────────────────
    print("[1/4] Scoring dataset with TOPReward...")
    t0 = time.time()

    from awr.score_dataset import score_dataset
    score_dataset(
        repo_id=args.dataset,
        root=args.dataset_root,
        instruction=args.instruction,
        camera_key=args.camera_key,
        output_dir=scores_dir,
        num_eval_frames=args.eval_frames,
        max_frames_per_prefix=MAX_FRAMES_PER_PREFIX,
        dummy=args.dummy,
    )

    t1 = time.time()
    run_log["steps"]["scoring"] = {"elapsed_s": round(t1 - t0, 2)}
    print(f"  Done in {t1 - t0:.1f}s\n")

    # ── Step 2: Compute advantages ───────────────────────────────────────────
    print("[2/4] Computing advantage weights...")
    t0 = time.time()

    from awr.compute_advantages import compute_advantages
    adv = compute_advantages(
        scores_dir=scores_dir,
        output_path=advantages_path,
    )

    t1 = time.time()
    run_log["steps"]["advantages"] = {
        "elapsed_s": round(t1 - t0, 2),
        "stats": adv["stats"],
    }
    print(f"  Done in {t1 - t0:.1f}s\n")

    # ── Step 3: Inspect ──────────────────────────────────────────────────────
    print("[3/4] Generating inspection plots...")
    t0 = time.time()

    from awr.inspect_scores import run_inspect
    run_inspect(scores_dir, advantages_path, plots_dir)

    t1 = time.time()
    run_log["steps"]["inspect"] = {"elapsed_s": round(t1 - t0, 2)}
    print(f"  Done in {t1 - t0:.1f}s\n")

    # ── Step 4: Train ────────────────────────────────────────────────────────
    if args.skip_train:
        print("[4/4] Skipping training (--skip-train).")
        run_log["steps"]["training"] = {"skipped": True}
    else:
        print("[4/4] Training with AWR...")
        t0 = time.time()

        from awr.weighted_trainer import train_awr
        train_awr(
            repo_id=args.dataset,
            root=args.dataset_root,
            advantages_path=advantages_path,
            base_checkpoint=args.checkpoint,
            output_dir=checkpoint_dir,
            policy_type=args.policy_type,
            num_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            save_every=SAVE_EVERY,
        )

        t1 = time.time()
        run_log["steps"]["training"] = {"elapsed_s": round(t1 - t0, 2)}
        print(f"  Done in {t1 - t0:.1f}s\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    run_log["total_elapsed_s"] = round(total, 2)

    log_path = output_dir / "run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print()
    print("=" * 60)
    print("Done!")
    print(f"  Total time:           {total:.1f}s")
    if not args.skip_train:
        print(f"  Finetuned checkpoint: {checkpoint_dir}")
    print(f"  Plots:                {plots_dir}")
    print(f"  Run log:              {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
