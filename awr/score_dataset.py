"""
Step 1: Score a LeRobot dataset with TOPReward.

Usage:
    python -m awr.score_dataset \
        --dataset "your_username/so100_pick_cube" \
        --instruction "Pick up the red cube and place it in the box" \
        --output awr/outputs/scores/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from the parent repo
sys.path.insert(0, str(Path(__file__).parent.parent))

from awr.config import (
    CAMERA_KEY,
    DATASET_REPO_ID,
    DATASET_ROOT,
    INSTRUCTION,
    MAX_FRAMES_PER_PREFIX,
    NUM_EVAL_FRAMES,
    OUTPUT_DIR,
)
from reward_wrapper import RewardScorer, score_trajectory


def load_dataset(repo_id: str | None, root: str | None):
    """Load a LeRobot dataset from HuggingFace or local path."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ImportError(
            "lerobot not installed. Run: pip install lerobot"
        ) from e

    if root:
        return LeRobotDataset(repo_id=repo_id or "local", root=root)
    if repo_id:
        return LeRobotDataset(repo_id=repo_id)
    raise ValueError("Provide --dataset (repo_id) or --dataset-root (local path).")


def extract_episode_frames(dataset, episode_id: int, camera_key: str):
    """Return list of PIL Images for one episode."""
    from torchvision.transforms.functional import to_pil_image

    episode_data_index = dataset.episode_data_index
    start = episode_data_index["from"][episode_id].item()
    end = episode_data_index["to"][episode_id].item()

    frames = []
    for idx in range(start, end):
        item = dataset[idx]
        if camera_key not in item:
            available = [k for k in item.keys() if "image" in k.lower() or "obs" in k.lower()]
            raise KeyError(
                f"Camera key '{camera_key}' not found in dataset. "
                f"Available image-like keys: {available}"
            )
        tensor = item[camera_key]
        frames.append(to_pil_image(tensor))

    return frames


def score_dataset(
    repo_id: str | None,
    root: str | None,
    instruction: str,
    camera_key: str,
    output_dir: Path,
    num_eval_frames: int,
    max_frames_per_prefix: int,
    dummy: bool = False,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {'from ' + repo_id if repo_id else 'locally'}...")
    dataset = load_dataset(repo_id, root)
    num_episodes = len(dataset.episode_data_index["from"])
    print(f"  {num_episodes} episodes, {len(dataset)} total frames")
    print(f"  Camera key: {camera_key}")
    print()

    scorer = RewardScorer(dummy=dummy)
    results = []

    for ep_id in range(num_episodes):
        t0 = time.time()

        try:
            frames = extract_episode_frames(dataset, ep_id, camera_key)
        except KeyError as e:
            print(f"  [Episode {ep_id:03d}] ERROR: {e}")
            continue

        result = score_trajectory(
            scorer,
            frames,
            instruction,
            k=num_eval_frames,
            max_frames_per_prefix=max_frames_per_prefix,
        )

        elapsed = time.time() - t0

        record = {
            "episode_id": ep_id,
            "num_frames": len(frames),
            "voc": result["voc"],
            "raw_scores": result["raw_scores"],
            "normalized": result["normalized"],
            "rank_normalized": result["rank_normalized"],
            "prefix_lengths": result["prefix_lengths"],
            "scoring_time_s": round(elapsed, 2),
        }
        results.append(record)

        out_path = output_dir / f"episode_{ep_id:03d}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        voc_str = f"{result['voc']:+.3f}"
        print(
            f"  Episode {ep_id:03d} | frames={len(frames):4d} | "
            f"VOC={voc_str} | {elapsed:.1f}s"
        )

    print(f"\nSaved {len(results)} score files to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Score a LeRobot dataset with TOPReward.")
    parser.add_argument("--dataset", default=DATASET_REPO_ID or None,
                        help="LeRobot dataset repo_id (HuggingFace)")
    parser.add_argument("--dataset-root", default=DATASET_ROOT,
                        help="Local dataset root path")
    parser.add_argument("--instruction", default=INSTRUCTION,
                        help="Task instruction string")
    parser.add_argument("--camera-key", default=CAMERA_KEY,
                        help="Camera observation key in dataset")
    parser.add_argument("--eval-frames", type=int, default=NUM_EVAL_FRAMES,
                        help="Number of prefix evaluations per episode")
    parser.add_argument("--max-frames-per-prefix", type=int, default=MAX_FRAMES_PER_PREFIX,
                        help="Max frames subsampled per VLM call")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "scores",
                        help="Output directory for score JSONs")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy scorer (no model loading, for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("TOPReward Dataset Scoring")
    print("=" * 60)
    print(f"Dataset:     {args.dataset or args.dataset_root}")
    print(f"Instruction: {args.instruction}")
    print(f"Camera key:  {args.camera_key}")
    print(f"Eval frames: {args.eval_frames}")
    print(f"Output:      {args.output}")
    print(f"Dummy mode:  {args.dummy}")
    print()

    score_dataset(
        repo_id=args.dataset,
        root=args.dataset_root,
        instruction=args.instruction,
        camera_key=args.camera_key,
        output_dir=args.output,
        num_eval_frames=args.eval_frames,
        max_frames_per_prefix=args.max_frames_per_prefix,
        dummy=args.dummy,
    )


if __name__ == "__main__":
    main()
