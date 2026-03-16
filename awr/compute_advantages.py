"""
Step 2: Convert TOPReward scores to per-step advantage weights.

Algorithm:
  1. Load all episode score JSONs
  2. Interpolate K scored prefixes to per-frame progress values
  3. Compute per-step deltas: delta[t] = progress[t] - progress[t-1]
  4. Subtract dataset mean delta (centers advantages around 0)
  5. weight[t] = clip(TAU * exp(advantage[t]), WEIGHT_CLIP_MIN, DELTA_MAX)

Usage:
    python -m awr.compute_advantages \
        --scores-dir awr/outputs/scores/ \
        --output awr/outputs/advantages/advantages.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from awr.config import (
    DELTA_MAX,
    OUTPUT_DIR,
    SUBTRACT_MEAN,
    TAU,
    WEIGHT_CLIP_MIN,
)


def interpolate_to_per_frame(prefix_lengths: list, normalized: list, num_frames: int) -> np.ndarray:
    """Linearly interpolate K scored prefix values to all `num_frames` frames."""
    all_frame_indices = np.arange(num_frames)
    return np.interp(all_frame_indices, prefix_lengths, normalized)


def compute_episode_deltas(per_frame_progress: np.ndarray) -> np.ndarray:
    """delta[t] = progress[t] - progress[t-1]; delta[0] = progress[0]."""
    deltas = np.diff(per_frame_progress, prepend=0.0)
    return deltas


def compute_weights(
    deltas: np.ndarray,
    mean_delta: float,
    tau: float = TAU,
    delta_max: float = DELTA_MAX,
    clip_min: float = WEIGHT_CLIP_MIN,
    subtract_mean: bool = SUBTRACT_MEAN,
) -> np.ndarray:
    advantages = deltas - mean_delta if subtract_mean else deltas
    weights = tau * np.exp(advantages)
    return np.clip(weights, clip_min, delta_max)


def load_scores(scores_dir: Path) -> list[dict]:
    score_files = sorted(scores_dir.glob("episode_*.json"))
    if not score_files:
        raise FileNotFoundError(f"No episode_*.json files found in {scores_dir}")

    episodes = []
    for path in score_files:
        with open(path) as f:
            episodes.append(json.load(f))

    episodes.sort(key=lambda e: e["episode_id"])
    return episodes


def compute_advantages(
    scores_dir: Path,
    output_path: Path,
    tau: float = TAU,
    delta_max: float = DELTA_MAX,
    clip_min: float = WEIGHT_CLIP_MIN,
    subtract_mean: bool = SUBTRACT_MEAN,
) -> dict:
    print(f"Loading scores from {scores_dir}...")
    episodes = load_scores(scores_dir)
    print(f"  {len(episodes)} episodes loaded")

    # Pass 1: compute per-frame progress and deltas for every episode
    per_episode_progress = []
    per_episode_deltas = []

    for ep in episodes:
        progress = interpolate_to_per_frame(
            ep["prefix_lengths"], ep["normalized"], ep["num_frames"]
        )
        deltas = compute_episode_deltas(progress)
        per_episode_progress.append(progress)
        per_episode_deltas.append(deltas)

    # Dataset-level mean delta (used for centering)
    all_deltas = np.concatenate(per_episode_deltas)
    mean_delta = float(all_deltas.mean())
    print(f"  Dataset mean delta: {mean_delta:.6f}")

    # Pass 2: compute weights
    episode_records = []
    all_weights = []

    for ep, progress, deltas in zip(episodes, per_episode_progress, per_episode_deltas):
        weights = compute_weights(
            deltas, mean_delta,
            tau=tau, delta_max=delta_max, clip_min=clip_min,
            subtract_mean=subtract_mean,
        )
        all_weights.append(weights)

        episode_records.append({
            "episode_id": ep["episode_id"],
            "num_frames": ep["num_frames"],
            "voc": ep["voc"],
            "weights": weights.tolist(),
            "per_frame_progress": progress.tolist(),
            "per_frame_deltas": deltas.tolist(),
        })

    # Global weight stats
    flat_weights = np.concatenate(all_weights)
    stats = {
        "mean_weight": float(flat_weights.mean()),
        "std_weight": float(flat_weights.std()),
        "min_weight": float(flat_weights.min()),
        "max_weight": float(flat_weights.max()),
        "pct_above_1": float((flat_weights > 1.0).mean()),
        "pct_below_1": float((flat_weights < 1.0).mean()),
    }

    print(f"  Weight stats: mean={stats['mean_weight']:.3f}, "
          f"std={stats['std_weight']:.3f}, "
          f"range=[{stats['min_weight']:.3f}, {stats['max_weight']:.3f}]")
    print(f"  Above 1.0: {stats['pct_above_1']*100:.1f}%  "
          f"Below 1.0: {stats['pct_below_1']*100:.1f}%")

    if abs(stats["mean_weight"] - 1.0) > 0.5:
        print("  WARNING: mean weight is far from 1.0. Check TAU or SUBTRACT_MEAN setting.")

    output = {
        "mean_delta": mean_delta,
        "tau": tau,
        "delta_max": delta_max,
        "weight_clip_min": clip_min,
        "subtract_mean": subtract_mean,
        "episodes": episode_records,
        "stats": stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved advantages to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Compute per-step advantage weights from TOPReward scores.")
    parser.add_argument("--scores-dir", type=Path, default=OUTPUT_DIR / "scores",
                        help="Directory with episode_*.json score files")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "advantages" / "advantages.json",
                        help="Output path for advantages JSON")
    parser.add_argument("--tau", type=float, default=TAU,
                        help="Weight scaling factor (paper default: 2.0)")
    parser.add_argument("--delta-max", type=float, default=DELTA_MAX,
                        help="Max weight (paper default: 2.0)")
    parser.add_argument("--clip-min", type=float, default=WEIGHT_CLIP_MIN,
                        help="Min weight (prevents zero-gradient)")
    parser.add_argument("--no-subtract-mean", action="store_true",
                        help="Disable mean delta subtraction")
    args = parser.parse_args()

    print("=" * 60)
    print("AWR: Compute Advantage Weights")
    print("=" * 60)
    print(f"Scores dir:    {args.scores_dir}")
    print(f"Output:        {args.output}")
    print(f"TAU:           {args.tau}")
    print(f"DELTA_MAX:     {args.delta_max}")
    print(f"Clip min:      {args.clip_min}")
    print(f"Subtract mean: {not args.no_subtract_mean}")
    print()

    compute_advantages(
        scores_dir=args.scores_dir,
        output_path=args.output,
        tau=args.tau,
        delta_max=args.delta_max,
        clip_min=args.clip_min,
        subtract_mean=not args.no_subtract_mean,
    )


if __name__ == "__main__":
    main()
