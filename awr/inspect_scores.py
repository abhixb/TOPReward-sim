"""
Utility: Visualize TOPReward scores and advantage weights before training.

Generates four plots:
  1. all_progress_curves.png   — overlaid per-episode progress curves
  2. weight_distribution.png   — histogram of all advantage weights
  3. weight_heatmap.png        — per-episode weight heatmap (green=high, red=low)
  4. dataset_summary.png       — VOC bar chart per episode

Usage:
    python -m awr.inspect_scores \
        --scores-dir awr/outputs/scores/ \
        --advantages awr/outputs/advantages/advantages.json \
        --output awr/outputs/plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from awr.config import OUTPUT_DIR

# ─── Dark theme (mirrors parent plotting.py) ──────────────────────────────────
BG      = "#0a0b0f"
SURFACE = "#12131a"
TEXT    = "#e8e9ed"
TEXT_DIM = "#6b6f82"
GRID    = "#1e2030"
ACCENT  = "#22d3a7"
SECONDARY = "#6366f1"
WARNING = "#f59e0b"
DANGER  = "#ef4444"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "text.color":       TEXT,
    "xtick.color":      TEXT_DIM,
    "ytick.color":      TEXT_DIM,
    "grid.color":       GRID,
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "font.size":        10,
})
_SAVE_KW = dict(dpi=150, facecolor=BG, bbox_inches="tight")


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_scores(scores_dir: Path) -> list[dict]:
    files = sorted(scores_dir.glob("episode_*.json"))
    if not files:
        raise FileNotFoundError(f"No episode_*.json in {scores_dir}")
    episodes = []
    for p in files:
        with open(p) as f:
            episodes.append(json.load(f))
    return sorted(episodes, key=lambda e: e["episode_id"])


def load_advantages(advantages_path: Path) -> dict:
    with open(advantages_path) as f:
        return json.load(f)


# ─── Plot 1: All progress curves ─────────────────────────────────────────────

def plot_all_progress_curves(episodes: list[dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.plasma
    n = len(episodes)
    vocs = [ep["voc"] for ep in episodes]
    voc_norm = (np.array(vocs) - min(vocs)) / (max(vocs) - min(vocs) + 1e-8)

    for ep, vn in zip(episodes, voc_norm):
        color = cmap(vn)
        ax.plot(ep["prefix_lengths"], ep["normalized"],
                color=color, linewidth=1.5, alpha=0.75)

    # Color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(vocs), max(vocs)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("VOC", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    ax.set_xlabel("Frame (prefix length)")
    ax.set_ylabel("Normalized Progress")
    ax.set_title(f"All Progress Curves ({n} episodes)", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    path = output_dir / "all_progress_curves.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 2: Weight distribution histogram ────────────────────────────────────

def plot_weight_distribution(advantages: dict, output_dir: Path):
    all_weights = []
    for ep in advantages["episodes"]:
        all_weights.extend(ep["weights"])
    w = np.array(all_weights)

    fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = min(80, max(20, len(w) // 50))
    counts, bins, patches = ax.hist(w, bins=n_bins, color=ACCENT, alpha=0.8, edgecolor=BG)

    # Color patches by position relative to 1.0
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(ACCENT if left >= 1.0 else DANGER)
        patch.set_alpha(0.75)

    ax.axvline(1.0, color=TEXT_DIM, linestyle="--", linewidth=1.5, label="weight=1.0")
    ax.axvline(w.mean(), color=WARNING, linestyle="-", linewidth=2,
               label=f"mean={w.mean():.3f}")

    stats = advantages["stats"]
    info = (
        f"mean={stats['mean_weight']:.3f}  std={stats['std_weight']:.3f}\n"
        f"above 1.0: {stats['pct_above_1']*100:.1f}%  "
        f"below 1.0: {stats['pct_below_1']*100:.1f}%"
    )
    ax.text(0.97, 0.95, info, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID, alpha=0.9))

    ax.set_xlabel("Advantage Weight")
    ax.set_ylabel("Count")
    ax.set_title("Weight Distribution (TAU={}, clip=[{}, {}])".format(
        advantages["tau"], advantages["weight_clip_min"], advantages["delta_max"]
    ), fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRID)

    path = output_dir / "weight_distribution.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 3: Per-episode weight heatmap ───────────────────────────────────────

def plot_weight_heatmap(advantages: dict, output_dir: Path):
    eps = advantages["episodes"]
    n = len(eps)

    # Normalise all episodes to same length via resampling for display
    display_len = 200
    heatmap = np.zeros((n, display_len))
    labels = []
    mean_weights = []

    for i, ep in enumerate(eps):
        w = np.array(ep["weights"])
        x_src = np.linspace(0, 1, len(w))
        x_dst = np.linspace(0, 1, display_len)
        heatmap[i] = np.interp(x_dst, x_src, w)
        labels.append(f"ep{ep['episode_id']:03d} voc={ep['voc']:+.2f}")
        mean_weights.append(float(np.mean(w)))

    # Sort by mean weight descending (best episodes at top)
    order = np.argsort(mean_weights)[::-1]
    heatmap = heatmap[order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, max(4, n * 0.45)))
    clip_min = advantages["weight_clip_min"]
    clip_max = advantages["delta_max"]
    im = ax.imshow(
        heatmap, aspect="auto", cmap="RdYlGn",
        vmin=clip_min, vmax=clip_max, interpolation="nearest",
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Frame (normalised to 200 bins)")
    ax.set_title("Per-Episode Advantage Weights (green=high, red=low)", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("Weight", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    path = output_dir / "weight_heatmap.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 4: Dataset summary (VOC bar chart) ──────────────────────────────────

def plot_dataset_summary(episodes: list[dict], output_dir: Path):
    vocs = [ep["voc"] for ep in episodes]
    labels = [f"ep{ep['episode_id']:03d}" for ep in episodes]
    mean_voc = np.mean(vocs)

    colors = [ACCENT if v > 0.5 else WARNING if v > 0 else DANGER for v in vocs]

    fig, ax = plt.subplots(figsize=(max(8, len(episodes) * 0.6), 5))
    ax.bar(range(len(vocs)), vocs, color=colors, alpha=0.85, edgecolor=BG, linewidth=0.5)
    ax.axhline(mean_voc, color=TEXT_DIM, linestyle="--", linewidth=1.5,
               label=f"mean VOC={mean_voc:.3f}")
    ax.axhline(0, color=GRID, linewidth=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("VOC (Spearman rho)")
    ax.set_title(f"Dataset Summary — VOC per Episode (n={len(episodes)})", fontsize=12, fontweight="bold")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRID)

    path = output_dir / "dataset_summary.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_inspect(scores_dir: Path, advantages_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from {scores_dir}...")
    episodes = load_scores(scores_dir)
    print(f"  {len(episodes)} episodes")

    print(f"Loading advantages from {advantages_path}...")
    advantages = load_advantages(advantages_path)

    print("\nGenerating plots...")
    plot_all_progress_curves(episodes, output_dir)
    plot_weight_distribution(advantages, output_dir)
    plot_weight_heatmap(advantages, output_dir)
    plot_dataset_summary(episodes, output_dir)
    print(f"\nAll plots saved to {output_dir}")

    # Sanity check printout
    stats = advantages["stats"]
    print("\nSanity checks:")
    if abs(stats["mean_weight"] - 1.0) < 0.3:
        print(f"  [OK] Mean weight {stats['mean_weight']:.3f} is near 1.0")
    else:
        print(f"  [WARN] Mean weight {stats['mean_weight']:.3f} is far from 1.0 — adjust TAU")

    if stats["std_weight"] > 0.1:
        print(f"  [OK] Weight std {stats['std_weight']:.3f} shows spread")
    else:
        print(f"  [WARN] Weight std {stats['std_weight']:.3f} is low — weights are too uniform")

    vocs = [ep["voc"] for ep in episodes]
    n_positive = sum(v > 0.5 for v in vocs)
    print(f"  Episodes with VOC > 0.5 (likely success): {n_positive}/{len(vocs)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect TOPReward scores and advantage weights.")
    parser.add_argument("--scores-dir", type=Path, default=OUTPUT_DIR / "scores")
    parser.add_argument("--advantages", type=Path,
                        default=OUTPUT_DIR / "advantages" / "advantages.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "plots")
    args = parser.parse_args()

    print("=" * 60)
    print("AWR: Inspect Scores & Weights")
    print("=" * 60)
    run_inspect(args.scores_dir, args.advantages, args.output)


if __name__ == "__main__":
    main()
