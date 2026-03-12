"""
Dark-themed matplotlib plotting for TOPReward simulation results.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Theme colors
BG = "#0a0b0f"
SURFACE = "#12131a"
TEXT = "#e8e9ed"
TEXT_DIM = "#6b6f82"
GRID = "#1e2030"
ACCENT = "#22d3a7"
SECONDARY = "#6366f1"
WARNING = "#f59e0b"
DANGER = "#ef4444"

# Set rcParams
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM,
    "grid.color": GRID,
    "grid.alpha": 0.5,
    "font.family": "monospace",
    "font.size": 10,
})

_SAVE_KW = dict(dpi=150, facecolor=BG, bbox_inches="tight")


def plot_progress_curve(normalized, prefix_lengths, instruction, voc, save_path):
    """Plot normalized progress curve with VOC annotation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prefix_lengths, normalized, color=ACCENT, linewidth=2, marker="o", markersize=4)
    ax.fill_between(prefix_lengths, normalized, alpha=0.15, color=ACCENT)
    ax.set_xlabel("Frame (prefix length)")
    ax.set_ylabel("Normalized Progress")
    ax.set_title(f"Progress Curve", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # VOC annotation
    ax.text(
        0.98, 0.05, f"VOC = {voc:.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11, fontweight="bold", color=ACCENT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=ACCENT, alpha=0.9),
    )

    # Instruction as subtitle
    ax.text(
        0.5, 1.02, f'"{instruction}"',
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, color=TEXT_DIM, style="italic",
    )

    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_avg_pred_vs_gt(all_results, save_path):
    """Line plot: mean GT vs mean predicted completion % across episodes, with std band."""
    # All episodes have the same number of prefixes (k eval frames).
    # Align by prefix index (sorted by GT completion within each episode).
    n_prefixes = min(len(r["prefix_lengths"]) for r in all_results)

    gt_matrix = []   # shape (n_episodes, n_prefixes)
    pred_matrix = []

    for r in all_results:
        n_frames = r["num_frames"]
        gt_pcts = [100.0 * p / n_frames for p in r["prefix_lengths"][:n_prefixes]]
        pred_pcts = [100.0 * n for n in r["normalized"][:n_prefixes]]
        gt_matrix.append(gt_pcts)
        pred_matrix.append(pred_pcts)

    gt_arr = np.array(gt_matrix)    # (episodes, prefixes)
    pred_arr = np.array(pred_matrix)

    mean_gt = gt_arr.mean(axis=0)
    mean_pred = pred_arr.mean(axis=0)
    std_pred = pred_arr.std(axis=0)

    mean_voc = np.mean([r["voc"] for r in all_results])
    n_eps = len(all_results)
    xs = np.arange(n_prefixes)

    fig, ax = plt.subplots(figsize=(10, 6))

    # GT line
    ax.plot(xs, mean_gt, color="#4a9eff", linewidth=2, marker="o", markersize=5, label="Mean GT", zorder=3)
    # Predicted line + std band
    ax.plot(xs, mean_pred, color="#ff6b4a", linewidth=2, marker="s", markersize=5, label="Mean Predicted", zorder=3)
    ax.fill_between(xs, mean_pred - std_pred, mean_pred + std_pred, color="#ff6b4a", alpha=0.15, label="Pred std", zorder=2)

    ax.set_xlabel("Frame (sorted by GT completion)", fontsize=11)
    ax.set_ylabel("Task Completion %", fontsize=11)
    ax.set_title(f"Average Across {n_eps} Episodes (Mean VOC={mean_voc:.3f})", fontsize=13, fontweight="bold")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRID, loc="upper left")

    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_avg_pred_vs_gt(all_results, save_path):
    """Line plot: mean GT vs mean predicted completion % across episodes, with std band."""
    n_prefixes = min(len(r["prefix_lengths"]) for r in all_results)

    gt_matrix = []
    pred_matrix = []

    for r in all_results:
        n_frames = r["num_frames"]
        gt_pcts = [100.0 * p / n_frames for p in r["prefix_lengths"][:n_prefixes]]
        pred_pcts = [100.0 * n for n in r["normalized"][:n_prefixes]]
        gt_matrix.append(gt_pcts)
        pred_matrix.append(pred_pcts)

    gt_arr = np.array(gt_matrix)
    pred_arr = np.array(pred_matrix)

    mean_gt = gt_arr.mean(axis=0)
    mean_pred = pred_arr.mean(axis=0)
    std_pred = pred_arr.std(axis=0)

    mean_voc = np.mean([r["voc"] for r in all_results])
    n_eps = len(all_results)
    xs = np.arange(n_prefixes)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(xs, mean_gt, color="#4a9eff", linewidth=2, marker="o", markersize=5, label="Mean GT", zorder=3)
    ax.plot(xs, mean_pred, color="#ff6b4a", linewidth=2, marker="s", markersize=5, label="Mean Predicted", zorder=3)
    ax.fill_between(xs, mean_pred - std_pred, mean_pred + std_pred, color="#ff6b4a", alpha=0.15, label="Pred std", zorder=2)

    ax.set_xlabel("Frame (sorted by GT completion)", fontsize=11)
    ax.set_ylabel("Task Completion %", fontsize=11)
    ax.set_title(f"Average Across {n_eps} Episodes (Mean VOC={mean_voc:.3f})", fontsize=13, fontweight="bold")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRID, loc="upper left")

    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_scatter(all_results, save_path):
    """Scatter: ground truth completion % vs predicted (reward-normalized) completion %.

    Each episode contributes one point per scored prefix. Ground truth completion
    is the prefix length as a fraction of total frames; predicted completion is
    the min-max normalized reward score (already 0-1).
    """
    from scipy import stats

    gt_all = []
    pred_all = []

    for r in all_results:
        n_frames = r["num_frames"]
        for plen, norm in zip(r["prefix_lengths"], r["normalized"]):
            gt_pct = 100.0 * plen / n_frames
            pred_pct = 100.0 * norm
            gt_all.append(gt_pct)
            pred_all.append(pred_pct)

    gt_arr = np.array(gt_all)
    pred_arr = np.array(pred_all)

    # Pearson correlation
    if len(gt_arr) > 2:
        r_val, _ = stats.pearsonr(gt_arr, pred_arr)
    else:
        r_val = 0.0

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(gt_arr, pred_arr, c=ACCENT, alpha=0.45, s=50, edgecolors="none", zorder=3)

    # Perfect prediction diagonal
    ax.plot([0, 100], [0, 100], linestyle="--", color=TEXT_DIM, linewidth=1.5, label="Perfect prediction", zorder=2)

    ax.set_xlabel("Ground Truth Completion %", fontsize=11)
    ax.set_ylabel("Predicted Completion %", fontsize=11)
    ax.set_title(f"All Episodes Scatter (n={len(gt_arr)}, r={r_val:.3f})", fontsize=13, fontweight="bold")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRID, loc="upper left")

    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_log_prob_curve(raw_scores, prefix_lengths, instruction, save_path):
    """Plot raw log-probability curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prefix_lengths, raw_scores, color=SECONDARY, linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Frame (prefix length)")
    ax.set_ylabel("log P(True)")
    ax.set_title("Log-Probability Curve", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax.text(
        0.5, 1.02, f'"{instruction}"',
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, color=TEXT_DIM, style="italic",
    )

    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_combined(normalized, raw_scores, prefix_lengths, instruction, voc,
                  detected_success, inference_ms, first_frame, last_frame, save_path):
    """Combined plot: frames on top, dual-axis chart on bottom."""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, height_ratios=[1, 2.5], hspace=0.3, wspace=0.05)

    # Top row: first and last frames
    ax_first = fig.add_subplot(gs[0, 0])
    ax_last = fig.add_subplot(gs[0, 1])

    if first_frame is not None:
        ax_first.imshow(np.asarray(first_frame))
    ax_first.set_title("Start", fontsize=10, color=TEXT_DIM)
    ax_first.axis("off")

    if last_frame is not None:
        ax_last.imshow(np.asarray(last_frame))
    ax_last.set_title("End", fontsize=10, color=TEXT_DIM)
    ax_last.axis("off")

    # Bottom row: dual-axis chart spanning both columns
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = ax1.twinx()

    # Normalized progress (teal, filled)
    ln1 = ax1.plot(prefix_lengths, normalized, color=ACCENT, linewidth=2, marker="o", markersize=3, label="Progress (norm)")
    ax1.fill_between(prefix_lengths, normalized, alpha=0.15, color=ACCENT)
    ax1.set_xlabel("Frame (prefix length)")
    ax1.set_ylabel("Normalized Progress", color=ACCENT)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor=ACCENT)

    # Raw log-prob (indigo, dashed)
    ln2 = ax2.plot(prefix_lengths, raw_scores, color=SECONDARY, linewidth=2, linestyle="--", marker="s", markersize=3, label="log P(True)")
    ax2.set_ylabel("log P(True)", color=SECONDARY)
    ax2.tick_params(axis="y", labelcolor=SECONDARY)

    ax1.grid(True, alpha=0.3)

    # Legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", fontsize=8, facecolor=SURFACE, edgecolor=GRID)

    # Info text box
    success_str = "YES" if detected_success else "NO"
    success_color = ACCENT if detected_success else DANGER
    info = f"VOC: {voc:.3f}  |  Success: {success_str}  |  Avg inference: {inference_ms:.1f}ms/prefix"
    ax1.text(
        0.5, -0.12, info,
        transform=ax1.transAxes, ha="center", va="top",
        fontsize=9, color=TEXT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=success_color, alpha=0.9),
    )

    # Title
    fig.suptitle(f'"{instruction}"', fontsize=10, color=TEXT_DIM, style="italic", y=0.98)

    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_summary_voc(voc_scores, labels, save_path):
    """Horizontal bar chart of VOC scores across episodes."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))

    colors = [ACCENT if v > 0.5 else WARNING if v > 0 else DANGER for v in voc_scores]
    bars = ax.barh(range(len(labels)), voc_scores, color=colors, height=0.6, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("VOC (Spearman rho)")
    ax.set_title("VOC Scores by Episode", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Mean line
    mean_voc = np.mean(voc_scores)
    ax.axvline(mean_voc, color=TEXT_DIM, linestyle="--", linewidth=1.5)
    ax.text(mean_voc, len(labels) - 0.5, f" mean={mean_voc:.3f}", color=TEXT_DIM, fontsize=9, va="bottom")

    ax.set_xlim(-1.05, 1.05)
    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_comparison(all_normalized, all_prefix_lengths, all_labels, save_path):
    """Overlaid normalized progress curves for multiple episodes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis
    n = len(all_normalized)

    for i, (norm, plen, label) in enumerate(zip(all_normalized, all_prefix_lengths, all_labels)):
        color = cmap(i / max(n - 1, 1))
        ax.plot(plen, norm, color=color, linewidth=1.5, alpha=0.8, label=label)

    ax.set_xlabel("Frame (prefix length)")
    ax.set_ylabel("Normalized Progress")
    ax.set_title("Progress Curves Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, facecolor=SURFACE, edgecolor=GRID, loc="upper left", ncol=2)

    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)


def plot_scatter(all_results, save_path):
    """Scatter plot: VOC vs avg log-prob, colored by sim success."""
    fig, ax = plt.subplots(figsize=(10, 6))

    vocs = [r["voc"] for r in all_results]
    log_probs = [r["avg_log_prob"] for r in all_results]
    sim_ok = [r["sim_success"] for r in all_results]
    reward_ok = [r["reward_success"] for r in all_results]

    for i, (v, lp, sok, rok) in enumerate(zip(vocs, log_probs, sim_ok, reward_ok)):
        color = ACCENT if sok else DANGER
        marker = "o" if rok else "x"
        ax.scatter(v, lp, c=color, marker=marker, s=80, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(f"Ep {i+1}", (v, lp), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color=TEXT_DIM)

    # Legend handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT, markersize=8, label="Sim OK"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DANGER, markersize=8, label="Sim FAIL"),
        Line2D([0], [0], marker="x", color=TEXT_DIM, markersize=8, linestyle="None", label="Reward FAIL"),
    ]
    ax.legend(handles=handles, fontsize=8, facecolor=SURFACE, edgecolor=GRID, loc="best")

    ax.set_xlabel("VOC (Spearman rho)")
    ax.set_ylabel("Avg log P(True)")
    ax.set_title("VOC vs Log-Probability", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, **_SAVE_KW)
    plt.close(fig)
