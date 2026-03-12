"""
Main CLI: run SO-ARM100 pick-and-place simulation, score with TOPReward, save plots.
"""

import argparse
import json
import pathlib
import time

import numpy as np
from PIL import Image

import config
from plotting import plot_avg_pred_vs_gt, plot_combined, plot_comparison, plot_log_prob_curve, plot_progress_curve, plot_scatter, plot_summary_voc
from reward_wrapper import RewardScorer, detect_success, score_trajectory
from sim_env import SO100PickPlaceEnv


def main():
    parser = argparse.ArgumentParser(description="TOPReward Sim — SO-100 Pick and Place")
    parser.add_argument("--policy", choices=["pick_place", "noisy_pick_place", "random"], default="pick_place")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-frames", type=int, default=config.NUM_EVAL_FRAMES)
    parser.add_argument("--output", type=str, default=config.OUTPUT_DIR)
    parser.add_argument("--randomize", action="store_true", default=True)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--instruction", type=str, default=config.TASK_INSTRUCTION)
    parser.add_argument("--dummy", action="store_true", help="Use dummy scorer (no GPU needed)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Model name")
    parser.add_argument("--4bit", dest="quant_4bit", action="store_true", default=True, help="4-bit NF4 quantization (~5GB VRAM, default)")
    parser.add_argument("--8bit", dest="quant_8bit", action="store_true", help="8-bit quantization (~9GB VRAM)")
    args = parser.parse_args()

    randomize = args.randomize and not args.no_randomize
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    print("=" * 60)
    print("  TOPReward Sim — SO-100 Pick and Place")
    print("=" * 60)
    print(f"  Policy:      {args.policy}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Eval frames: {args.eval_frames}")
    print(f"  Randomize:   {randomize}")
    print(f"  Instruction: {args.instruction}")
    quant = "dummy" if args.dummy else "4bit" if args.quant_4bit else "8bit"
    print(f"  Model:       {'dummy' if args.dummy else args.model} ({quant})")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    # Init scorer
    scorer = RewardScorer(
        model_name=args.model,
        load_in_4bit=args.quant_4bit,
        load_in_8bit=args.quant_8bit and not args.quant_4bit,
        dummy=args.dummy,
    )

    # Collect results across episodes
    all_vocs = []
    all_labels = []
    all_normalized = []
    all_prefix_lengths = []
    all_results = []

    env = SO100PickPlaceEnv()

    for ep in range(1, args.episodes + 1):
        ep_dir = output_dir / f"episode_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Episode {ep}/{args.episodes} ---")
        t0 = time.time()

        # Run simulation
        result = env.run_scripted_episode(policy=args.policy, randomize=randomize)
        sim_time = time.time() - t0
        print(f"  Sim: {result.num_steps} frames, success={result.success}, {sim_time:.1f}s")

        # Get PIL frames
        pil_frames = [Image.fromarray(f) for f in result.frames]

        # Score trajectory
        t0 = time.time()
        scores = score_trajectory(scorer, pil_frames, args.instruction, k=args.eval_frames)
        score_time = time.time() - t0

        # Detect success via reward
        detection = detect_success(scorer, pil_frames, args.instruction)

        print(f"  Score: VOC={scores['voc']:.3f}, {len(scores['prefix_lengths'])} prefixes, {score_time:.1f}s")
        print(f"  Detection: reward_success={detection['success']}, avg_log_prob={detection['avg_log_prob']:.3f}")

        # Save frames
        if result.frames:
            Image.fromarray(result.frames[0]).save(str(ep_dir / "frame_first.jpg"))
            Image.fromarray(result.frames[-1]).save(str(ep_dir / "frame_last.jpg"))

        # Save plots
        plot_progress_curve(
            scores["normalized"], scores["prefix_lengths"],
            args.instruction, scores["voc"],
            str(ep_dir / "progress_curve.png"),
        )
        plot_log_prob_curve(
            scores["raw_scores"], scores["prefix_lengths"],
            args.instruction,
            str(ep_dir / "log_prob_curve.png"),
        )

        first_pil = Image.fromarray(result.frames[0]) if result.frames else None
        last_pil = Image.fromarray(result.frames[-1]) if result.frames else None
        plot_combined(
            scores["normalized"], scores["raw_scores"], scores["prefix_lengths"],
            args.instruction, scores["voc"], detection["success"],
            scores["per_frame_ms"], first_pil, last_pil,
            str(ep_dir / "combined_plot.png"),
        )

        # Save result JSON
        ep_result = {
            "episode": ep,
            "policy": args.policy,
            "instruction": args.instruction,
            "sim_success": bool(result.success),
            "reward_success": bool(detection["success"]),
            "voc": float(scores["voc"]),
            "avg_log_prob": float(detection["avg_log_prob"]),
            "confidence": float(detection["confidence"]),
            "num_frames": len(result.frames),
            "prefix_lengths": [int(x) for x in scores["prefix_lengths"]],
            "raw_scores": [float(x) for x in scores["raw_scores"]],
            "normalized": [float(x) for x in scores["normalized"]],
            "rank_normalized": [float(x) for x in scores["rank_normalized"]],
            "per_frame_ms": float(scores["per_frame_ms"]),
            "total_score_ms": float(scores["total_ms"]),
            "sim_time_s": float(sim_time),
            "cube_initial_pos": result.cube_initial_pos.tolist(),
            "bin_pos": result.bin_pos.tolist(),
        }
        with open(ep_dir / "result.json", "w") as f:
            json.dump(ep_result, f, indent=2)

        # Collect for summary
        all_vocs.append(scores["voc"])
        label = f"Ep {ep} ({'OK' if result.success else 'FAIL'})"
        all_labels.append(label)
        all_normalized.append(scores["normalized"])
        all_prefix_lengths.append(scores["prefix_lengths"])
        all_results.append(ep_result)

    env.close()

    # Summary plots
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")

    plot_summary_voc(all_vocs, all_labels, str(output_dir / "summary_plot.png"))
    plot_comparison(all_normalized, all_prefix_lengths, all_labels, str(output_dir / "comparison_plot.png"))
    plot_scatter(all_results, str(output_dir / "scatter_plot.png"))
    plot_avg_pred_vs_gt(all_results, str(output_dir / "avg_pred_vs_gt.png"))

    # Summary JSON
    sim_successes = sum(1 for r in all_results if r["sim_success"])
    reward_successes = sum(1 for r in all_results if r["reward_success"])
    summary = {
        "policy": args.policy,
        "instruction": args.instruction,
        "num_episodes": args.episodes,
        "sim_success_rate": sim_successes / args.episodes,
        "reward_success_rate": reward_successes / args.episodes,
        "mean_voc": float(np.mean(all_vocs)),
        "std_voc": float(np.std(all_vocs)),
        "mean_per_frame_ms": float(np.mean([r["per_frame_ms"] for r in all_results])),
        "episodes": all_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"  Sim success rate:    {sim_successes}/{args.episodes} ({100*sim_successes/args.episodes:.0f}%)")
    print(f"  Reward success rate: {reward_successes}/{args.episodes} ({100*reward_successes/args.episodes:.0f}%)")
    print(f"  Mean VOC:            {np.mean(all_vocs):.3f} +/- {np.std(all_vocs):.3f}")
    print(f"  Mean inference:      {np.mean([r['per_frame_ms'] for r in all_results]):.1f} ms/prefix")
    print(f"\n  Results saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
