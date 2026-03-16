# AWR Fine-Tuning with TOPReward

Advantage-weighted regression (AWR) using TOPReward scores to improve LeRobot policies.

## How It Works

1. **Score** — TOPReward evaluates each demo frame-by-frame, producing per-episode progress curves.
2. **Advantages** — Progress deltas reveal which actions advanced the task. Deltas are converted to advantage weights: good actions > 1, bad/stalled actions < 1.
3. **Train** — The policy is fine-tuned with a weighted loss: it learns more from high-advantage steps and less from stalled or regressing ones.

## Quick Start

```bash
# 1. Score your dataset
python -m awr.score_dataset \
    --dataset "your_username/so100_pick_cube" \
    --instruction "Pick up the red cube and place it in the box"

# 2. Compute advantage weights
python -m awr.compute_advantages

# 3. Inspect before training (recommended)
python -m awr.inspect_scores

# 4. Train
python -m awr.weighted_trainer \
    --dataset "your_username/so100_pick_cube" \
    --checkpoint "your_username/act_so100_pick_cube"

# Or run everything at once:
python -m awr.pipeline \
    --dataset "your_username/so100_pick_cube" \
    --instruction "Pick up the red cube and place it in the box" \
    --checkpoint "your_username/act_so100_pick_cube"

# Score + advantages + plots only (no training):
python -m awr.pipeline \
    --dataset "your_username/so100_pick_cube" \
    --instruction "Pick up the red cube and place it in the box" \
    --skip-train

# Test with dummy scorer (no GPU needed):
python -m awr.pipeline \
    --dataset "your_username/so100_pick_cube" \
    --instruction "..." \
    --dummy --skip-train
```

## Output Structure

```
awr/outputs/
├── scores/
│   ├── episode_000.json     # Per-episode TOPReward scores
│   └── ...
├── advantages/
│   └── advantages.json      # Per-step weights for all episodes
├── plots/
│   ├── all_progress_curves.png
│   ├── weight_distribution.png
│   ├── weight_heatmap.png
│   └── dataset_summary.png
├── checkpoints/
│   └── awr_finetuned/       # Improved policy checkpoint
└── run_log.json
```

## Configuration

Edit `awr/config.py`. Key parameters:

| Parameter | Default | Notes |
|---|---|---|
| `TAU` | 2.0 | Weight scale factor (paper default) |
| `DELTA_MAX` | 2.0 | Max weight cap (paper default) |
| `WEIGHT_CLIP_MIN` | 0.1 | Min weight (prevents zero-gradient) |
| `SUBTRACT_MEAN` | True | Centers advantages around 0 — keep True |
| `LEARNING_RATE` | 1e-5 | Conservative for fine-tuning |
| `NUM_TRAIN_STEPS` | 5000 | AWR typically needs fewer steps than BC |
| `CAMERA_KEY` | `observation.image` | Inspect your dataset if unsure |

## Adapting to Your LeRobot Version

`weighted_trainer.py` contains lines marked `# ADAPT THIS` at version-dependent spots:

- **Policy import** — check which class and import path your version uses
- **`policy.forward(batch)` return type** — may be a scalar or per-sample tensor
- **`batch["index"]` key** — inspect `dataset[0].keys()` to find the global index field
- **`policy.save_pretrained()`** — some versions use `torch.save(state_dict, ...)`

To inspect your dataset:
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="your_username/so100_pick_cube")
print(ds[0].keys())
print(ds.episode_data_index)
```

## Iterating

After AWR fine-tuning, deploy the improved policy, collect new rollouts, score them, and run another round. Each round should further concentrate the learned behavior on successful trajectories.

```bash
# Round 2: point at new demos recorded with the finetuned policy
python -m awr.pipeline \
    --dataset "your_username/so100_pick_cube_v2" \
    --checkpoint awr/outputs/checkpoints/awr_finetuned/final \
    ...
```

## Sanity Checks (inspect before training)

After running `compute_advantages`, check `inspect_scores` plots:

- **Mean weight near 1.0** — if far off, adjust `TAU`
- **Weight std > 0.1** — if weights are too uniform, TOPReward may not be discriminating
- **Failed episodes appear red** in the heatmap — if they look green, scoring may be noisy
- **VOC > 0.5** on most episodes — low VOC means TOPReward didn't see progress in those demos
