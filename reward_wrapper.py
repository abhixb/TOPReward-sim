"""
Thin wrapper for TOPReward scoring.
Uses QwenClient (Qwen3-VL-8B) for real inference, falls back to dummy if unavailable.
"""

import math
import time

import numpy as np
from scipy import stats

from config import MAX_FRAMES_PER_PREFIX, NUM_EVAL_FRAMES, SUCCESS_LAST_N, SUCCESS_THRESHOLD


class RewardScorer:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                 load_in_4bit: bool = False, load_in_8bit: bool = False, dummy: bool = False):
        """
        Initialize TOPReward scorer.

        Args:
            model_name: HuggingFace model ID for Qwen3-VL.
            load_in_4bit: Use 4-bit quantization (~5GB VRAM).
            load_in_8bit: Use 8-bit quantization (~9GB VRAM, better quality).
            dummy: Force dummy mode (no model loading).
        """
        self._client = None
        self._model_loaded = False

        if dummy:
            print("RewardScorer: dummy mode (--dummy flag).")
            return

        quant = "8bit" if load_in_8bit else "4bit" if load_in_4bit else "none"
        try:
            from topreward.clients.qwen import QwenClient
            print(f"RewardScorer: loading {model_name} (quant={quant})...")
            self._client = QwenClient(
                model_name=model_name,
                load_in_4bit=load_in_4bit or load_in_8bit,
            )
            self._model_loaded = True
            print("RewardScorer: model loaded.")
        except Exception as e:
            print(f"RewardScorer: failed to load model ({e}). Using dummy scores.")

    def score_prefix(self, frames: list, instruction: str) -> dict:
        """
        Score a trajectory prefix using progress estimation.

        Instead of binary P("True"), asks the VLM to estimate task completion
        percentage (0-100), giving a continuous reward signal for intermediate states.

        Args:
            frames: list of PIL Images (already subsampled to <= MAX_FRAMES_PER_PREFIX)
            instruction: task description string

        Returns:
            {"log_prob": float, "prob": float, "inference_ms": float}
        """
        if not self._model_loaded:
            return self._dummy_score(frames)

        t0 = time.time()
        frame_arrays = [np.array(f) for f in frames]
        pil_frames = [self._to_pil(f) for f in frames]

        # Use progress estimation prompt
        progress = self._estimate_progress(pil_frames, instruction)
        elapsed_ms = (time.time() - t0) * 1000

        # Map percentage to a log-prob-like scale for compatibility
        # progress=0% -> log_prob ~ -22, progress=100% -> log_prob ~ -6
        clamped = max(0.0, min(100.0, progress))
        log_prob = -22.0 + (clamped / 100.0) * 16.0

        return {
            "log_prob": float(log_prob),
            "prob": float(math.exp(max(log_prob, -50))),
            "inference_ms": elapsed_ms,
            "progress_pct": float(clamped),
        }

    def _estimate_progress(self, pil_frames: list, instruction: str) -> float:
        """Ask VLM to estimate task completion percentage from frames."""
        import re
        from qwen_vl_utils import process_vision_info

        prompt = (
            f"You are evaluating a robot manipulation task. "
            f"The task instruction is: \"{instruction}\"\n\n"
            f"Look at the video frames showing the robot's trajectory so far. "
            f"Estimate what percentage of the task has been completed (0-100%).\n\n"
            f"Consider these stages:\n"
            f"- 0%: Robot hasn't started moving toward the object\n"
            f"- 20%: Robot is approaching/reaching toward the object\n"
            f"- 40%: Robot has made contact with or is grasping the object\n"
            f"- 60%: Robot has lifted the object\n"
            f"- 80%: Robot is moving the object toward the target location\n"
            f"- 100%: Object has been placed at the target location\n\n"
            f"Reply with ONLY a number between 0 and 100."
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames, "fps": 2.0},
                {"type": "text", "text": prompt},
            ],
        }]

        prompt_chat = self._client.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        import torch
        inputs = self._client.processor(
            text=[prompt_chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        self._client.model.eval()
        with torch.no_grad():
            output_ids = self._client.model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.0,
                do_sample=False,
            )

        response = self._client.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Extract the generated part after the prompt
        prompt_text = self._client.processor.batch_decode(
            inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        generated = response[len(prompt_text):].strip() if response.startswith(prompt_text) else response.strip()

        # Parse percentage from response
        match = re.search(r'(\d+(?:\.\d+)?)', generated)
        if match:
            return float(match.group(1))
        return 0.0

    @staticmethod
    def _to_pil(frame):
        """Convert PIL or numpy frame to PIL."""
        from PIL import Image
        if isinstance(frame, Image.Image):
            return frame
        return Image.fromarray(np.array(frame))

    @staticmethod
    def _dummy_score(frames: list) -> dict:
        """Dummy scorer: simulates increasing log-prob with noise."""
        import random
        t0 = time.time()
        progress = len(frames) / 16.0
        fake_log_prob = -6.0 + progress * 5.5 + random.gauss(0, 0.25)
        fake_log_prob = min(fake_log_prob, -0.01)
        return {
            "log_prob": fake_log_prob,
            "prob": math.exp(fake_log_prob),
            "inference_ms": (time.time() - t0) * 1000,
        }


def score_trajectory(
    scorer: RewardScorer,
    all_frames_pil: list,
    instruction: str,
    k: int = NUM_EVAL_FRAMES,
    max_frames_per_prefix: int = MAX_FRAMES_PER_PREFIX,
) -> dict:
    """Score a full trajectory at k uniformly-spaced prefix lengths."""
    T = len(all_frames_pil)
    if T == 0:
        return {"raw_scores": [], "normalized": [], "prefix_lengths": [], "voc": 0.0, "per_frame_ms": 0.0, "total_ms": 0.0}

    indices = np.linspace(0, T - 1, k, dtype=int)
    prefix_lengths = sorted(set(int(i) + 1 for i in indices))

    raw_scores = []
    total_ms = 0.0

    for plen in prefix_lengths:
        prefix_frames = all_frames_pil[:plen]
        if len(prefix_frames) > max_frames_per_prefix:
            step = len(prefix_frames) / max_frames_per_prefix
            prefix_frames = [prefix_frames[int(i * step)] for i in range(max_frames_per_prefix)]
        result = scorer.score_prefix(prefix_frames, instruction)
        raw_scores.append(result["log_prob"])
        total_ms += result["inference_ms"]

    # Min-max normalize
    raw_arr = np.array(raw_scores)
    rmin, rmax = raw_arr.min(), raw_arr.max()
    normalized = ((raw_arr - rmin) / (rmax - rmin + 1e-8)).tolist()

    # Rank normalize (percentile-based, spreads values evenly)
    if len(raw_scores) > 1:
        ranks = stats.rankdata(raw_scores, method="average")
        rank_normalized = ((ranks - 1) / (len(ranks) - 1)).tolist()
    else:
        rank_normalized = [1.0] if raw_scores else []

    # VOC via Spearman rank correlation
    if len(raw_scores) > 2:
        rho, _ = stats.spearmanr(range(len(raw_scores)), raw_scores)
        voc = 0.0 if np.isnan(rho) else float(rho)
    else:
        voc = 0.0

    per_frame_ms = total_ms / len(prefix_lengths) if prefix_lengths else 0.0

    return {
        "raw_scores": raw_scores,
        "normalized": normalized,
        "rank_normalized": rank_normalized,
        "prefix_lengths": prefix_lengths,
        "voc": voc,
        "per_frame_ms": per_frame_ms,
        "total_ms": total_ms,
    }


def detect_success(
    scorer: RewardScorer,
    all_frames_pil: list,
    instruction: str,
    threshold: float = SUCCESS_THRESHOLD,
    last_n: int = SUCCESS_LAST_N,
    max_frames_per_prefix: int = MAX_FRAMES_PER_PREFIX,
) -> dict:
    """Detect success by scoring the last_n prefixes near trajectory end."""
    T = len(all_frames_pil)
    if T == 0:
        return {"success": False, "avg_log_prob": float("-inf"), "confidence": 0.0}

    end_indices = np.linspace(max(0, T - T // 4), T - 1, last_n, dtype=int)
    end_indices = sorted(set(int(i) + 1 for i in end_indices))

    log_probs = []
    for plen in end_indices:
        prefix_frames = all_frames_pil[:plen]
        if len(prefix_frames) > max_frames_per_prefix:
            step = len(prefix_frames) / max_frames_per_prefix
            prefix_frames = [prefix_frames[int(i * step)] for i in range(max_frames_per_prefix)]
        result = scorer.score_prefix(prefix_frames, instruction)
        log_probs.append(result["log_prob"])

    avg_log_prob = float(np.mean(log_probs))
    success = avg_log_prob > threshold
    confidence = 1.0 / (1.0 + math.exp(-(avg_log_prob - threshold)))

    return {
        "success": success,
        "avg_log_prob": avg_log_prob,
        "confidence": confidence,
    }


if __name__ == "__main__":
    from PIL import Image

    print("Testing reward_wrapper with dummy gradient frames...")
    scorer = RewardScorer(dummy=True)

    frames = []
    for i in range(30):
        v = int(255 * i / 29)
        img = Image.new("RGB", (320, 240), (v, v, v))
        frames.append(img)

    instruction = "Pick up the red cube and place it in the box"

    result = score_trajectory(scorer, frames, instruction, k=10)
    print(f"Prefix lengths: {result['prefix_lengths']}")
    print(f"Raw scores: {[f'{s:.3f}' for s in result['raw_scores']]}")
    print(f"Normalized:  {[f'{s:.3f}' for s in result['normalized']]}")
    print(f"VOC: {result['voc']:.3f}")
    print(f"Total time: {result['total_ms']:.1f}ms")

    det = detect_success(scorer, frames, instruction)
    print(f"\nSuccess detection: {det['success']} (avg_log_prob={det['avg_log_prob']:.3f}, confidence={det['confidence']:.3f})")
