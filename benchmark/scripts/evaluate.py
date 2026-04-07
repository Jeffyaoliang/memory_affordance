"""
Evaluation script for Memory Affordance benchmark.

Metrics:
1. Frame Retrieval Accuracy: Did the model select the correct frame?
   - Acc@1: exact match
   - Acc@5: correct frame in top-5 predictions
2. Mask IoU: Intersection over Union between predicted and GT masks
3. Combined Score: Acc@1 * IoU (only counts if frame is correct)

Input format (model predictions):
{
    "episode_id": "P01_01",
    "predictions": [
        {
            "task_id": "P01_01_t0",
            "predicted_image_id": 7,       # or list for top-k
            "predicted_image_ids": [7, 3, 12, 5, 1],  # optional top-k
            "predicted_mask_path": "pred_masks/P01_01_t0.png"  # optional
        }
    ]
}

Usage:
    python evaluate.py \
        --benchmark_dir benchmark/final \
        --predictions_path results/model_predictions.json \
        --output_path results/evaluation_results.json
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path


def compute_mask_iou(pred_mask_path: str, gt_mask_path: str) -> float:
    """Compute IoU between predicted and ground truth masks."""
    try:
        from PIL import Image
        pred = np.array(Image.open(pred_mask_path).convert("L")) > 127
        gt = np.array(Image.open(gt_mask_path).convert("L")) > 127

        # Resize pred to gt size if different
        if pred.shape != gt.shape:
            pred_img = Image.open(pred_mask_path).convert("L").resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST
            )
            pred = np.array(pred_img) > 127

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return float(intersection / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_episode(benchmark_entry: dict, predictions: dict,
                     benchmark_dir: str) -> dict:
    """Evaluate predictions for a single episode."""
    task_results = []

    # Build lookup: task_id -> GT
    gt_tasks = {t["task_id"]: t for t in benchmark_entry["tasks"]}

    for pred in predictions.get("predictions", []):
        task_id = pred["task_id"]
        gt = gt_tasks.get(task_id)
        if gt is None:
            continue

        gt_image_id = gt["target_image_id"]

        # Frame retrieval
        pred_image_id = pred.get("predicted_image_id")
        pred_image_ids = pred.get("predicted_image_ids", [pred_image_id] if pred_image_id is not None else [])

        acc_at_1 = 1.0 if (len(pred_image_ids) > 0 and pred_image_ids[0] == gt_image_id) else 0.0
        acc_at_5 = 1.0 if gt_image_id in pred_image_ids[:5] else 0.0

        # Mask IoU (only if frame is correct and mask provided)
        iou = 0.0
        if acc_at_1 > 0 and pred.get("predicted_mask_path"):
            gt_mask_path = os.path.join(benchmark_dir, gt["mask_path"])
            if os.path.exists(gt_mask_path) and os.path.exists(pred["predicted_mask_path"]):
                iou = compute_mask_iou(pred["predicted_mask_path"], gt_mask_path)

        task_results.append({
            "task_id": task_id,
            "acc_at_1": acc_at_1,
            "acc_at_5": acc_at_5,
            "mask_iou": iou,
            "combined_score": acc_at_1 * iou if iou > 0 else 0.0,
            "difficulty": gt.get("difficulty", "unknown"),
        })

    return {
        "episode_id": benchmark_entry["episode_id"],
        "num_tasks": len(task_results),
        "avg_acc_at_1": np.mean([r["acc_at_1"] for r in task_results]) if task_results else 0,
        "avg_acc_at_5": np.mean([r["acc_at_5"] for r in task_results]) if task_results else 0,
        "avg_mask_iou": np.mean([r["mask_iou"] for r in task_results]) if task_results else 0,
        "avg_combined": np.mean([r["combined_score"] for r in task_results]) if task_results else 0,
        "task_results": task_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Memory Affordance predictions")
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions_path) as f:
        all_predictions = json.load(f)

    # Index predictions by episode_id
    if isinstance(all_predictions, list):
        pred_by_episode = {p["episode_id"]: p for p in all_predictions}
    else:
        pred_by_episode = {all_predictions["episode_id"]: all_predictions}

    # Load and evaluate each episode
    master_path = os.path.join(args.benchmark_dir, "benchmark_master.json")
    with open(master_path) as f:
        master = json.load(f)

    episode_results = []
    for ep_info in master["episodes"]:
        ep_id = ep_info["episode_id"]
        ep_benchmark_path = os.path.join(args.benchmark_dir, ep_id, "benchmark.json")
        if not os.path.exists(ep_benchmark_path):
            continue

        with open(ep_benchmark_path) as f:
            benchmark_entry = json.load(f)

        preds = pred_by_episode.get(ep_id)
        if preds is None:
            continue

        result = evaluate_episode(benchmark_entry, preds, args.benchmark_dir)
        episode_results.append(result)

    # Aggregate metrics
    all_task_results = [tr for er in episode_results for tr in er["task_results"]]
    total_tasks = len(all_task_results)

    # Overall metrics
    overall = {
        "num_episodes": len(episode_results),
        "total_tasks": total_tasks,
        "overall_acc_at_1": np.mean([r["acc_at_1"] for r in all_task_results]) if all_task_results else 0,
        "overall_acc_at_5": np.mean([r["acc_at_5"] for r in all_task_results]) if all_task_results else 0,
        "overall_mask_iou": np.mean([r["mask_iou"] for r in all_task_results]) if all_task_results else 0,
        "overall_combined": np.mean([r["combined_score"] for r in all_task_results]) if all_task_results else 0,
    }

    # Per-difficulty metrics
    difficulties = set(r["difficulty"] for r in all_task_results)
    per_difficulty = {}
    for diff in difficulties:
        diff_results = [r for r in all_task_results if r["difficulty"] == diff]
        per_difficulty[diff] = {
            "num_tasks": len(diff_results),
            "acc_at_1": np.mean([r["acc_at_1"] for r in diff_results]),
            "acc_at_5": np.mean([r["acc_at_5"] for r in diff_results]),
            "mask_iou": np.mean([r["mask_iou"] for r in diff_results]),
            "combined": np.mean([r["combined_score"] for r in diff_results]),
        }

    # Save results
    output = {
        "overall": {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in overall.items()},
        "per_difficulty": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()}
            for k, v in per_difficulty.items()
        },
        "per_episode": [
            {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in er.items() if k != "task_results"}
            for er in episode_results
        ],
    }

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("=" * 60)
    print("Memory Affordance Benchmark - Evaluation Results")
    print("=" * 60)
    print(f"Episodes: {overall['num_episodes']}  |  Tasks: {overall['total_tasks']}")
    print(f"  Acc@1:     {overall['overall_acc_at_1']:.4f}")
    print(f"  Acc@5:     {overall['overall_acc_at_5']:.4f}")
    print(f"  Mask IoU:  {overall['overall_mask_iou']:.4f}")
    print(f"  Combined:  {overall['overall_combined']:.4f}")
    print()
    for diff, metrics in per_difficulty.items():
        print(f"  [{diff}] n={metrics['num_tasks']}  Acc@1={metrics['acc_at_1']:.4f}  IoU={metrics['mask_iou']:.4f}")
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
