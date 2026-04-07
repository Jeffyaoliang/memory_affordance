"""
Generate ground-truth segmentation masks for Memory Affordance benchmark.

For each valid QA pair (output of qa_uniqueness_check.py), this script:
1. Loads the target image (the keyframe specified by image_id)
2. Runs Rex-Omni on (image, target_object) to detect bbox + point
3. Runs SAM2 on (image, bbox, point) to produce a precise mask
4. Saves mask.png + image_with_mask.png + metadata for the benchmark

This script is a thin adapter on top of Affordance_Annotator/label-sam2.py,
operating on episode-level QA results instead of per-image results.

REQUIREMENTS (GPU):
    - Rex-Omni 7B (~16GB VRAM)
    - SAM2 hiera-large (~2GB VRAM)
    - flash-attention 2.7.4
    - PyTorch 2.5.1 + CUDA 12.1
    - ≥24GB total VRAM recommended

Usage:
    python generate_masks.py \
        --qa_check_dir results/episode_qa \
        --keyframes_root data/keyframes/ego4d_test \
        --output_dir results/episode_masks \
        --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ANNOTATOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Affordance_Annotator")
sys.path.insert(0, ANNOTATOR_DIR)


def load_models(device: str = "cuda"):
    """Lazy import & load Rex-Omni + SAM2. GPU required."""
    import torch
    from rex_omni import RexOmniWrapper
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("Loading Rex-Omni...")
    rex = RexOmniWrapper(
        model_path="IDEA-Research/Rex-Omni",
        backend="transformers",
        max_tokens=4096,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    print("Loading SAM2...")
    sam2 = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
    return rex, sam2


def detect_object(rex_model, image: Image.Image, object_name: str) -> dict:
    """Run Rex-Omni detection + pointing for one object on one image.

    API contract (from Affordance_Annotator/label-sam2.py reference):
        result = rex_model.inference(images=image, task="detection", categories=[cat])[0]
        result["extracted_predictions"][cat] = [{"coords": [...]}, ...]

    Returns:
        {bboxes: [[x1,y1,x2,y2], ...], points: [[x,y], ...]} or empty dict on failure
    """
    cat = object_name
    bboxes, points = [], []

    try:
        det_result = rex_model.inference(images=image, task="detection",
                                          categories=[cat])
        preds = det_result[0].get("extracted_predictions", {})
        if cat in preds:
            for pred in preds[cat]:
                bboxes.append(pred["coords"])
    except Exception as e:
        print(f"  Rex-Omni detection error for '{cat}': {e}")

    try:
        pt_result = rex_model.inference(images=image, task="pointing",
                                         categories=[cat])
        preds = pt_result[0].get("extracted_predictions", {})
        if cat in preds:
            for pred in preds[cat]:
                points.append(pred["coords"])
    except Exception as e:
        print(f"  Rex-Omni pointing error for '{cat}': {e}")

    return {"bboxes": bboxes, "points": points}


def segment_with_sam2(sam2_model, image: Image.Image, bboxes: list,
                      points: list) -> "np.ndarray":
    """Run SAM2 with detected bboxes and points to produce a mask.

    Returns:
        Boolean mask as np.ndarray (H, W), or None on failure.
    """
    import numpy as np

    if not bboxes:
        return None

    img_arr = np.array(image.convert("RGB"))
    sam2_model.set_image(img_arr)

    merged_mask = None
    for i, bbox in enumerate(bboxes):
        point = points[i] if i < len(points) else None
        try:
            kwargs = {"box": bbox, "multimask_output": True}
            if point:
                kwargs["point_coords"] = [point]
                kwargs["point_labels"] = [1]
            masks, scores, _ = sam2_model.predict(**kwargs)
            best = masks[int(scores.argmax())]
            # SAM2 may return float masks (logits/probabilities); coerce to bool
            best_bool = best > 0 if best.dtype != bool else best
            merged_mask = best_bool if merged_mask is None else np.logical_or(merged_mask, best_bool)
        except Exception as e:
            print(f"  SAM2 error on bbox {i}: {e}")
            continue
    return merged_mask


def save_mask_visualization(image: Image.Image, mask, save_path: str):
    """Save mask + image_with_mask visualization."""
    import numpy as np
    import cv2

    if mask is None:
        return False

    # Save raw mask
    mask_img = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_img).save(save_path)

    # Save visualization
    vis_path = save_path.replace("mask.png", "image_with_mask.png")
    img_arr = np.array(image.convert("RGB"))
    overlay = img_arr.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([30, 144, 255]) * 0.6).astype(np.uint8)

    # Add green contours
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    Image.fromarray(overlay).save(vis_path)
    return True


def process_episode(check_result: dict, keyframes_root: str, output_dir: str,
                    rex_model, sam2_model) -> dict:
    """Generate masks for all valid QA pairs in one episode."""
    episode_name = check_result["episode_name"]
    valid_qas = check_result.get("valid_qa_pairs", [])
    if not valid_qas:
        return {"episode": episode_name, "num_masks": 0, "skipped": True}

    ep_dir = os.path.join(output_dir, episode_name)
    os.makedirs(ep_dir, exist_ok=True)

    # We need keyframe filename per image_id; load qa_pairs.json (sibling file)
    qa_pairs_path = os.path.join(
        os.path.dirname(check_result.get("__source__", "")),
        "qa_pairs.json"
    )

    # Find keyframes meta
    kf_dir = os.path.join(keyframes_root, episode_name)
    kf_meta_path = os.path.join(kf_dir, "keyframes_meta.json")
    if not os.path.exists(kf_meta_path):
        print(f"  [{episode_name}] no keyframes_meta.json found")
        return {"episode": episode_name, "num_masks": 0, "skipped": True}
    with open(kf_meta_path) as f:
        kf_meta = json.load(f)
    # Map image_id -> filename. The image_id is the subsampled index used at QA gen time.
    # We need the original images list from qa_pairs.json.
    qa_pairs_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "results", "..",  # placeholder; better: pass via args
    )
    # Simpler: load qa_pairs.json from same directory as qa_check_results.json
    src_dir = check_result.get("__source_dir__")
    if src_dir:
        qpp = os.path.join(src_dir, "qa_pairs.json")
        if os.path.exists(qpp):
            with open(qpp) as f:
                qa_full = json.load(f)
            id_to_file = {img["image_id"]: img["filename"] for img in qa_full["images"]}
        else:
            id_to_file = {}
    else:
        id_to_file = {}

    results = []
    for i, qa in enumerate(tqdm(valid_qas, desc=f"  {episode_name}", leave=False)):
        image_id = qa["image_id"]
        target_object = qa["object"]

        fname = id_to_file.get(image_id)
        if not fname:
            results.append({"qa_index": i, "status": "no_filename"})
            continue

        img_path = os.path.join(kf_dir, fname)
        if not os.path.exists(img_path):
            results.append({"qa_index": i, "status": "image_missing", "path": img_path})
            continue

        image = Image.open(img_path).convert("RGB")
        det = detect_object(rex_model, image, target_object)
        if not det["bboxes"]:
            results.append({"qa_index": i, "status": "no_detection"})
            continue

        mask = segment_with_sam2(sam2_model, image, det["bboxes"], det["points"])
        if mask is None:
            results.append({"qa_index": i, "status": "no_mask"})
            continue

        out_dir = os.path.join(ep_dir, f"task_{i:02d}")
        os.makedirs(out_dir, exist_ok=True)
        mask_path = os.path.join(out_dir, "mask.png")
        save_mask_visualization(image, mask, mask_path)

        # Save metadata
        with open(os.path.join(out_dir, "task_meta.json"), "w") as f:
            json.dump({
                "qa_index": i,
                "image_id": image_id,
                "image_filename": fname,
                "target_object": target_object,
                "task_instruction": qa["task_instruction"],
                "bboxes": det["bboxes"],
                "points": det["points"],
                "mask_path": "mask.png",
                "vis_path": "image_with_mask.png",
            }, f, indent=2, ensure_ascii=False)

        results.append({"qa_index": i, "status": "ok", "mask_path": mask_path})

    summary = {
        "episode": episode_name,
        "num_qa": len(valid_qas),
        "num_masks_ok": sum(1 for r in results if r["status"] == "ok"),
        "tasks": results,
    }
    with open(os.path.join(ep_dir, "episode_mask_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate masks for QA pairs")
    parser.add_argument("--qa_dir", type=str, required=True,
                        help="Directory with per-episode subdirs containing qa_pairs.json "
                             "and/or qa_check_results.json")
    parser.add_argument("--keyframes_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/episode_masks")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_check", action="store_true",
                        help="Use only QAs that passed uniqueness check "
                             "(requires qa_check_results.json). Default: use all qa_pairs.json")
    args = parser.parse_args()

    rex, sam2 = load_models(args.device)

    # Iterate over episodes
    summaries = []
    for ep_name in sorted(os.listdir(args.qa_dir)):
        ep_dir = os.path.join(args.qa_dir, ep_name)
        if not os.path.isdir(ep_dir):
            continue

        check_path = os.path.join(ep_dir, "qa_check_results.json")
        qa_path = os.path.join(ep_dir, "qa_pairs.json")

        if args.use_check and os.path.exists(check_path):
            with open(check_path) as f:
                check = json.load(f)
            check["__source_dir__"] = ep_dir
        elif os.path.exists(qa_path):
            # Fallback: use raw qa_pairs.json (no uniqueness filtering)
            with open(qa_path) as f:
                qa_full = json.load(f)
            check = {
                "episode_name": qa_full["episode_name"],
                "valid_qa_pairs": qa_full["qa_pairs"],
                "__source_dir__": ep_dir,
            }
        else:
            print(f"  [{ep_name}] no qa_pairs.json or qa_check_results.json, skipping")
            continue

        summary = process_episode(check, args.keyframes_root, args.output_dir, rex, sam2)
        summaries.append(summary)

    # Master summary
    total_qa = sum(s["num_qa"] for s in summaries)
    total_ok = sum(s["num_masks_ok"] for s in summaries)
    print(f"\nDone. {len(summaries)} episodes, {total_ok}/{total_qa} masks generated successfully")
    with open(os.path.join(args.output_dir, "mask_generation_summary.json"), "w") as f:
        json.dump({"total_qa": total_qa, "total_ok": total_ok,
                   "episodes": summaries}, f, indent=2)


if __name__ == "__main__":
    main()
