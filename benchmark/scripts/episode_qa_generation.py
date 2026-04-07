"""
Generate affordance QA pairs for memory episodes using VLM.

This script takes a directory of keyframes (extracted from a video) and sends
them as an interleaved image-text sequence to a VLM, which generates affordance
QA pairs identifying which frame contains the target object.

Usage:
    python episode_qa_generation.py \
        --episode_dir data/keyframes/P01_01 \
        --output_dir results/episodes/P01_01 \
        --max_images 50 \
        --model_name qwen3.5-397b-a17b \
        --api_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --api_key sk-xxx
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add Affordance_Annotator to path for model_utils
ANNOTATOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Affordance_Annotator")
sys.path.insert(0, ANNOTATOR_DIR)
from utils.model_utils import create_model, Prompt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT_PATH = os.path.join(SCRIPT_DIR, "..", "prompts", "memory_affordance_qa.md")


def load_episode_images(episode_dir: str, max_images: int = 50,
                        sample_strategy: str = "uniform") -> list[dict]:
    """Load keyframe images from an episode directory.

    Args:
        episode_dir: Directory containing keyframe images + keyframes_meta.json
        max_images: Maximum number of images to include
        sample_strategy: How to subsample if too many frames

    Returns:
        List of {image_id, filename, image (PIL), frame_idx, timestamp_sec}
    """
    meta_path = os.path.join(episode_dir, "keyframes_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        keyframes = meta["keyframes"]
    else:
        # Fallback: load images directly
        exts = {'.jpg', '.jpeg', '.png'}
        files = sorted([f for f in os.listdir(episode_dir)
                       if Path(f).suffix.lower() in exts])
        keyframes = [{"filename": f, "frame_idx": i, "timestamp_sec": 0}
                     for i, f in enumerate(files)]

    # Subsample if needed
    if len(keyframes) > max_images:
        if sample_strategy == "uniform":
            step = len(keyframes) / max_images
            indices = [int(i * step) for i in range(max_images)]
            keyframes = [keyframes[i] for i in indices]
        elif sample_strategy == "first":
            keyframes = keyframes[:max_images]

    # Load images
    results = []
    for idx, kf in enumerate(keyframes):
        img_path = os.path.join(episode_dir, kf["filename"])
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        results.append({
            "image_id": idx,
            "filename": kf["filename"],
            "image": img,
            "frame_idx": kf.get("frame_idx", idx),
            "timestamp_sec": kf.get("timestamp_sec", 0),
        })

    return results


def build_interleaved_prompt(system_prompt: str, images: list[dict]) -> tuple[str, list[Image.Image]]:
    """Build an interleaved text-image prompt for the VLM.

    The prompt format interleaves image IDs with images:
        [Image 0] <image> [Image 1] <image> ...

    Returns:
        (prompt_text, list_of_pil_images)
    """
    text_parts = [system_prompt, "\n\n### Episode Images\n\n"]
    pil_images = []

    for item in images:
        text_parts.append(f"[Image {item['image_id']}] ")
        pil_images.append(item["image"])

    text_parts.append("\n\n### Now generate affordance QA pairs for the above episode.")

    return "".join(text_parts), pil_images


def parse_qa_response(response_text: str) -> list[dict]:
    """Parse VLM response into structured QA pairs."""
    results = []
    text = re.sub(r'^```(json)?', '', response_text, flags=re.MULTILINE)
    text = re.sub(r'```$', '', text, flags=re.MULTILINE)

    decoder = json.JSONDecoder()
    pos = 0
    length = len(text)

    while pos < length:
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length:
            break
        try:
            obj, idx = decoder.raw_decode(text[pos:])
            # Validate required fields
            if "image_id" in obj and "object" in obj and "task_instruction" in obj:
                obj["image_id"] = int(obj["image_id"])
                results.append(obj)
            pos += idx
        except json.JSONDecodeError:
            next_start = text.find('{', pos + 1)
            if next_start != -1:
                pos = next_start
            else:
                break
    return results


def process_episode(episode_dir: str, output_dir: str, system_prompt: str,
                    model, max_images: int, sample_strategy: str) -> dict:
    """Process a single episode: load images, call VLM, save results.

    Returns:
        Result dict with episode metadata and QA pairs.
    """
    episode_name = Path(episode_dir).name
    result_path = os.path.join(output_dir, episode_name, "qa_pairs.json")

    if os.path.exists(result_path):
        print(f"  Skipping {episode_name} (already processed)")
        with open(result_path) as f:
            return json.load(f)

    # Load images
    images = load_episode_images(episode_dir, max_images, sample_strategy)
    if len(images) < 5:
        print(f"  Skipping {episode_name} (only {len(images)} frames, need at least 5)")
        return None

    print(f"  Processing {episode_name}: {len(images)} images")

    # Build prompt
    prompt_text, pil_images = build_interleaved_prompt(system_prompt, images)

    # Call VLM
    prompt = Prompt(text=prompt_text, images=pil_images)
    response = model.generate(prompt)

    # Parse response
    qa_pairs = parse_qa_response(response.text)

    # Validate image_ids are in range
    valid_ids = set(img["image_id"] for img in images)
    qa_pairs = [qa for qa in qa_pairs if qa["image_id"] in valid_ids]

    # Build result
    result = {
        "episode_name": episode_name,
        "episode_dir": episode_dir,
        "num_images": len(images),
        "num_qa_pairs": len(qa_pairs),
        "images": [
            {k: v for k, v in img.items() if k != "image"}
            for img in images
        ],
        "qa_pairs": qa_pairs,
        "raw_response": response.text,
    }

    # Save
    save_dir = os.path.join(output_dir, episode_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  {episode_name}: generated {len(qa_pairs)} QA pairs")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate affordance QA for memory episodes")
    parser.add_argument("--episode_dir", type=str, default=None,
                        help="Single episode directory (keyframes)")
    parser.add_argument("--episodes_root", type=str, default=None,
                        help="Root dir containing multiple episode dirs")
    parser.add_argument("--output_dir", type=str, default="results/episode_qa")
    parser.add_argument("--prompt_path", type=str, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--max_images", type=int, default=50,
                        help="Max images per episode to send to VLM")
    parser.add_argument("--sample_strategy", choices=["uniform", "first"], default="uniform")
    parser.add_argument("--model_name", type=str, default="qwen3.5-397b-a17b")
    parser.add_argument("--api_url", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Load API key from env if not provided
    if args.api_key is None:
        args.api_key = os.environ.get("QWEN_AUTH_TOKEN", "")

    # Load system prompt
    with open(args.prompt_path) as f:
        system_prompt = f.read()

    # Create model
    model = create_model(
        model_name=args.model_name,
        api_url=args.api_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Collect episodes
    if args.episode_dir:
        episodes = [args.episode_dir]
    elif args.episodes_root:
        episodes = sorted([
            os.path.join(args.episodes_root, d)
            for d in os.listdir(args.episodes_root)
            if os.path.isdir(os.path.join(args.episodes_root, d))
        ])
    else:
        print("Error: specify --episode_dir or --episodes_root")
        sys.exit(1)

    print(f"Processing {len(episodes)} episodes with model {args.model_name}")
    print(f"Max images per episode: {args.max_images}")

    results = []
    for ep_dir in tqdm(episodes, desc="Episodes"):
        result = process_episode(ep_dir, args.output_dir, system_prompt,
                                 model, args.max_images, args.sample_strategy)
        if result:
            results.append(result)

    # Save summary
    summary_path = os.path.join(args.output_dir, "generation_summary.json")
    summary = {
        "total_episodes": len(results),
        "total_qa_pairs": sum(r["num_qa_pairs"] for r in results),
        "model": args.model_name,
        "max_images": args.max_images,
        "episodes": [
            {"name": r["episode_name"], "num_images": r["num_images"],
             "num_qa_pairs": r["num_qa_pairs"]}
            for r in results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {summary['total_episodes']} episodes, {summary['total_qa_pairs']} QA pairs total.")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
