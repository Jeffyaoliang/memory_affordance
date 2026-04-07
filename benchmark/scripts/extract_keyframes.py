"""
Extract keyframes from video files for Memory Affordance benchmark.

Strategies:
1. Uniform sampling: extract every N-th frame
2. Scene-change detection: extract frames at scene boundaries
3. Combined: uniform + scene-change for denser coverage at transitions

Usage:
    python extract_keyframes.py --video_dir data/epic-kitchens/EPIC-KITCHENS/P01/videos \
        --output_dir data/keyframes/P01 \
        --strategy uniform --interval 30 --max_frames 200
"""

import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_uniform(video_path: str, interval: int = 30, max_frames: int = 500) -> list[dict]:
    """Extract frames at uniform intervals.

    Args:
        video_path: Path to video file
        interval: Extract every N-th frame
        max_frames: Maximum number of frames to extract

    Returns:
        List of {frame_idx, timestamp_sec} dicts
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0 or fps <= 0:
        return []

    # If uniform interval gives too many frames, increase interval
    n_candidates = total_frames // interval
    if n_candidates > max_frames:
        interval = total_frames // max_frames

    keyframes = []
    for idx in range(0, total_frames, interval):
        if len(keyframes) >= max_frames:
            break
        keyframes.append({
            "frame_idx": idx,
            "timestamp_sec": round(idx / fps, 3),
        })
    return keyframes


def extract_scene_change(video_path: str, threshold: float = 30.0,
                         min_gap: int = 15, max_frames: int = 500) -> list[dict]:
    """Extract frames at scene change boundaries using histogram difference.

    Args:
        video_path: Path to video file
        threshold: Histogram difference threshold for scene change
        min_gap: Minimum frames between two keyframes
        max_frames: Maximum number of frames to extract
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0 or fps <= 0:
        cap.release()
        return []

    keyframes = []
    prev_hist = None
    last_kf_idx = -min_gap  # allow first frame

    for idx in tqdm(range(total_frames), desc="Scene detection", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        # Compute grayscale histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > threshold and (idx - last_kf_idx) >= min_gap:
                keyframes.append({
                    "frame_idx": idx,
                    "timestamp_sec": round(idx / fps, 3),
                })
                last_kf_idx = idx
                if len(keyframes) >= max_frames:
                    break

        prev_hist = hist

    cap.release()
    return keyframes


def extract_combined(video_path: str, uniform_interval: int = 60,
                     scene_threshold: float = 30.0, max_frames: int = 500) -> list[dict]:
    """Combine uniform + scene-change: union of both, sorted by frame index."""
    uniform = extract_uniform(video_path, uniform_interval, max_frames)
    scene = extract_scene_change(video_path, scene_threshold, min_gap=15, max_frames=max_frames)

    # Merge and deduplicate (within 5 frames = same keyframe)
    all_kfs = {kf["frame_idx"]: kf for kf in uniform}
    for kf in scene:
        # Check if close to existing
        close = any(abs(kf["frame_idx"] - existing) < 5 for existing in all_kfs)
        if not close:
            all_kfs[kf["frame_idx"]] = kf

    merged = sorted(all_kfs.values(), key=lambda x: x["frame_idx"])
    return merged[:max_frames]


def _phash(frame, hash_size: int = 16) -> int:
    """Compute a simple perceptual hash from a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedupe_keyframes(video_path: str, keyframes: list[dict],
                     hamming_threshold: int = 8, hash_size: int = 16) -> list[dict]:
    """Remove visually-similar adjacent keyframes via perceptual hash.

    Args:
        video_path: source video
        keyframes: list of {frame_idx, ...}
        hamming_threshold: drop a frame if its phash differs from the previous
            kept frame by <= this many bits (lower = stricter dedup)
        hash_size: phash grid size; final hash is hash_size*hash_size bits

    Returns:
        Filtered keyframe list (same dict shape)
    """
    if not keyframes:
        return []

    cap = cv2.VideoCapture(video_path)
    kept = []
    last_hash = None
    for kf in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf["frame_idx"])
        ret, frame = cap.read()
        if not ret:
            continue
        h = _phash(frame, hash_size=hash_size)
        if last_hash is None or _hamming(h, last_hash) > hamming_threshold:
            kept.append(kf)
            last_hash = h
    cap.release()
    return kept


def save_keyframes(video_path: str, keyframes: list[dict], output_dir: str) -> str:
    """Extract and save keyframe images + metadata.

    Returns:
        Path to the metadata JSON file.
    """
    video_name = Path(video_path).stem
    kf_dir = os.path.join(output_dir, video_name)
    os.makedirs(kf_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved = []
    for kf in tqdm(keyframes, desc=f"Saving {video_name}", leave=False):
        idx = kf["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        fname = f"frame_{idx:08d}.jpg"
        fpath = os.path.join(kf_dir, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved.append({
            **kf,
            "filename": fname,
            "width": frame.shape[1],
            "height": frame.shape[0],
        })

    cap.release()

    # Save metadata
    meta = {
        "video_path": video_path,
        "video_name": video_name,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 3) if fps > 0 else 0,
        "num_keyframes": len(saved),
        "keyframes": saved,
    }
    meta_path = os.path.join(kf_dir, "keyframes_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta_path


def process_video_dir(video_dir: str, output_dir: str, strategy: str,
                      interval: int, max_frames: int, scene_threshold: float,
                      dedupe: bool = True, dedupe_threshold: int = 8):
    """Process all videos in a directory."""
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.MP4'}
    videos = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if Path(f).suffix in video_exts:
                videos.append(os.path.join(root, f))
    videos.sort()

    print(f"Found {len(videos)} videos in {video_dir}")

    for vpath in tqdm(videos, desc="Processing videos"):
        video_name = Path(vpath).stem
        meta_path = os.path.join(output_dir, video_name, "keyframes_meta.json")
        if os.path.exists(meta_path):
            print(f"  Skipping {video_name} (already extracted)")
            continue

        if strategy == "uniform":
            kfs = extract_uniform(vpath, interval, max_frames)
        elif strategy == "scene_change":
            kfs = extract_scene_change(vpath, scene_threshold, max_frames=max_frames)
        elif strategy == "combined":
            kfs = extract_combined(vpath, interval, scene_threshold, max_frames)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        n_before = len(kfs)
        if dedupe and kfs:
            kfs = dedupe_keyframes(vpath, kfs, hamming_threshold=dedupe_threshold)
            print(f"  {video_name}: dedup {n_before} → {len(kfs)} keyframes (threshold={dedupe_threshold})")

        if kfs:
            save_keyframes(vpath, kfs, output_dir)
            print(f"  {video_name}: {len(kfs)} keyframes extracted")
        else:
            print(f"  {video_name}: no keyframes extracted (empty/corrupt video?)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyframes from videos")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--strategy", choices=["uniform", "scene_change", "combined"],
                        default="uniform")
    parser.add_argument("--interval", type=int, default=30,
                        help="Frame interval for uniform sampling (default: every 30 frames = ~1fps at 30fps video)")
    parser.add_argument("--max_frames", type=int, default=200,
                        help="Max keyframes per video")
    parser.add_argument("--scene_threshold", type=float, default=30.0,
                        help="Histogram difference threshold for scene change detection")
    parser.add_argument("--no_dedupe", action="store_true",
                        help="Disable perceptual-hash dedup of similar adjacent frames")
    parser.add_argument("--dedupe_threshold", type=int, default=8,
                        help="Hamming distance threshold for dedup (lower = stricter, default 8)")
    args = parser.parse_args()

    process_video_dir(args.video_dir, args.output_dir, args.strategy,
                      args.interval, args.max_frames, args.scene_threshold,
                      dedupe=not args.no_dedupe,
                      dedupe_threshold=args.dedupe_threshold)
