"""
Extract frames from store walkthrough videos.

Strategy:
1. Try keyframe extraction (scene change detection via frame diff)
2. Fall back to strided extraction (every N frames)
3. Deduplicate near-identical frames via perceptual hashing
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

STORE_DIR = Path(__file__).parent / "data" / "store_photos"
OUTPUT_DIR = STORE_DIR / "video_frames"


def frame_diff(prev_gray, curr_gray) -> float:
    """Mean absolute difference between two grayscale frames."""
    return np.mean(np.abs(prev_gray.astype(float) - curr_gray.astype(float)))


def extract_keyframes(video_path: Path, threshold: float = 12.0, min_interval: int = 15) -> list[np.ndarray]:
    """Extract keyframes based on scene change detection."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Failed to open {video_path.name}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {video_path.name}: {total} frames, {fps:.1f} fps, {total/fps:.1f}s")

    frames = []
    prev_gray = None
    frame_idx = 0
    last_keyframe = -min_interval

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale for faster diff computation
        small = cv2.resize(gray, (320, 240))

        if prev_gray is not None:
            diff = frame_diff(prev_gray, small)
            if diff > threshold and (frame_idx - last_keyframe) >= min_interval:
                frames.append(frame)
                last_keyframe = frame_idx
        else:
            # Always keep first frame
            frames.append(frame)
            last_keyframe = frame_idx

        prev_gray = small
        frame_idx += 1

    cap.release()
    return frames


def extract_strided(video_path: Path, stride_seconds: float = 1.0) -> list[np.ndarray]:
    """Extract frames at fixed time intervals."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride_frames = max(1, int(fps * stride_seconds))
    print(f"  {video_path.name}: {total} frames, stride={stride_frames} ({stride_seconds}s)")

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride_frames == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def deduplicate(frames: list[np.ndarray], threshold: float = 5.0) -> list[np.ndarray]:
    """Remove near-duplicate frames using mean absolute diff."""
    if len(frames) <= 1:
        return frames

    kept = [frames[0]]
    prev_small = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), (320, 240))

    for frame in frames[1:]:
        curr_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240))
        diff = frame_diff(prev_small, curr_small)
        if diff > threshold:
            kept.append(frame)
            prev_small = curr_small

    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["keyframe", "strided", "both"], default="both")
    parser.add_argument("--stride", type=float, default=1.0, help="Seconds between strided frames")
    parser.add_argument("--threshold", type=float, default=12.0, help="Scene change threshold")
    parser.add_argument("--dedup-threshold", type=float, default=5.0, help="Dedup similarity threshold")
    args = parser.parse_args()

    videos = sorted(STORE_DIR.glob("*.mp4"))
    print(f"Found {len(videos)} videos")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_frames = 0

    for video_path in videos:
        print(f"\nProcessing {video_path.name}...")
        vid_name = video_path.stem

        all_frames = []

        if args.mode in ("keyframe", "both"):
            kf = extract_keyframes(video_path, threshold=args.threshold)
            print(f"  Keyframes: {len(kf)}")
            all_frames.extend(kf)

        if args.mode in ("strided", "both"):
            sf = extract_strided(video_path, stride_seconds=args.stride)
            print(f"  Strided: {len(sf)}")
            all_frames.extend(sf)

        # Deduplicate
        if args.mode == "both" and all_frames:
            before = len(all_frames)
            all_frames = deduplicate(all_frames, threshold=args.dedup_threshold)
            print(f"  After dedup: {len(all_frames)} (removed {before - len(all_frames)})")

        # Save frames
        for i, frame in enumerate(all_frames):
            out_path = OUTPUT_DIR / f"{vid_name}_frame_{i:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        total_frames += len(all_frames)
        print(f"  Saved {len(all_frames)} frames")

    print(f"\n=== COMPLETE ===")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
