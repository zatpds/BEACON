#!/usr/bin/env python3
"""
Save sample observation frames (and short clips) from a dataset for visual
inspection -- useful for confirming that a domain variant looks as intended.

Example usage:
python scripts/get_obs.py \
        --path data/thread/rsws_thread_20.hdf5 \
        --num_samples 5 \
        --camera_key agentview45_image \
        --tag thread_rsws
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import h5py
import imageio.v2 as imageio
import numpy as np


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
    return sorted(demo_keys, key=lambda x: int(x.split("_")[-1]))


def _to_uint8_hwc(frames: np.ndarray) -> np.ndarray:
    """
    Convert frame arrays to uint8 HWC format.

    Supported input shapes:
      - (T, H, W, C)
      - (T, C, H, W)   where C in {1, 3, 4}
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D frames (T,*,*,*), got shape {frames.shape}")

    # Convert from CHW to HWC if needed
    if frames.shape[-1] in (1, 3, 4):
        out = frames
    elif frames.shape[1] in (1, 3, 4):
        out = np.transpose(frames, (0, 2, 3, 1))
    else:
        raise ValueError(
            f"Unable to infer channel axis from shape {frames.shape}. "
            "Expected (T,H,W,C) or (T,C,H,W)."
        )

    if out.dtype != np.uint8:
        # Handle common [0,1] float images safely.
        out = np.clip(out, 0.0, 1.0) * 255.0 if np.issubdtype(out.dtype, np.floating) else out
        out = out.astype(np.uint8)

    return out


def _sample_demo_ids(num_demos: int, num_samples: int, seed: int) -> List[int]:
    num_samples = min(num_samples, num_demos)
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_demos), k=num_samples))


def _write_videos_from_dataset(
    dataset_path: str,
    output_dir: str,
    dataset_tag: str,
    camera_key: str,
    num_samples: int,
    max_frames: int,
    fps: int,
    seed: int,
) -> Tuple[int, int]:
    """
    Returns:
        (num_written, total_candidates)
    """
    out_dir = Path(output_dir) / dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{dataset_path} does not contain top-level 'data' group.")
        demo_keys = _sorted_demo_keys(f["data"])
        if not demo_keys:
            raise ValueError(f"No demo_*/ groups found in {dataset_path}.")

        sampled_ids = _sample_demo_ids(
            num_demos=len(demo_keys),
            num_samples=num_samples,
            seed=seed,
        )

        written = 0
        for demo_id in sampled_ids:
            demo_key = demo_keys[demo_id]
            obs_group = f["data"][demo_key].get("obs")
            if obs_group is None:
                print(f"[{dataset_tag}] skip {demo_key}: missing obs group")
                continue
            if camera_key not in obs_group:
                available = ", ".join(obs_group.keys())
                print(
                    f"[{dataset_tag}] skip {demo_key}: missing obs/{camera_key}. "
                    f"Available keys: {available}"
                )
                continue

            frames = obs_group[camera_key][:]
            frames = _to_uint8_hwc(frames)
            if max_frames > 0:
                frames = frames[:max_frames]
            if len(frames) == 0:
                print(f"[{dataset_tag}] skip {demo_key}: no frames")
                continue

            video_path = out_dir / f"{dataset_tag}_{demo_key}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=fps)
            print(f"[{dataset_tag}] wrote {video_path} ({len(frames)} frames)")
            written += 1

    return written, len(demo_keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sample videos from a HDF5 dataset."
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to HDF5.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sample_videos",
        help="Directory to store generated videos.",
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="agentview_image",
        help="Observation image key under demo/obs to visualize.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of demo videos to sample per dataset.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum frames per video (0 means full trajectory).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for demo sampling.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="default",
        help="Dataset label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    written, total = _write_videos_from_dataset(
        dataset_path=args.path,
        output_dir=args.output_dir,
        dataset_tag=args.tag,
        camera_key=args.camera_key,
        num_samples=args.num_samples,
        max_frames=args.max_frames,
        fps=args.fps,
        seed=args.seed,
    )

    print("\nDone.")
    print(f"Output: wrote {written} videos (dataset demos: {total})")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
