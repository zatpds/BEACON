#!/usr/bin/env python3
"""
Generate DTW pair-info JSON for OT co-training, with distribution analysis.

Computes Dynamic Time Warping alignments between every (target, source) demo
pair, writes a JSON file compatible with ChunkSamplingDTWTrainDataset, saves
a histogram of the distance distribution, and prints suggested cutoff ranges.

Example:
    python utils/gen_dtw.py \
        --src data/thread/rs_thread_500.hdf5 \
        --tgt data/thread/rsws_thread_20.hdf5 \
        --output data/thread/pair_info_rsws.json
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw


DEFAULT_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_gripper_qpos",
]

CUTOFF_PERCENTILES = [15, 20, 25, 30, 40, 50]


def extract_features(hdf5_path, obs_keys):
    """Load per-demo feature arrays from an HDF5 dataset.

    Returns dict[demo_id] = np.ndarray of shape (T, D).
    """
    features = {}
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for dk in demo_keys:
            arrs = [np.array(f[f"data/{dk}/obs/{k}"]) for k in obs_keys]
            features[dk] = np.concatenate(arrs, axis=-1).astype(np.float64)
    return features


def _compute_dtw_pair(args):
    """Worker: compute DTW between one (target, source) pair."""
    tgt_id, tgt_feat, src_id, src_feat = args

    alignment = dtw(tgt_feat, src_feat, distance_only=False)
    path_len = len(alignment.index1)
    raw_dist = alignment.distance / path_len

    pairing = defaultdict(list)
    for ti, si in zip(alignment.index1, alignment.index2):
        if si not in pairing[int(ti)]:
            pairing[int(ti)].append(int(si))

    return tgt_id, {
        "demo_name": src_id,
        "raw_dtw_dist": float(raw_dist),
        "weight": 0.0,
        "pairing": {str(k): v for k, v in sorted(pairing.items())},
    }


def analyze_distances(pair_info, output_stem, bins):
    """Print stats, plot histogram, and return suggested cutoffs."""
    dists = []
    for pairs in pair_info.values():
        for p in pairs:
            v = p["raw_dtw_dist"]
            if not math.isnan(v):
                dists.append(v)

    if not dists:
        print("No distances to analyze.")
        return

    dists_arr = np.array(dists)
    pct_values = np.percentile(dists_arr, CUTOFF_PERCENTILES)

    print("\n--- DTW Distance Distribution ---")
    print(f"  Count : {len(dists_arr)}")
    print(f"  Min   : {dists_arr.min():.6f}")
    print(f"  Max   : {dists_arr.max():.6f}")
    print(f"  Mean  : {dists_arr.mean():.6f}")
    print(f"  Std   : {dists_arr.std():.6f}")
    print()
    print("  Suggested cutoff range (percentile -> value):")
    for pct, val in zip(CUTOFF_PERCENTILES, pct_values):
        print(f"    p{pct:<3d} = {val:.6f}")

    cuts = [15, 25, 40]
    
    tight = pct_values[CUTOFF_PERCENTILES.index(cuts[0])]
    moderate = pct_values[CUTOFF_PERCENTILES.index(cuts[1])]
    loose = pct_values[CUTOFF_PERCENTILES.index(cuts[2])]
    print()
    print(f"  Recommended: cutoff in [{tight:.4f}, {loose:.4f}]")
    print(f"    tight  (p{cuts[0]}) = {tight:.4f}  — keep only closest ~{cuts[0]}% of pairs")
    print(f"    moderate(p{cuts[1]}) = {moderate:.4f}  — keep closest ~{cuts[1]}%")
    print(f"    loose  (p{cuts[2]}) = {loose:.4f}  — keep closest ~{cuts[2]}%")

    # histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dists_arr, bins=bins, edgecolor="black", alpha=0.75, label="DTW distances")

    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    labels = [f"p{cuts[0]} tight ({tight:.4f})", f"p{cuts[1]} moderate ({moderate:.4f})", f"p{cuts[2]} loose ({loose:.4f})"]
    for val, color, lab in zip([tight, moderate, loose], colors, labels):
        ax.axvline(val, color=color, linestyle="--", linewidth=1.5, label=lab)

    ax.set_xlabel("raw_dtw_dist")
    ax.set_ylabel("Count")
    ax.set_title("DTW Distance Distribution with Suggested Cutoffs")
    ax.legend(fontsize=8)
    fig.tight_layout()

    hist_path = output_stem + "_dist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"\n  Histogram saved to {hist_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DTW pair-info JSON with distribution analysis"
    )
    parser.add_argument("--src", required=True, help="Source domain HDF5 path")
    parser.add_argument("--tgt", required=True, help="Target domain HDF5 path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--obs_keys",
        nargs="+",
        default=DEFAULT_OBS_KEYS,
        help="Observation keys to concatenate as the DTW feature vector",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = cpu_count)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else cpu_count()

    print(f"Source: {args.src}")
    print(f"Target: {args.tgt}")
    print(f"Obs keys: {args.obs_keys}")
    print(f"Workers: {n_workers}")

    t0 = time.time()
    src_feats = extract_features(args.src, args.obs_keys)
    tgt_feats = extract_features(args.tgt, args.obs_keys)
    print(
        f"Loaded {len(src_feats)} source demos, {len(tgt_feats)} target demos "
        f"({time.time() - t0:.1f}s)"
    )

    total_pairs = len(tgt_feats) * len(src_feats)
    print(f"Computing {total_pairs} DTW pairs ...")

    work_items = []
    for tgt_id, tgt_feat in tgt_feats.items():
        for src_id, src_feat in src_feats.items():
            work_items.append((tgt_id, tgt_feat, src_id, src_feat))

    pair_info = defaultdict(list)
    done = 0
    t1 = time.time()

    with Pool(n_workers) as pool:
        for tgt_id, entry in pool.imap_unordered(_compute_dtw_pair, work_items, chunksize=8):
            pair_info[tgt_id].append(entry)
            done += 1
            if done % 500 == 0 or done == total_pairs:
                elapsed = time.time() - t1
                rate = done / elapsed
                eta = (total_pairs - done) / rate if rate > 0 else 0
                print(f"  {done}/{total_pairs}  ({rate:.1f} pairs/s, ETA {eta:.0f}s)")

    for tgt_id in pair_info:
        pair_info[tgt_id].sort(key=lambda x: int(x["demo_name"].split("_")[1]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pair_info, f)

    elapsed_total = time.time() - t0
    print(f"\nWrote {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")
    print(f"DTW computation done in {elapsed_total:.1f}s")

    output_stem = os.path.splitext(args.output)[0]
    analyze_distances(pair_info, output_stem, args.bins)


if __name__ == "__main__":
    main()
