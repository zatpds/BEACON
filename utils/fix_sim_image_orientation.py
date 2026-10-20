"""
Repair luggage *sim* HDF5 datasets whose camera images were stored upside down.

Background
----------
robosuite/MuJoCo render camera images origin-at-bottom (vertically flipped).
robomimic's EnvRobosuite flips them back to right-side-up at rollout/eval time.
The luggage sim demos were collected from a *raw* robosuite env and stored
without that flip, so their ``obs/agentview_image`` (and ``next_obs`` if present)
are upside down relative to:
  * what the policy sees during rollout/eval (robomimic flips), and
  * the real-robot dataset (stored right-side-up).

This script flips every ``*_image`` dataset under ``obs``/``next_obs`` along the
height axis, in place, making the sim data match the robomimic convention.

It is safe to keep around: a ``agentview_orientation_fixed`` flag is written to
``data.attrs`` so a file is never flipped twice.

Usage:
    python utils/fix_sim_image_orientation.py \\
        data/luggage/rs_luggage_500.hdf5 data/luggage/rsw_luggage_250.hdf5
"""

import argparse
import sys

import h5py
import numpy as np

FLAG = "agentview_orientation_fixed"


def fix_file(path):
    with h5py.File(path, "a") as f:
        data = f["data"]
        if data.attrs.get(FLAG, False):
            print(f"[skip] {path}: already fixed ({FLAG}=True)")
            return 0

        demo_keys = [k for k in data.keys() if k.startswith("demo_")]
        n_flipped = 0
        for dk in demo_keys:
            demo = data[dk]
            for grp_name in ("obs", "next_obs"):
                if grp_name not in demo:
                    continue
                grp = demo[grp_name]
                for key in grp.keys():
                    if not key.endswith("_image"):
                        continue
                    arr = grp[key][:]
                    # arr shape (T, H, W, C) -> flip height (axis=1)
                    grp[key][...] = arr[:, ::-1]
                    n_flipped += 1
        data.attrs[FLAG] = True
        print(f"[done] {path}: flipped {n_flipped} image datasets "
              f"across {len(demo_keys)} demos")
        return n_flipped


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="HDF5 file(s) to repair in place")
    args = p.parse_args()
    for path in args.paths:
        fix_file(path)


if __name__ == "__main__":
    sys.exit(main())
