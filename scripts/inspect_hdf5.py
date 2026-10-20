#!/usr/bin/env python3
"""
Print the structure and summary statistics of a robomimic-style HDF5 dataset.

Useful for checking a freshly generated or converted dataset before training:
demo count and lengths, observation keys and shapes, action ranges, and the
env metadata that robomimic needs to rebuild the environment for rollouts.

Usage:
    python scripts/inspect_hdf5.py data/stack/rs_stack_500.hdf5 [more.hdf5 ...]
"""

import sys

import h5py
import numpy as np


def inspect_hdf5(path):
    print(f"\n--- {path} ---")

    with h5py.File(path, "r") as f:
        print("\ntop-level keys:")
        for k in f.keys():
            print(f"  {k}")

        if "data" not in f:
            print("  no 'data' group found -- not a robomimic dataset")
            return

        data = f["data"]
        demo_keys = sorted(
            data.keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else x
        )
        print(f"\ndemos: {len(demo_keys)}  (first 5: {demo_keys[:5]})")

        demo0_key = demo_keys[0]
        demo0 = data[demo0_key]
        print(f"\nstructure of {demo0_key}:")
        for k in demo0.keys():
            if isinstance(demo0[k], h5py.Group):
                print(f"  {k}/")
                for sk in demo0[k].keys():
                    if isinstance(demo0[k][sk], h5py.Group):
                        print(f"    {sk}/")
                        for ssk in demo0[k][sk].keys():
                            ds = demo0[k][sk][ssk]
                            print(f"      {ssk}: shape={ds.shape}, dtype={ds.dtype}")
                    else:
                        ds = demo0[k][sk]
                        print(f"    {sk}: shape={ds.shape}, dtype={ds.dtype}")
            else:
                print(f"  {k}: shape={demo0[k].shape}, dtype={demo0[k].dtype}")

        demo_lengths = [
            data[dk]["actions"].shape[0] for dk in demo_keys if "actions" in data[dk]
        ]
        print("\ndemo lengths:")
        print(f"  total steps: {sum(demo_lengths)}")
        print(f"  mean: {np.mean(demo_lengths):.1f}  std: {np.std(demo_lengths):.1f}")
        print(f"  min: {min(demo_lengths)}  max: {max(demo_lengths)}")

        all_actions = np.concatenate(
            [data[dk]["actions"][:] for dk in demo_keys if "actions" in data[dk]], axis=0
        )
        print("\nactions (all demos):")
        print(f"  shape: {all_actions.shape}, dtype: {all_actions.dtype}")
        print(f"  min:  {all_actions.min(axis=0)}")
        print(f"  max:  {all_actions.max(axis=0)}")
        print(f"  mean: {all_actions.mean(axis=0)}")
        print(f"  std:  {all_actions.std(axis=0)}")

        if "obs" in demo0:
            print("\nobservations (all demos):")
            for ok in sorted(demo0["obs"].keys()):
                sample = demo0["obs"][ok]
                if len(sample.shape) > 2:
                    print(f"  {ok}: shape={sample.shape}, dtype={sample.dtype} (image)")
                    continue
                all_obs = np.concatenate(
                    [data[dk]["obs"][ok][:] for dk in demo_keys if ok in data[dk]["obs"]],
                    axis=0,
                )
                print(f"  {ok}: dim={sample.shape[1:]}, dtype={sample.dtype}")
                if all_obs.dtype in (np.float32, np.float64, np.int32, np.int64):
                    print(f"    min:  {all_obs.min(axis=0)}")
                    print(f"    max:  {all_obs.max(axis=0)}")
                    print(f"    mean: {all_obs.mean(axis=0)}")
                    print(f"    std:  {all_obs.std(axis=0)}")

        print("\nattributes:")
        for name, attrs in (("file", f.attrs), ("data", data.attrs), (demo0_key, demo0.attrs)):
            for key in attrs.keys():
                val = attrs[key]
                if isinstance(val, bytes):
                    val = val.decode()
                print(f"  [{name}] {key}: {val}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    for p in sys.argv[1:]:
        inspect_hdf5(p)
