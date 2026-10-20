"""Repack a robomimic HDF5 demo file to contiguous, uncompressed datasets.

Some luggage sim files were written with gzip compression and h5py's automatic
chunk selection, which produced a pathological layout (e.g. chunks of shape
``(46, 24, 24, 1)`` for ``(T, 96, 96, 3)`` images).  With that layout, reading a
single sequence window forces HDF5 to locate and inflate dozens of tiny chunks,
making random-access dataloading ~13x slower than the contiguous/uncompressed
layout used by the stack/mug datasets.

This script losslessly rewrites every dataset as contiguous + uncompressed,
copying all groups and attributes verbatim, then atomically replaces the
original (unless ``--out`` is given).  Atomic replacement is safe for jobs that
already have the file open: their open handle keeps reading the old inode until
it is closed.

Usage:
    python utils/repack_hdf5_contiguous.py FILE [FILE ...]
    python utils/repack_hdf5_contiguous.py in.hdf5 --out out.hdf5
"""

import argparse
import os
import time

import h5py


def _copy_group(src, dst):
    for name, attr in src.attrs.items():
        dst.attrs[name] = attr
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            _copy_group(item, dst.create_group(key))
        else:
            # No chunks / no compression -> contiguous storage.
            new_ds = dst.create_dataset(key, data=item[()])
            for name, attr in item.attrs.items():
                new_ds.attrs[name] = attr


def repack(path, out=None):
    src_path = os.path.expanduser(path)
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)

    in_place = out is None
    tmp_path = (src_path + ".repack.tmp") if in_place else os.path.expanduser(out)

    t0 = time.time()
    with h5py.File(src_path, "r") as fs, h5py.File(tmp_path, "w") as fd:
        _copy_group(fs, fd)

    # Sanity check: same demo count and total samples.
    with h5py.File(src_path, "r") as fs, h5py.File(tmp_path, "r") as fd:
        sd, dd = fs["data"], fd["data"]
        assert set(sd.keys()) == set(dd.keys()), "demo set mismatch after repack"
        n_demos = len(sd.keys())
        s_tot = sum(int(sd[d].attrs["num_samples"]) for d in sd.keys())
        d_tot = sum(int(dd[d].attrs["num_samples"]) for d in dd.keys())
        assert s_tot == d_tot, f"sample count mismatch {s_tot} != {d_tot}"

    if in_place:
        os.replace(tmp_path, src_path)
        final = src_path
    else:
        final = tmp_path

    size_mb = os.path.getsize(final) / 1e6
    print(
        f"repacked {path} -> {final}  ({size_mb:.0f} MB, {n_demos} demos, "
        f"{s_tot} samples) in {time.time() - t0:.1f}s"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="HDF5 file(s) to repack")
    ap.add_argument("--out", default=None,
                    help="Output path (single input only). Default: replace in place.")
    args = ap.parse_args()

    if args.out is not None and len(args.files) != 1:
        ap.error("--out can only be used with a single input file")

    for f in args.files:
        repack(f, out=args.out)


if __name__ == "__main__":
    main()
