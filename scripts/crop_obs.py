#!/usr/bin/env python3
"""
Crop image observations in a robomimic-style HDF5 dataset.

Input layout (must follow robomimic conventions):
    data/                                   attrs: env_args, total
        demo_0/                             attrs: model_file, num_samples, [camera_info]
            actions, dones, rewards, states
            obs/{<cam>_image, ...}
            next_obs/{<cam>_image, ...}
        demo_1/
        ...

For each image dataset (any obs/next_obs key ending in `_image`), this script
applies the same asymmetric crop expressed as fractions of the source frame:

    new = src[:, dy_t : H - dy_b, dx_l : W - dx_r, :]

where dy_t = round(H * top), dy_b = round(H * bottom), and similarly for x.
All non-image datasets, demo attrs, and group attrs are passed through.
The `env_args.env_kwargs.camera_heights/widths` field is updated to reflect
the new resolution, and per-demo `camera_info` (if present) has its intrinsic
principal point shifted to match the crop (cx -= dx_l, cy -= dy_t).

Example:
    python scripts/crop_obs.py \\
        --src data/stack_R/real_stack_40_raw.hdf5 \\
        --left 0.10 --right 0.10 --top 0.0 --bottom 0.20

By default the destination is written next to the source with a suffix derived
from the resulting image size, e.g. real_stack_40_raw.hdf5 -> real_stack_40_raw_crop96x96.hdf5
(the script just appends _crop<H>x<W> before the extension; pass --dst to
override).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


IMAGE_SUFFIX = "_image"


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    demos = [k for k in data_group.keys() if k.startswith("demo_")]
    return sorted(demos, key=lambda x: int(x.split("_")[-1]))


def _detect_image_keys(demo_group: h5py.Group) -> List[str]:
    """Return image observation keys (keys ending in _image) found under obs/."""
    obs = demo_group.get("obs")
    if obs is None:
        return []
    return sorted([k for k in obs.keys() if k.endswith(IMAGE_SUFFIX)])


def _crop_indices(
    H: int, W: int, top: float, bottom: float, left: float, right: float
) -> Tuple[int, int, int, int]:
    """Return (dy_t, dy_b, dx_l, dx_r) integer pixel margins for a HxW frame."""
    dy_t = int(round(H * top))
    dy_b = int(round(H * bottom))
    dx_l = int(round(W * left))
    dx_r = int(round(W * right))
    if dy_t + dy_b >= H or dx_l + dx_r >= W:
        raise ValueError(
            f"crop margins (top={top}, bottom={bottom}, left={left}, right={right}) "
            f"leave no pixels for image of size HxW={H}x{W}"
        )
    return dy_t, dy_b, dx_l, dx_r


def _crop_array(
    arr: np.ndarray, dy_t: int, dy_b: int, dx_l: int, dx_r: int
) -> np.ndarray:
    """Crop a (T, H, W, C) image stack by pixel margins."""
    if arr.ndim != 4:
        raise ValueError(f"expected (T, H, W, C) image stack, got shape {arr.shape}")
    T, H, W, C = arr.shape
    return arr[:, dy_t : H - dy_b, dx_l : W - dx_r, :]


def _update_env_args(env_args_str: str, new_h: int, new_w: int) -> str:
    """Patch env_args JSON so camera_heights/widths reflect the new resolution."""
    em = json.loads(env_args_str)
    kw = em.setdefault("env_kwargs", {})

    def _patch(field, new_val):
        if field not in kw:
            return
        cur = kw[field]
        if isinstance(cur, list):
            kw[field] = [new_val for _ in cur]
        else:
            kw[field] = new_val

    _patch("camera_heights", new_h)
    _patch("camera_widths", new_w)
    return json.dumps(em)


def _update_camera_info(camera_info_str: str, dx_l: int, dy_t: int) -> str:
    """Shift principal point in per-camera intrinsics: cx -= dx_l, cy -= dy_t.

    Camera info is stored as a JSON dict mapping camera name -> {intrinsics, extrinsics}.
    Only the intrinsic principal point is affected by an asymmetric pixel crop.
    """
    info = json.loads(camera_info_str)
    for cam_name, cam in info.items():
        K = cam.get("intrinsics")
        if K is None:
            continue
        # K is a 3x3 nested list with cx at K[0][2], cy at K[1][2].
        K[0][2] = float(K[0][2]) - dx_l
        K[1][2] = float(K[1][2]) - dy_t
        cam["intrinsics"] = K
    return json.dumps(info)


def _copy_attrs(src: h5py.HLObject, dst: h5py.HLObject) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _copy_dataset(
    src_ds: h5py.Dataset,
    dst_parent: h5py.Group,
    name: str,
    compression: Optional[str],
) -> None:
    """Copy a dataset verbatim, optionally re-compressing."""
    data = src_ds[()]
    kwargs = {}
    if compression and compression != "none" and data.size > 0:
        kwargs["compression"] = compression
    ds = dst_parent.create_dataset(name, data=data, **kwargs)
    _copy_attrs(src_ds, ds)


def crop_obs_file(
    src_path: str,
    dst_path: str,
    top: float,
    bottom: float,
    left: float,
    right: float,
    image_keys: Optional[List[str]],
    compression: Optional[str],
) -> Dict:
    """Run the crop and return a small summary dict."""
    src_path = os.path.abspath(src_path)
    dst_path = os.path.abspath(dst_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        os.remove(dst_path)

    with h5py.File(src_path, "r") as fi, h5py.File(dst_path, "w") as fo:
        # root attrs
        _copy_attrs(fi, fo)

        if "data" not in fi:
            raise KeyError(f"{src_path} has no top-level 'data' group")
        di = fi["data"]
        do = fo.create_group("data")

        demo_keys = _sorted_demo_keys(di)
        if not demo_keys:
            raise ValueError(f"{src_path} has no demo_* groups")

        # detect image keys (auto if not supplied)
        sample_demo = di[demo_keys[0]]
        detected_keys = _detect_image_keys(sample_demo)
        if image_keys is None:
            image_keys = detected_keys
            if not image_keys:
                raise ValueError(
                    f"no image keys found under {demo_keys[0]}/obs (expected suffix {IMAGE_SUFFIX!r})"
                )
        missing = [k for k in image_keys if k not in detected_keys]
        if missing:
            raise KeyError(f"image keys {missing} not present under {demo_keys[0]}/obs")

        # determine common HxW from sample demo (assume all demos share resolution)
        sample_obs = sample_demo["obs"]
        any_key = image_keys[0]
        _T, H, W, _C = sample_obs[any_key].shape
        dy_t, dy_b, dx_l, dx_r = _crop_indices(H, W, top, bottom, left, right)
        new_H = H - dy_t - dy_b
        new_W = W - dx_l - dx_r

        # copy data group attrs, patching env_args
        for k, v in di.attrs.items():
            if k == "env_args" and isinstance(v, (str, np.bytes_)):
                v_str = v.decode() if isinstance(v, np.bytes_) else v
                do.attrs[k] = _update_env_args(v_str, new_H, new_W)
            else:
                do.attrs[k] = v

        # iterate demos
        for dk in demo_keys:
            gi = di[dk]
            go = do.create_group(dk)
            # demo attrs (patch camera_info)
            for k, v in gi.attrs.items():
                if k == "camera_info" and isinstance(v, (str, np.bytes_)):
                    v_str = v.decode() if isinstance(v, np.bytes_) else v
                    go.attrs[k] = _update_camera_info(v_str, dx_l, dy_t)
                else:
                    go.attrs[k] = v

            # top-level demo datasets (actions, dones, rewards, states)
            for name in gi.keys():
                if isinstance(gi[name], h5py.Dataset):
                    _copy_dataset(gi[name], go, name, compression)

            # obs / next_obs subgroups
            for grp in ("obs", "next_obs"):
                if grp not in gi:
                    continue
                go_obs = go.create_group(grp)
                _copy_attrs(gi[grp], go_obs)
                for name in gi[grp].keys():
                    src_ds = gi[grp][name]
                    if name in image_keys:
                        arr = _crop_array(src_ds[()], dy_t, dy_b, dx_l, dx_r)
                        kwargs = {}
                        if compression and compression != "none" and arr.size > 0:
                            kwargs["compression"] = compression
                        ds = go_obs.create_dataset(name, data=arr, **kwargs)
                        _copy_attrs(src_ds, ds)
                    else:
                        _copy_dataset(src_ds, go_obs, name, compression)

    return {
        "src": src_path,
        "dst": dst_path,
        "num_demos": len(demo_keys),
        "image_keys": image_keys,
        "src_HxW": (H, W),
        "dst_HxW": (new_H, new_W),
        "pixel_margins_tblr": (dy_t, dy_b, dx_l, dx_r),
    }


def _default_dst_path(src_path: str, new_h: int, new_w: int) -> str:
    p = Path(src_path)
    return str(p.with_name(f"{p.stem}_crop{new_h}x{new_w}{p.suffix}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop image observations in a robomimic-style HDF5 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=str, required=True, help="path to input hdf5")
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="output hdf5 path (default: <src_stem>_crop<H>x<W><ext> next to src)",
    )
    parser.add_argument(
        "--left", type=float, default=0.0, help="fraction of width to drop from left"
    )
    parser.add_argument(
        "--right", type=float, default=0.0, help="fraction of width to drop from right"
    )
    parser.add_argument(
        "--top", type=float, default=0.0, help="fraction of height to drop from top"
    )
    parser.add_argument(
        "--bottom",
        type=float,
        default=0.0,
        help="fraction of height to drop from bottom",
    )
    parser.add_argument(
        "--camera_keys",
        type=str,
        nargs="*",
        default=None,
        help="image keys to crop (e.g. agentview_image). default: auto-detect (any key ending in _image).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="none",
        choices=["none", "gzip", "lzf"],
        help="HDF5 compression for output datasets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dst is None:
        # peek to compute default name based on actual resulting size
        with h5py.File(args.src, "r") as f:
            d0 = _sorted_demo_keys(f["data"])[0]
            keys = _detect_image_keys(f[f"data/{d0}"])
            if not keys:
                raise SystemExit(
                    "no image observations found in src; pass --dst explicitly"
                )
            _T, H, W, _C = f[f"data/{d0}/obs/{keys[0]}"].shape
        dy_t, dy_b, dx_l, dx_r = _crop_indices(
            H, W, args.top, args.bottom, args.left, args.right
        )
        args.dst = _default_dst_path(args.src, H - dy_t - dy_b, W - dx_l - dx_r)

    summary = crop_obs_file(
        src_path=args.src,
        dst_path=args.dst,
        top=args.top,
        bottom=args.bottom,
        left=args.left,
        right=args.right,
        image_keys=args.camera_keys,
        compression=args.compression,
    )

    print("\nCrop done.")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
