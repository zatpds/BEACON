"""
UMAP visualization of encoder features, source vs. target (paper, Appendix E:
feature alignment). The protocol follows the latent-feature analysis of
"A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot
Policies" (Wei et al.).

For each method (Single Domain baseline, Co-Training, MMD, UOT, BEACON), we
embed ~1000 randomly sampled observations from the source domain and the
target domain through the trained model's encoder and visualize the resulting
features with UMAP.

Two feature levels are visualized:
  - Visual features  : output of the VisualCore (vision stem) per camera,
                       concatenated across cameras.
  - Observation features : full output of the ObservationGroupEncoder
                       (vision + low-dim modalities concatenated).

For the "Single Domain" column (no shared encoder), source data is embedded
with a source-only checkpoint and target data with a target-only checkpoint,
then plotted together to show the disjoint "no-alignment" state.

Which checkpoints to compare is given by a small JSON manifest (see
example_manifest.json next to this script and the format description below).

Usage:
    python scripts/feature_align/plot_feature_align.py \
        --manifest scripts/feature_align/example_manifest.json --label stack
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Checkpoint manifest
# The script reads a small JSON file describing one sim-to-sim setup:
#
#   {
#     "source_data":      "data/stack/rs_stack_500.hdf5",
#     "target_data":      "data/stack/rsw_stack_20.hdf5",
#     "source_only_ckpt": "runs/source_only_stack_sim/<run>/models/model_epoch_600.pth",
#     "target_only_ckpt": "runs/target_only_stack_sim/<run>/models/model_epoch_300.pth",
#     "methods": {
#       "Co-Training": "runs/cotrain_stack_sim/<run>/models/model_epoch_600.pth",
#       "MMD":         "runs/mmd_stack_sim/<run>/models/model_epoch_600.pth",
#       "UOT":         "runs/uot_stack_sim/<run>/models/model_epoch_600.pth",
#       "BEACON":      "runs/beacon_stack_sim/<run>/models/model_epoch_600.pth"
#     }
#   }
#
# "source_only_ckpt" may be null: the "Single Domain" column then passes both
# domains through the target-only checkpoint. Relative paths are resolved
# against the repository root. See example_manifest.json next to this script.


def load_manifest(path):
    with open(path, "r") as f:
        raw = json.load(f, object_pairs_hook=OrderedDict)
    required = ("source_data", "target_data", "target_only_ckpt", "methods")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Manifest {path} is missing keys: {missing}")
    raw.setdefault("source_only_ckpt", None)
    raw["methods"] = OrderedDict(raw["methods"])
    return raw


# HDF5 sampling

def sample_observations(hdf5_path, obs_keys, num_samples, seed=0):
    """Randomly sample @num_samples observations (single time-step each)
    from a SequenceDataset-style HDF5 file.

    Returns:
        obs_dict: dict mapping obs_key -> numpy array
                  (images stay uint8 HWC; low-dim stay float)
    """
    rng = np.random.default_rng(seed)
    full_path = os.path.join(REPO_ROOT, hdf5_path) if not os.path.isabs(hdf5_path) else hdf5_path

    with h5py.File(full_path, "r") as f:
        demo_keys = sorted(
            f["data"].keys(), key=lambda x: int(x.split("_")[1])
        )
        # build a flat list of (demo_key, frame_idx) pairs
        index_list = []
        for dk in demo_keys:
            n = f["data"][dk]["actions"].shape[0]
            index_list.extend([(dk, i) for i in range(n)])

        if num_samples > len(index_list):
            print(
                f"  [sample_observations] WARNING: requested {num_samples} > total "
                f"frames {len(index_list)} in {hdf5_path}; using all frames."
            )
            num_samples = len(index_list)

        choice_idx = rng.choice(len(index_list), size=num_samples, replace=False)
        choice_idx.sort()  # sequential reads inside HDF5 are faster

        obs_dict = {k: [] for k in obs_keys}
        for ci in choice_idx:
            dk, fi = index_list[ci]
            for k in obs_keys:
                obs_dict[k].append(f["data"][dk]["obs"][k][fi])

        for k in obs_keys:
            obs_dict[k] = np.stack(obs_dict[k], axis=0)

    return obs_dict


# Feature extraction

def get_obs_encoder(model):
    """Return the (EMA-averaged when available) obs_encoder module."""
    if model.ema is not None:
        nets = model.ema.averaged_model
    else:
        nets = model.nets
    return nets["policy"]["obs_encoder"]


def extract_features(model, obs_np, device, batch_size=64, restrict_keys=None):
    """Run @obs_np (numpy obs dict) through the encoder and return:

        visual_feats : numpy array [N, D_visual]   concatenated VisualCore
                                                   outputs across rgb keys.
        obs_feats    : numpy array [N, D_obs]      observation encoder output
                                                   (visual + low-dim concat).

    If @restrict_keys is given, only those obs keys (that the model supports)
    are used to build the features. This is needed to make features
    comparable when different checkpoints have different obs key sets.
    The order in @restrict_keys is preserved in the concatenation, ensuring
    consistent feature layout across models.
    """
    obs_encoder_group = get_obs_encoder(model)
    obs_encoder = obs_encoder_group.nets["obs"]

    model_keys = list(obs_encoder.obs_shapes.keys())
    if restrict_keys is not None:
        used_keys = [k for k in restrict_keys if k in model_keys]
    else:
        used_keys = model_keys

    rgb_keys = [k for k in used_keys
                if ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb")]

    n = next(iter(obs_np.values())).shape[0]
    feature_activation = obs_encoder.activation

    visual_chunks = []
    obs_chunks = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_np = {k: obs_np[k][start:end] for k in used_keys}
            batch_processed = ObsUtils.process_obs_dict(batch_np)
            batch_torch = {
                k: torch.from_numpy(v).to(device).float()
                for k, v in batch_processed.items()
            }

            # Manually replicate ObservationEncoder.forward but restricted
            # to @used_keys, so feature dim is consistent across models.
            visual_per_key = []
            obs_per_key = []
            for k in used_keys:
                x = batch_torch[k]
                # apply randomizers' forward_in (typically no-op at eval)
                for rand in obs_encoder.obs_randomizers[k]:
                    if rand is not None:
                        x = rand.forward_in(x)
                if obs_encoder.obs_nets[k] is not None:
                    x = obs_encoder.obs_nets[k](x)
                    if feature_activation is not None:
                        x = feature_activation(x)
                for rand in reversed(obs_encoder.obs_randomizers[k]):
                    if rand is not None:
                        x = rand.forward_out(x)
                x = x.flatten(start_dim=1)
                obs_per_key.append(x)
                if k in rgb_keys:
                    visual_per_key.append(x)

            visual = torch.cat(visual_per_key, dim=-1) if visual_per_key else None
            full_obs = torch.cat(obs_per_key, dim=-1)

            if visual is not None:
                visual_chunks.append(visual.cpu().numpy())
            obs_chunks.append(full_obs.cpu().numpy())

    visual_feats = np.concatenate(visual_chunks, axis=0) if visual_chunks else None
    obs_feats = np.concatenate(obs_chunks, axis=0)
    return visual_feats, obs_feats


# Checkpoint loading

def load_model(ckpt_path, device):
    """Load a robomimic policy checkpoint and return the underlying algo."""
    ckpt_path = os.path.join(REPO_ROOT, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
    print(f"  Loading checkpoint: {os.path.relpath(ckpt_path, REPO_ROOT)}")

    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    if isinstance(ckpt_dict.get("env_metadata"), list):
        ckpt_dict["env_metadata"] = ckpt_dict["env_metadata"][0]
    if isinstance(ckpt_dict.get("shape_metadata"), list):
        ckpt_dict["shape_metadata"] = ckpt_dict["shape_metadata"][0]

    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict, device=device, verbose=False,
    )
    model = policy.policy
    model.set_eval()
    return model


# UMAP + plotting helpers

def run_umap(features, seed=0, n_components=2, n_neighbors=30, min_dist=0.3):
    """Run UMAP to project @features (numpy [N, D]) to @n_components dims.

    n_neighbors / min_dist tuned to match the look in the paper's Fig 6
    (relatively local geometry preserved for visual features).
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(features)


def _style_3d_axes(ax, n_grid=8):
    """Apply a clean, modern 'studio dashboard' look to a 3D axes.

    Each of the three back panes gets a slightly different grey tint,
    with crisp white gridlines (n_grid divisions per axis). The cube
    outline is drawn manually with the top edges omitted (open-top cube).
    """
    pane_face = {
        "x": (0.86, 0.86, 0.86, 1.0),   # left  : medium grey
        "y": (0.90, 0.90, 0.90, 1.0),   # back  : light grey
        "z": (0.82, 0.82, 0.82, 1.0),   # floor : slightly darker grey
    }
    border = "#3a3a3a"
    grid_color = "#ffffff"
    tick_color = "#3a3a3a"

    from matplotlib.ticker import MaxNLocator
    for axis_name, axis in (("x", ax.xaxis), ("y", ax.yaxis), ("z", ax.zaxis)):
        axis.pane.set_facecolor(pane_face[axis_name])
        # Hide the auto pane edges; we'll draw selected cube edges manually
        # below so we can omit the top of the cube.
        axis.pane.set_edgecolor("none")
        axis.pane.set_linewidth(0.0)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = grid_color
        axis._axinfo["grid"]["linewidth"] = 1.3
        axis._axinfo["grid"]["linestyle"] = "-"
        axis.set_major_locator(MaxNLocator(nbins=n_grid))
        axis.set_tick_params(colors=tick_color, length=0, pad=2, labelsize=0)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True)
    _draw_open_top_cube(ax, color=border, linewidth=1.4)


def _draw_open_top_cube(ax, color="#3a3a3a", linewidth=1.4):
    """Draw the cube outline with the top removed.

    Edges drawn:
      - Floor outline: 4 edges around the bottom face (z = z_min).
      - Back-left vertical corner: where the left (x_min) and back (y_max)
        panes meet.

    Front-corner verticals and top edges are intentionally omitted so the
    cube reads as 'open' from the top and front-corner lines don't cut
    through the data.
    """
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    kw = dict(color=color, linewidth=linewidth, zorder=0.5)

    # Floor outline (rectangle at z = z0)
    ax.plot([x0, x1], [y0, y0], [z0, z0], **kw)
    ax.plot([x0, x1], [y1, y1], [z0, z0], **kw)
    ax.plot([x0, x0], [y0, y1], [z0, z0], **kw)
    ax.plot([x1, x1], [y0, y1], [z0, z0], **kw)

    # Back-left vertical corner (where the left & back panes meet).
    ax.plot([x0, x0], [y1, y1], [z0, z1], **kw)

    # Re-clamp limits in case the manual plot calls expanded them.
    ax.set_xlim3d(x0, x1)
    ax.set_ylim3d(y0, y1)
    ax.set_zlim3d(z0, z1)


def scatter_features(ax, embedding, n_source, title=None, ylabel=None,
                     point_size=8, alpha=0.55, is_3d=False,
                     view_init=(20, -60)):
    """Render a 2-class UMAP scatter (source = blue, target = red).

    Works for both 2D and 3D axes — pass is_3d=True for an Axes3D.
    """
    src = embedding[:n_source]
    tgt = embedding[n_source:]
    blue = "#3b6cf2"
    red = "#e34234"

    if is_3d:
        ax.scatter(src[:, 0], src[:, 1], src[:, 2],
                   c=blue, s=point_size, alpha=alpha,
                   label="Source", linewidths=0, depthshade=True)
        ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2],
                   c=red, s=point_size, alpha=alpha,
                   label="Target (real / sim-wood)", linewidths=0, depthshade=True)
        ax.view_init(elev=view_init[0], azim=view_init[1])
        _style_3d_axes(ax)
        # Equal-aspect cube: avoids the previous squashed look.
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.scatter(src[:, 0], src[:, 1], c=blue, s=point_size,
                   alpha=alpha, label="Source", linewidths=0,
                   rasterized=True)
        ax.scatter(tgt[:, 0], tgt[:, 1], c=red, s=point_size,
                   alpha=alpha, label="Target",
                   linewidths=0, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#444444")

    if title is not None:
        ax.set_title(title, fontsize=18, fontweight="bold")
    if ylabel is not None:
        if is_3d:
            # 3D axes don't have a clean left-side ylabel slot. Use a
            # text annotation in figure-axes coords on the left of the cell
            # so it isn't rotated by the 3D projection.
            ax.text2D(
                -0.10, 0.5, ylabel,
                transform=ax.transAxes, rotation=90,
                ha="center", va="center",
                fontsize=16, fontweight="bold",
            )
        else:
            ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")


# Main pipeline

def collect_features_for_method(method_name, ckpt_or_pair,
                                source_obs_np, target_obs_np,
                                device, restrict_keys=None):
    """Compute (visual, observation) features for a single method column.

    ckpt_or_pair:
        - For "Source and Target": a (source_only_ckpt, target_only_ckpt) tuple.
          Source data goes through source_only model, target data goes
          through target_only model.
        - Otherwise: a single co-trained checkpoint path used for both
          domains.

    @restrict_keys (list[str]): if given, only these obs keys are used for
        feature extraction (must be a subset of every model's obs_shapes).
    """
    print(f"\n[{method_name}]")
    if isinstance(ckpt_or_pair, tuple):
        src_ckpt, tgt_ckpt = ckpt_or_pair
        print("  Using SEPARATE encoders (source-only + target-only).")

        model = load_model(src_ckpt, device)
        src_visual, src_obs = extract_features(
            model, source_obs_np, device, restrict_keys=restrict_keys,
        )
        del model
        torch.cuda.empty_cache()

        model = load_model(tgt_ckpt, device)
        tgt_visual, tgt_obs = extract_features(
            model, target_obs_np, device, restrict_keys=restrict_keys,
        )
        del model
        torch.cuda.empty_cache()
    else:
        model = load_model(ckpt_or_pair, device)
        src_visual, src_obs = extract_features(
            model, source_obs_np, device, restrict_keys=restrict_keys,
        )
        tgt_visual, tgt_obs = extract_features(
            model, target_obs_np, device, restrict_keys=restrict_keys,
        )
        del model
        torch.cuda.empty_cache()

    if src_visual is not None and tgt_visual is not None:
        visual = np.concatenate([src_visual, tgt_visual], axis=0).astype(np.float32)
    else:
        visual = None
    obs = np.concatenate([src_obs, tgt_obs], axis=0).astype(np.float32)

    n_src = src_obs.shape[0]
    print(f"  source feat: visual={None if src_visual is None else src_visual.shape}, "
          f"obs={src_obs.shape}")
    print(f"  target feat: visual={None if tgt_visual is None else tgt_visual.shape}, "
          f"obs={tgt_obs.shape}")
    return visual, obs, n_src


def main(args):
    cfg = load_manifest(args.manifest)
    source_data = cfg["source_data"]
    target_data = cfg["target_data"]
    target_only_ckpt = cfg["target_only_ckpt"]
    source_only_ckpt = cfg.get("source_only_ckpt")

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"Using device: {device}")
    print(f"Manifest: {args.manifest} (label: {args.label})")
    print(f"  source data: {source_data}")
    print(f"  target data: {target_data}")
    print(f"  source-only ckpt: {source_only_ckpt}")
    print(f"  target-only ckpt: {target_only_ckpt}")

    # Build column list
    # "Single Domain" baseline:
    #   - if BOTH source-only and target-only ckpts exist: pass source data
    #     through source-only encoder + target data through target-only encoder
    #     (paper's "no shared encoder" baseline).
    #   - if only target-only is available: pass BOTH domains through the
    #     target-only encoder (paper's "Real-Only" convention).
    if source_only_ckpt is not None:
        single_domain_ckpt = (source_only_ckpt, target_only_ckpt)
    else:
        single_domain_ckpt = target_only_ckpt
    columns = [("Single Domain", single_domain_ckpt)]
    for method_name, ckpt in cfg["methods"].items():
        columns.append((method_name, ckpt))

    # Inspect all checkpoint obs key sets to find a SHARED set
    # Different checkpoints may use different obs key sets (e.g. source-only
    # has both agentview + eye_in_hand cameras while target-only has just
    # agentview). To make features dimensionally comparable across
    # checkpoints/columns, we restrict feature extraction to the
    # intersection of obs keys across every checkpoint we'll use.
    all_ckpts = [target_only_ckpt] + list(cfg["methods"].values())
    if source_only_ckpt is not None:
        all_ckpts = [source_only_ckpt] + all_ckpts
    print("\nDetermining shared obs keys across all checkpoints...")

    rgb_set = set()
    low_set = set()
    shared_low = None
    shared_rgb = None
    for cp in all_ckpts:
        cp_full = os.path.join(REPO_ROOT, cp) if not os.path.isabs(cp) else cp
        ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=cp_full)
        cfg_json = json.loads(ckpt_dict["config"])
        obs_modalities = cfg_json["observation"]["modalities"]["obs"]
        low_here = list(obs_modalities.get("low_dim", []))
        rgb_here = list(obs_modalities.get("rgb", []))
        rgb_set.update(rgb_here)
        low_set.update(low_here)
        shared_low = set(low_here) if shared_low is None else shared_low & set(low_here)
        shared_rgb = set(rgb_here) if shared_rgb is None else shared_rgb & set(rgb_here)
        print(f"  {os.path.relpath(cp_full, REPO_ROOT)}")
        print(f"      low_dim: {low_here}")
        print(f"      rgb    : {rgb_here}")

    # Use a deterministic order: low_dim alphabetical, rgb alphabetical
    low_keys = sorted(shared_low)
    rgb_keys = sorted(shared_rgb)
    shared_obs_keys = low_keys + rgb_keys
    print(f"  shared low_dim keys : {low_keys}")
    print(f"  shared rgb keys     : {rgb_keys}")

    print(f"\nSampling {args.num_samples} obs from source ({source_data})")
    src_obs_np = sample_observations(
        source_data, shared_obs_keys, args.num_samples, seed=args.seed,
    )
    print(f"Sampling {args.num_samples} obs from target ({target_data})")
    tgt_obs_np = sample_observations(
        target_data, shared_obs_keys, args.num_samples, seed=args.seed + 1,
    )

    # Run feature extraction + UMAP per column
    column_results = []
    for method_name, ckpt_or_pair in columns:
        t0 = time.time()
        visual, obs, n_src = collect_features_for_method(
            method_name, ckpt_or_pair, src_obs_np, tgt_obs_np, device,
            restrict_keys=shared_obs_keys,
        )
        elapsed = time.time() - t0
        print(f"  feature extraction: {elapsed:.1f}s")

        t0 = time.time()
        visual_emb = (
            run_umap(visual, seed=args.seed, n_components=args.ndim)
            if visual is not None else None
        )
        obs_emb = run_umap(obs, seed=args.seed, n_components=args.ndim)
        print(f"  UMAP: {time.time() - t0:.1f}s")

        column_results.append({
            "method": method_name,
            "visual_emb": visual_emb,
            "obs_emb": obs_emb,
            "n_source": n_src,
        })

    # Plot full + subset variants
    is_3d = (args.ndim == 3)

    # Full 5-column plot: same clean style as subset (no title, larger
    # text, vertical dividers) but with slightly shorter dividers and
    # tighter inter-column spacing since 5 columns are denser.
    full_path = args.output
    render_figure(
        column_results=column_results,
        method_subset=None,
        is_3d=is_3d,
        title=None,
        legend_fontsize=15,
        vertical_dividers=True,
        divider_height_frac=0.82,
        wspace=-0.05,
        output_path=full_path,
        dpi=args.dpi,
    )

    # Reduced 3-column plot (Single Domain, Co-Training, BEACON): no title,
    # full-height vertical dividers, larger legend.
    subset_path = os.path.splitext(full_path)[0] + "_subset.png"
    render_figure(
        column_results=column_results,
        method_subset=["Single Domain", "Co-Training", "BEACON"],
        is_3d=is_3d,
        title=None,
        legend_fontsize=15,
        vertical_dividers=True,
        divider_height_frac=1.0,
        wspace=0.0,
        output_path=subset_path,
        dpi=args.dpi,
    )

    # Cache UMAP results so future re-renders don't need to redo UMAP.
    cache_path = os.path.splitext(full_path)[0] + "_data.npz"
    np.savez(
        cache_path,
        **{
            f"col{ci}_method": res["method"]
            for ci, res in enumerate(column_results)
        },
        **{
            f"col{ci}_visual": (res["visual_emb"] if res["visual_emb"] is not None
                                else np.zeros(0))
            for ci, res in enumerate(column_results)
        },
        **{
            f"col{ci}_obs": res["obs_emb"]
            for ci, res in enumerate(column_results)
        },
        **{
            f"col{ci}_nsrc": res["n_source"]
            for ci, res in enumerate(column_results)
        },
    )
    print(f"Saved feature bundle  to {cache_path}")


# Figure rendering

def render_figure(column_results, method_subset, is_3d, title,
                  legend_fontsize, vertical_dividers, output_path, dpi,
                  divider_height_frac=1.0, wspace=0.02):
    """Render a single figure from the precomputed @column_results.

    Args:
        column_results: list of dicts with keys ``method``, ``visual_emb``,
            ``obs_emb``, ``n_source``. The full set computed in main().
        method_subset: optional list of method names to include (in order).
            None means use all columns.
        is_3d: True for 3D scatters, False for 2D.
        title: figure suptitle text, or None to omit.
        legend_fontsize: legend label font size.
        vertical_dividers: if True, draw thin vertical dividers between
            adjacent method columns.
        output_path: path to save the figure.
        dpi: output dpi.
        divider_height_frac: fraction of column height the divider spans
            (1.0 = nearly full height, 0.85 = slightly shorter).
        wspace: horizontal space between subplots (matplotlib units).
    """
    if method_subset is not None:
        cols = [r for name in method_subset
                for r in column_results if r["method"] == name]
        missing = [m for m in method_subset
                   if not any(r["method"] == m for r in column_results)]
        if missing:
            raise ValueError(
                f"Methods {missing} not found in column_results. "
                f"Available: {[r['method'] for r in column_results]}"
            )
    else:
        cols = column_results

    n_cols = len(cols)
    has_visual = any(r["visual_emb"] is not None for r in cols)
    n_rows = 2 if has_visual else 1

    cell_size = 4.2 if is_3d else 3.6
    fig = plt.figure(figsize=(cell_size * n_cols, cell_size * n_rows))

    subplot_kw = {"projection": "3d"} if is_3d else {}
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r][c] = fig.add_subplot(
                n_rows, n_cols, r * n_cols + c + 1, **subplot_kw,
            )

    for ci, res in enumerate(cols):
        if has_visual:
            ylabel = "Visual features" if ci == 0 else None
            scatter_features(
                axes[0][ci], res["visual_emb"], res["n_source"],
                title=res["method"], ylabel=ylabel, is_3d=is_3d,
            )
            ylabel = "Observation features" if ci == 0 else None
            scatter_features(
                axes[1][ci], res["obs_emb"], res["n_source"],
                title=None, ylabel=ylabel, is_3d=is_3d,
            )
        else:
            ylabel = "Observation features" if ci == 0 else None
            scatter_features(
                axes[0][ci], res["obs_emb"], res["n_source"],
                title=res["method"], ylabel=ylabel, is_3d=is_3d,
            )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=12,
                   markerfacecolor="#3b6cf2", markeredgewidth=0,
                   label="Source"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=12,
                   markerfacecolor="#e34234", markeredgewidth=0,
                   label="Target"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, -0.02),
    )

    if title is not None:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    if not is_3d:
        fig.tight_layout()
    else:
        top = 0.93 if title else 0.97
        fig.subplots_adjust(left=0.05, right=0.99, top=top, bottom=0.05,
                            wspace=wspace, hspace=0.02)

    if vertical_dividers and n_cols > 1:
        # Draw thin grey vertical lines in figure coordinates between
        # adjacent columns. Position each line at the midpoint between
        # the right edge of column c and the left edge of column c+1.
        from matplotlib.lines import Line2D
        # Re-evaluate axes positions AFTER subplots_adjust / tight_layout.
        fig.canvas.draw()
        for ci in range(n_cols - 1):
            left_box = axes[0][ci].get_position()
            right_box = axes[0][ci + 1].get_position()
            x = 0.5 * (left_box.x1 + right_box.x0)
            full_top = axes[0][ci].get_position().y1
            full_bot = axes[-1][ci].get_position().y0
            full_h = full_top - full_bot
            line_h = full_h * divider_height_frac
            mid = 0.5 * (full_top + full_bot)
            top_lim = mid + 0.5 * line_h
            bot_lim = mid - 0.5 * line_h
            line = Line2D(
                [x, x], [bot_lim, top_lim],
                transform=fig.transFigure,
                color="#888888", linewidth=1.2, linestyle="-",
                alpha=0.9,
            )
            fig.add_artist(line)

    if not os.path.isabs(output_path):
        output_path = os.path.join(REPO_ROOT, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "UMAP visualization of encoder features, source vs. target, "
            "for a set of checkpoints listed in a JSON manifest."
        )
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="JSON manifest listing the datasets and checkpoints to compare "
             "(see example_manifest.json).",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Short label used in the default output filename "
             "(default: manifest filename without extension).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="Number of obs to sample per domain (default: 1000).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the output figure. If omitted, defaults to "
             "scripts/feature_align/figures/feature_align_<label>_<ndim>d.png.",
    )
    parser.add_argument(
        "--ndim", type=int, default=3, choices=[2, 3],
        help="Number of UMAP components / scatter plot dimensions "
             "(default: 3).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for sampling and UMAP.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Output figure DPI.",
    )
    args = parser.parse_args()
    if args.label is None:
        args.label = os.path.splitext(os.path.basename(args.manifest))[0]
    if args.output is None:
        args.output = (
            f"scripts/feature_align/figures/feature_align_"
            f"{args.label}_{args.ndim}d.png"
        )

    # Make this script runnable from anywhere; ensure repo root is on path
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    main(args)
