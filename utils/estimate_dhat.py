"""
Localized discrepancy maximization for d_hat estimation.

Given a reference model h_ref (trained on target-only data), estimates the
discrepancy d_hat between source and target domains.

Algorithm
---------
1. Load h_ref from checkpoint (target-only).
2. Freeze encoder: theta_enc = theta_enc,ref.
3. Define local class:
     H_loc = {h_theta : theta_enc = theta_enc,ref,
              ||theta_free - theta_free,ref||_2 <= rho}
4. Projected gradient descent on:
     J_disc(theta) = -G(theta) + beta * ||theta_free - theta_free,ref||^2
   where  G(theta) = (1/n) sum_target l(h_theta(x),y)
                    - (1/m) sum_source l(h_theta(x),y)
   After each step, project theta_free back onto the rho-ball.
5. Every few steps, evaluate G_full on complete data and track maximum.
6. d_hat = max_t  G_full(theta_t).

Usage
-----
  python utils/estimate_dhat.py \
      --ckpt_path runs/target_only_stack_sim/<run>/models/model_epoch_300.pth \
      --source_data data/stack/rs_stack_500.hdf5 \
      --target_data data/stack/rsws_stack_20.hdf5 \
      --source_obs_key_map '{"agentview_image":"agentview45_image"}' \
      --name stack_rsws --rho_frac 0.05

"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import algo_factory
from robomimic.config import config_factory
from robomimic.utils.dataset import SequenceDataset


# Loss helpers

def compute_diffusion_loss(model, batch):
    """Forward pass returning the batch-mean diffusion MSE loss."""
    actions = batch["actions"]
    B = actions.shape[0]

    inputs = {"obs": batch["obs"], "goal": batch.get("goal_obs", {})}
    obs_features = TensorUtils.time_distributed(
        inputs, model.nets["policy"]["obs_encoder"], inputs_as_kwargs=True,
    )
    obs_cond = obs_features.flatten(start_dim=1)

    noise = torch.randn(actions.shape, device=model.device)
    timesteps = torch.randint(
        0,
        model.noise_scheduler.config.num_train_timesteps,
        (B,),
        device=model.device,
    ).long()

    noisy_actions = model.noise_scheduler.add_noise(actions, noise, timesteps)
    noise_pred = model.nets["policy"]["noise_pred_net"](
        noisy_actions, timesteps, global_cond=obs_cond,
    )
    return F.mse_loss(noise_pred, noise)


def evaluate_G_full(model, target_loader, source_loader,
                    obs_normalization_stats=None, num_passes=1):
    """Evaluate G(theta) on the *full* labeled data (no gradient).

    G = (1/n) sum_target l  -  (1/m) sum_source l

    Each pass draws fresh (noise, timestep) per sample.  When num_passes > 1,
    the per-sample loss is averaged over passes before computing G, which
    reduces the Monte-Carlo variance of the estimate and mitigates the upward
    bias that comes from taking max over noisy G values.

    Returns (G, mean_target_loss, mean_source_loss).
    """
    was_training = model.nets.training
    model.nets.eval()

    tgt_accum = np.zeros(num_passes)
    src_accum = np.zeros(num_passes)

    with torch.no_grad():
        for k in range(num_passes):
            total_tgt, n_tgt = 0.0, 0
            for batch in target_loader:
                inp = model.process_batch_for_training(batch)
                inp = model.postprocess_batch_for_training(
                    inp, obs_normalization_stats=obs_normalization_stats)
                loss = compute_diffusion_loss(model, inp)
                bs = batch["actions"].shape[0]
                total_tgt += loss.item() * bs
                n_tgt += bs
            tgt_accum[k] = total_tgt / max(n_tgt, 1)

            total_src, n_src = 0.0, 0
            for batch in source_loader:
                inp = model.process_batch_for_training(batch)
                inp = model.postprocess_batch_for_training(
                    inp, obs_normalization_stats=obs_normalization_stats)
                loss = compute_diffusion_loss(model, inp)
                bs = batch["actions"].shape[0]
                total_src += loss.item() * bs
                n_src += bs
            src_accum[k] = total_src / max(n_src, 1)

    if was_training:
        model.nets.train()

    mean_tgt = float(tgt_accum.mean())
    mean_src = float(src_accum.mean())
    return mean_tgt - mean_src, mean_tgt, mean_src


def compute_proximity(model, theta_ref):
    """||theta_free - theta_free,ref||^2"""
    prox = torch.tensor(0.0, device=next(iter(theta_ref.values())).device)
    for name, p in model.nets["policy"]["noise_pred_net"].named_parameters():
        prox = prox + (p - theta_ref[name]).pow(2).sum()
    return prox


@torch.no_grad()
def project_to_ball(model, theta_ref, rho):
    """Project theta_free onto the L2 ball {θ : ||θ - θ_ref||_2 ≤ ρ}.

    If the current displacement exceeds ρ, rescale:
        θ_free ← θ_ref + ρ * (θ_free - θ_ref) / ||θ_free - θ_ref||_2

    Returns (was_clipped, current_norm).
    """
    delta_sq = 0.0
    for name, p in model.nets["policy"]["noise_pred_net"].named_parameters():
        delta_sq += (p.data - theta_ref[name]).pow(2).sum().item()
    dist = delta_sq ** 0.5

    if dist > rho:
        scale = rho / dist
        for name, p in model.nets["policy"]["noise_pred_net"].named_parameters():
            p.data.copy_(theta_ref[name] + scale * (p.data - theta_ref[name]))
        return True, rho
    return False, dist


# Obs key remapping

def _translate_obs_keys(obs_keys, obs_key_map):
    """Translate canonical (model) obs keys to actual HDF5 keys.

    *obs_key_map*: ``{actual_hdf5_key: canonical_model_key}``, e.g.
    ``{"agentview_image": "agentview45_image"}``.
    """
    if not obs_key_map:
        return list(obs_keys)
    canonical_to_actual = {v: k for k, v in obs_key_map.items()}
    return [canonical_to_actual.get(k, k) for k in obs_keys]


class _RemappedDataset(torch.utils.data.Dataset):
    """Wraps a SequenceDataset, renaming obs keys from HDF5 names to model names."""

    def __init__(self, dataset, remap):
        self.dataset = dataset
        self._rmap = remap          # {hdf5_key: model_key}

    def _rename(self, d):
        return {self._rmap.get(k, k): v for k, v in d.items()}

    def __getitem__(self, index):
        batch = self.dataset[index]
        if "obs" in batch:
            batch["obs"] = self._rename(batch["obs"])
        if "next_obs" in batch:
            batch["next_obs"] = self._rename(batch["next_obs"])
        if "goal_obs" in batch:
            batch["goal_obs"] = self._rename(batch["goal_obs"])
        return batch

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


# Dataset creation

def make_dataset(hdf5_path, config, obs_keys, demo_limit=None,
                 obs_key_map=None):
    """Build a SequenceDataset using the training hyper-parameters from *config*.

    If *obs_key_map* is provided (``{hdf5_key: model_key}``), the dataset
    reads with HDF5-native keys and renames them to canonical model keys in
    the output batch — same pattern as ``create_single_dataset`` in train.py.
    """
    hdf5_obs_keys = _translate_obs_keys(obs_keys, obs_key_map)
    ds = SequenceDataset(
        hdf5_path=os.path.expanduser(hdf5_path),
        obs_keys=hdf5_obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
        load_next_obs=config.train.hdf5_load_next_obs,
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=config.train.hdf5_filter_key,
        demo_limit=demo_limit,
    )
    if obs_key_map:
        ds = _RemappedDataset(ds, obs_key_map)
    return ds


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Estimate d_hat via localized discrepancy maximization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="Path to target-only reference checkpoint (.pth)",
    )
    parser.add_argument(
        "--source_data", type=str, required=True,
        help="Path to source-domain HDF5 dataset",
    )
    parser.add_argument(
        "--target_data", type=str, required=True,
        help="Path to target-domain HDF5 dataset",
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Name of target domain",
    )

    # optimisation hyper-parameters
    parser.add_argument("--num_steps", type=int, default=200,
                        help="Gradient-descent steps (default: 200)")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="Proximity penalty weight (default: 0.01)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Evaluate G_full every N steps (default: 10)")
    parser.add_argument("--eval_passes", type=int, default=4,
                        help="Monte-Carlo passes per G_full evaluation; "
                             "averaging over K passes reduces the upward bias "
                             "from taking max over noisy estimates (default: 4)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (default: 128)")

    # ball constraint  ||theta_free - theta_free,ref||_2 <= rho
    rho_group = parser.add_mutually_exclusive_group()
    rho_group.add_argument(
        "--rho", type=float, default=None,
        help="Hard L2-ball radius for free params. "
             "Overrides --rho_frac if both given.")
    rho_group.add_argument(
        "--rho_frac", type=float, default=0.05,
        help="Set rho as a fraction of ||theta_free,ref||_2 "
             "(default: 0.05, i.e. 5%% of reference norm)")

    # obs key remapping
    parser.add_argument(
        "--source_obs_key_map", type=str, default=None,
        help='JSON dict mapping source HDF5 obs key to model canonical key, '
             'e.g. \'{"agentview_image":"agentview45_image"}\'. '
             'Needed when source HDF5 stores images under a different name '
             'than the model expects.')
    parser.add_argument(
        "--target_obs_key_map", type=str, default=None,
        help='Same as --source_obs_key_map but for the target dataset.')

    # data limits
    parser.add_argument("--source_demos", type=int, default=None,
                        help="Limit source demos (default: all)")
    parser.add_argument("--target_demos", type=int, default=None,
                        help="Limit target demos (default: all)")

    # output
    parser.add_argument("--output", type=str, default=None,
                        help="Path for the JSON summary file "
                             "(default: auto-generated next to checkpoint)")

    parser.add_argument("--seed", type=int, default=47,
                        help="Random seed (default: 47)")

    args = parser.parse_args()

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Load checkpoint
    ckpt_path = os.path.expanduser(args.ckpt_path)
    print("\nLoading reference checkpoint: {}".format(ckpt_path))
    ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path)
    raw_config = ckpt_dict["config"]
    if isinstance(raw_config, str):
        raw_config = json.loads(raw_config)
    shape_meta = ckpt_dict["shape_metadata"]

    config = config_factory(raw_config["algo_name"])
    config.unlock()
    config.update(raw_config)

    # Disable EMA: we evaluate G_full with current theta, not EMA weights,
    # and this also avoids diffusers EMAModel API version mismatches.
    config.algo.ema.enabled = False
    config.lock()

    ObsUtils.initialize_obs_utils_with_config(config)

    # Build model & load weights
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model_state = ckpt_dict["model"]
    model_state.pop("ema", None)
    model.deserialize(model_state)

    print("  algo_name : {}".format(config.algo_name))
    print("  checkpoint loaded OK")

    # Freeze encoder, store reference free params
    for p in model.nets["policy"]["obs_encoder"].parameters():
        p.requires_grad = False

    theta_free_ref = {
        name: p.data.clone()
        for name, p in model.nets["policy"]["noise_pred_net"].named_parameters()
    }

    n_enc = sum(p.numel() for p in model.nets["policy"]["obs_encoder"].parameters())
    n_free = sum(p.numel() for p in model.nets["policy"]["noise_pred_net"].parameters())
    theta_ref_norm = sum(v.pow(2).sum().item() for v in theta_free_ref.values()) ** 0.5
    print("  encoder params (frozen) : {:,}".format(n_enc))
    print("  free params (optimised) : {:,}".format(n_free))
    print("  ||theta_free,ref||_2    : {:.4f}".format(theta_ref_norm))

    # Resolve rho (hard ball radius)
    if args.rho is not None:
        rho = args.rho
    else:
        rho = args.rho_frac * theta_ref_norm
    print("  rho (ball radius)       : {:.4f}  ({:.2f}% of ref norm)".format(
        rho, 100.0 * rho / theta_ref_norm if theta_ref_norm > 0 else 0.0))

    # Datasets
    obs_keys = shape_meta["all_obs_keys"]

    src_obs_key_map = (json.loads(args.source_obs_key_map)
                       if args.source_obs_key_map else None)
    tgt_obs_key_map = (json.loads(args.target_obs_key_map)
                       if args.target_obs_key_map else None)

    source_ds = make_dataset(args.source_data, config, obs_keys,
                             demo_limit=args.source_demos,
                             obs_key_map=src_obs_key_map)
    target_ds = make_dataset(args.target_data, config, obs_keys,
                             demo_limit=args.target_demos,
                             obs_key_map=tgt_obs_key_map)

    print("\n  source : {} ({:,} samples)".format(args.source_data, len(source_ds)))
    if src_obs_key_map:
        print("           obs_key_map: {}".format(src_obs_key_map))
    print("  target : {} ({:,} samples)".format(args.target_data, len(target_ds)))
    if tgt_obs_key_map:
        print("           obs_key_map: {}".format(tgt_obs_key_map))

    # training loaders (shuffled, drop incomplete tail)
    src_train_loader = DataLoader(
        source_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )
    tgt_train_loader = DataLoader(
        target_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )

    # evaluation loaders (full data, deterministic order)
    src_eval_loader = DataLoader(
        source_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=0,
    )
    tgt_eval_loader = DataLoader(
        target_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=0,
    )

    # Optimizer (free params only)
    opt_params = [p for p in model.nets["policy"]["noise_pred_net"].parameters()
                  if p.requires_grad]
    optimizer = torch.optim.Adam(opt_params, lr=args.lr)

    # Initial evaluation
    G0, tgt0, src0 = evaluate_G_full(
        model, tgt_eval_loader, src_eval_loader,
        num_passes=args.eval_passes)
    print("\n--- Initial G_full (at reference checkpoint, {} MC passes) ---".format(
        args.eval_passes))
    print("  G_full = {:.6f}  (tgt_loss = {:.6f}, src_loss = {:.6f})".format(
        G0, tgt0, src0))

    # Optimisation loop
    print("\nStarting discrepancy maximisation")
    print("  num_steps  = {}".format(args.num_steps))
    print("  beta       = {}".format(args.beta))
    print("  lr         = {}".format(args.lr))
    print("  rho        = {:.4f}".format(rho))
    print("  eval_every = {}".format(args.eval_every))
    print("  eval_passes= {}\n".format(args.eval_passes))

    G_max = G0
    best_step = 0
    history = [dict(step=0, G_full=G0, G_max=G_max,
                    target_loss=tgt0, source_loss=src0, proximity=0.0)]

    src_iter = iter(src_train_loader)
    tgt_iter = iter(tgt_train_loader)
    t_start = time.time()

    for step in range(1, args.num_steps + 1):
        model.nets.train()

        # sample mini-batches (cycle on exhaustion)
        try:
            src_batch = next(src_iter)
        except StopIteration:
            src_iter = iter(src_train_loader)
            src_batch = next(src_iter)

        try:
            tgt_batch = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_train_loader)
            tgt_batch = next(tgt_iter)

        src_inp = model.process_batch_for_training(src_batch)
        src_inp = model.postprocess_batch_for_training(
            src_inp, obs_normalization_stats=None)
        tgt_inp = model.process_batch_for_training(tgt_batch)
        tgt_inp = model.postprocess_batch_for_training(
            tgt_inp, obs_normalization_stats=None)

        # G(theta) on mini-batch
        loss_tgt = compute_diffusion_loss(model, tgt_inp)
        loss_src = compute_diffusion_loss(model, src_inp)
        G_batch = loss_tgt - loss_src

        # proximity penalty
        prox = compute_proximity(model, theta_free_ref)

        # J_disc = -G + beta * ||theta_free - theta_free,ref||^2
        J_disc = -G_batch + args.beta * prox

        optimizer.zero_grad()
        J_disc.backward()
        optimizer.step()

        # project back onto rho-ball
        was_clipped, cur_dist = project_to_ball(model, theta_free_ref, rho)

        # periodic full evaluation
        if step % args.eval_every == 0 or step == args.num_steps:
            G_full, tgt_loss, src_loss = evaluate_G_full(
                model, tgt_eval_loader, src_eval_loader,
                num_passes=args.eval_passes,
            )
            prox_val = compute_proximity(model, theta_free_ref).item()

            if G_full > G_max:
                G_max = G_full
                best_step = step

            elapsed = time.time() - t_start
            dist = prox_val ** 0.5
            history.append(dict(
                step=step, G_full=G_full, G_max=G_max,
                target_loss=tgt_loss, source_loss=src_loss,
                proximity=prox_val, dist=dist,
                at_boundary=(dist >= rho * 0.999),
                G_batch=G_batch.item(), J_disc=J_disc.item(),
            ))

            clip_flag = "*" if dist >= rho * 0.999 else " "
            print("[step {:4d}/{:4d}]  G_full={:+.6f}  G_max={:+.6f}  "
                  "tgt={:.6f}  src={:.6f}  "
                  "dist={:.4f}/{:.4f}{} ({:.1f}s)".format(
                      step, args.num_steps, G_full, G_max,
                      tgt_loss, src_loss,
                      dist, rho, clip_flag, elapsed))

    total_time = time.time() - t_start
    d_hat = G_max

    # Result
    print("\n--- Result ---")
    print("  d_hat = {:.6f}   (best at step {})".format(d_hat, best_step))
    print("  total time : {:.1f}s".format(total_time))

    # Write summary
    summary = dict(
        d_hat=d_hat,
        best_step=best_step,
        initial_G=G0,
        config=dict(
            ckpt_path=args.ckpt_path,
            source_data=args.source_data,
            target_data=args.target_data,
            num_steps=args.num_steps,
            beta=args.beta,
            lr=args.lr,
            rho=rho,
            rho_frac=rho / theta_ref_norm if theta_ref_norm > 0 else 0.0,
            eval_every=args.eval_every,
            eval_passes=args.eval_passes,
            batch_size=args.batch_size,
            source_demos=args.source_demos,
            target_demos=args.target_demos,
            seed=args.seed,
        ),
        dataset_info=dict(
            source_samples=len(source_ds),
            target_samples=len(target_ds),
        ),
        model_info=dict(
            encoder_params=n_enc,
            free_params=n_free,
            theta_ref_norm=theta_ref_norm,
            algo_name=config.algo_name,
        ),
        total_time_seconds=round(total_time, 2),
        history=history,
    )

    output_path = args.output
    if output_path is None:
        out_dir = f"utils/dhat_results"
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(out_dir, f"dhat_summary_{args.name}_{ts}.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary written to: {}".format(os.path.abspath(output_path)))


if __name__ == "__main__":
    main()
