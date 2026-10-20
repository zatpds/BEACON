"""
Training script for robomimic Diffusion Policy with optional co-training.

Supports six training modes selected via --cotraining:

  off (default)  Standard single-domain BC training.
  vanilla        Convex combination: L = alpha * L_src + (1-alpha) * L_tgt.
  ot             Unbalanced Optimal Transport alignment on encoder features.
  mmd            Energy-distance MMD alignment on encoder features.
  bea            Best-Effort Adaptation (q-weighted ERM via alternating min).
  msbea          Multi-source BEA (one q per source domain).

Supported tasks (--task):

  stack (default)  Cube stacking (Stack / StackWood environments)
  mug              Mug cleanup into drawer (MugCleanup / MugCleanupWood)
  thread           Needle threading (Threading / ThreadingWood environments)
  luggage          Push football over luggage ridge (LuggagePush; the wood
                   variant is the same env with table_style/luggage_style)

Demo limits are controlled by the "demo_limit" field in each data entry of the
config JSON.  If omitted, all demos in the HDF5 file are used.

Usage examples (see configs/README.md for how to adapt a config to another
task or domain pair):

  # BEACON, sim-to-sim (texture gap)
  python train.py --config configs/sim/beacon.json --task stack --cotraining bea

  # BEACON / MS-BEACON, sim-to-real
  python train.py --config configs/real/beacon.json   --task stack --cotraining bea
  python train.py --config configs/real/msbeacon.json --task stack --cotraining msbea

  # Baselines
  python train.py --config configs/sim/target_only.json --task stack
  python train.py --config configs/sim/cotrain.json --task stack --cotraining vanilla --alpha 0.85
  python train.py --config configs/sim/mmd.json     --task stack --cotraining mmd
  python train.py --config configs/sim/uot.json     --task stack --cotraining ot
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from collections import OrderedDict
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.config import config_factory
from robomimic.envs.wrappers import EnvWrapper as _EnvWrapper
from robomimic.utils.dataset import MetaDataset, SequenceDataset
from robomimic.utils.log_utils import DataLogger, PrintLogger, flush_warnings
from robomimic.utils.ot_dataset import ChunkSamplingDTWTrainDataset


def register_custom_task_envs(task):
    """Import side-effecting modules that register custom robosuite envs.

    Some tasks (e.g. ``luggage``) are implemented as in-repo robosuite
    environments that are only added to ``REGISTERED_ENVS`` when their module
    is imported.  Robomimic builds rollout envs via ``robosuite.make`` using the
    ``env_name`` stored in the dataset, so we must import the module before any
    env is created.
    """
    if task == "luggage":
        # Importing the module registers ``LuggagePush`` with robosuite.
        import scripts.rs_collect.luggage_env.luggage_push  # noqa: F401


# Rollout env helpers (obs key remapping)

class ObsKeyRemapWrapper(_EnvWrapper):
    """Thin wrapper that renames observation dict keys during rollout.

    Used when the rollout env produces camera obs under a different key than
    what the policy expects (e.g. ``agentview45_image`` → ``agentview_image``).

    Inherits from robomimic's ``EnvWrapper`` so that ``isinstance`` checks
    in ``run_rollout`` pass correctly.
    """

    def __init__(self, env, key_map):
        super().__init__(env)
        self._key_map = key_map

    def _remap(self, obs):
        return {self._key_map.get(k, k): v for k, v in obs.items()}

    def reset(self):
        return self._remap(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._remap(obs), reward, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __repr__(self):
        return "ObsKeyRemap({}, {})".format(self.env, self._key_map)


def _build_obs_key_map(env_meta, config):
    """Auto-detect obs key remapping for rollout envs with non-standard cameras.

    Example: env has camera ``agentview45`` (producing ``agentview45_image``)
    but policy expects ``agentview_image`` → returns
    ``{"agentview45_image": "agentview_image"}``.

    Only applies to robosuite envs.
    """
    from robomimic.envs.env_base import EnvType
    if env_meta.get("type") != EnvType.ROBOSUITE_TYPE:
        return {}

    expected_rgb = list(config.observation.modalities.obs.rgb)
    cam_names = env_meta.get("env_kwargs", {}).get("camera_names", [])

    if not cam_names or not expected_rgb:
        return {}

    actual_keys = {"{}_image".format(cam) for cam in cam_names}
    key_map = {}
    used_actual = set()

    for expected_key in expected_rgb:
        if expected_key in actual_keys:
            continue
        base_cam = expected_key.replace("_image", "")
        for cam in cam_names:
            actual_key = "{}_image".format(cam)
            if actual_key not in used_actual and cam.startswith(base_cam):
                key_map[actual_key] = expected_key
                used_actual.add(actual_key)
                print("  [ObsKeyRemap] {} → {}".format(actual_key, expected_key))
                break

    return key_map


# Per-dataset obs key remapping (training data loading)

def _translate_obs_keys(obs_keys, obs_key_map):
    """Translate canonical obs keys to actual HDF5 keys.

    *obs_key_map* maps actual (HDF5/env) key → canonical (model) key, e.g.
    ``{"agentview_image": "agentview45_image"}``.  This is the same direction
    used by ``ObsKeyRemapWrapper`` and ``_build_obs_key_map``.

    To find the HDF5 key for a canonical key we invert the lookup.
    Returns a new list with translated keys, preserving order.
    """
    if not obs_key_map:
        return list(obs_keys)
    canonical_to_actual = {v: k for k, v in obs_key_map.items()}
    return [canonical_to_actual.get(k, k) for k in obs_keys]


class _RemappedDataset(torch.utils.data.Dataset):
    """Wraps a SequenceDataset, renaming obs keys in the output batch.

    Used when the HDF5 has obs stored under different key names than the
    canonical names the model expects (e.g. ``agentview_image`` in HDF5
    but the model expects ``agentview45_image``).
    """

    def __init__(self, dataset, reverse_map):
        """
        Args:
            dataset: underlying SequenceDataset (reads HDF5-native keys).
            reverse_map: dict mapping actual-HDF5-key → canonical-model-key.
        """
        self.dataset = dataset
        self._rmap = reverse_map

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

    def get_obs_normalization_stats(self):
        from copy import deepcopy
        stats = self.dataset.get_obs_normalization_stats()
        return {self._rmap.get(k, k): v for k, v in stats.items()}

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __repr__(self):
        return "_RemappedDataset({}, {})".format(self.dataset, self._rmap)


# OT dataset helper

class _RemappedOTDataset(ChunkSamplingDTWTrainDataset):
    """ChunkSamplingDTWTrainDataset with per-source obs key remapping.

    When the two HDF5 files (ot_src / ot_tgt) store the same camera image
    under different key names, this subclass transparently translates keys
    so the model always sees canonical names.
    """

    def __init__(self, *args, per_ds_obs_key_map=None, **kwargs):
        self._per_ds_fwd = per_ds_obs_key_map or {}
        self._per_ds_inv = {
            ds: {v: k for k, v in m.items()}
            for ds, m in self._per_ds_fwd.items()
        }
        super().__init__(*args, **kwargs)

    def get_obs_sequence_from_demo(self, demo_id, dataset_name, index_in_demo,
                                   keys, num_frames_to_stack=0, seq_length=1,
                                   prefix="obs"):
        inv = self._per_ds_inv.get(dataset_name, {})
        hdf5_keys = tuple(inv.get(k, k) for k in keys)

        obs, pad_mask = self.get_sequence_from_demo(
            demo_id=demo_id,
            dataset_name=dataset_name,
            index_in_demo=index_in_demo,
            keys=tuple("{}/{}".format(prefix, k) for k in hdf5_keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        obs = {k.split("/")[1]: obs[k] for k in obs}

        fwd = self._per_ds_fwd.get(dataset_name, {})
        if fwd:
            obs = {fwd.get(k, k): v for k, v in obs.items()}

        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask
        return obs


def get_ot_dataset(config, obs_keys, per_ds_obs_key_map=None):
    """Create the paired OT dataset from config.

    *per_ds_obs_key_map*: optional dict mapping dataset name (``"ot_src"`` /
    ``"ot_tgt"``) to an obs_key_map ``{actual_hdf5_key: canonical_key}``.
    """
    no_window = config.algo.ot.no_window
    cls = _RemappedOTDataset if per_ds_obs_key_map else ChunkSamplingDTWTrainDataset
    kwargs = dict(
        dataset_to_paths=config.train.ot.dataset,
        pair_info_path=config.train.ot.pair_info_path,
        sharpness=config.algo.ot.sharpness,
        cutoff=config.algo.ot.cutoff,
        window_size=config.algo.ot.window_size,
        no_window=no_window,
        dataset_masks=config.train.ot.dataset_masks,
        obs_keys=obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
        load_next_obs=config.train.hdf5_load_next_obs,
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=True,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=None,
    )
    if per_ds_obs_key_map:
        kwargs["per_ds_obs_key_map"] = per_ds_obs_key_map
    return cls(**kwargs)


# Safe experiment-directory helper (never overwrites previous runs)

def get_exp_dir(config, resume=False):
    """
    Create experiment directory.  Unlike the upstream robomimic helper this
    **never** prompts the user and **never** deletes existing run folders.
    Every new run gets its own timestamped sub-directory.

    Returns:
        log_dir, ckpt_dir, video_dir, time_dir
    """
    time_str = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5))).strftime("%m%d-%H:%M:%S-%Y")

    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)

    if resume:
        if not os.path.exists(base_output_dir):
            raise FileNotFoundError(
                "Resuming training, but output dir {} does not exist".format(base_output_dir)
            )
        # pick the most-recent existing run
        time_str = sorted(os.listdir(base_output_dir))[-1]

    time_dir = os.path.join(base_output_dir, time_str)

    ckpt_dir = None
    if config.experiment.save.enabled:
        ckpt_dir = os.path.join(time_dir, "models")
        os.makedirs(ckpt_dir, exist_ok=resume)

    log_dir = os.path.join(time_dir, "logs")
    os.makedirs(log_dir, exist_ok=resume)

    video_dir = os.path.join(time_dir, "videos")
    os.makedirs(video_dir, exist_ok=resume)

    return log_dir, ckpt_dir, video_dir, time_dir


# Co-training helpers

def compute_diffusion_loss(model, batch):
    """
    Forward pass of DiffusionPolicyUNet returning the loss *tensor*
    (still in the computation graph, suitable for further combination
    before calling backward).

    Args:
        model: a DiffusionPolicyUNet algo instance (with .nets, .noise_scheduler, etc.)
        batch: a processed & postprocessed batch dict on the correct device

    Returns:
        loss (torch.Tensor): scalar MSE loss between predicted and true noise
    """
    actions = batch["actions"]
    B = actions.shape[0]

    inputs = {"obs": batch["obs"], "goal": batch["goal_obs"]}
    obs_features = TensorUtils.time_distributed(
        inputs, model.nets["policy"]["obs_encoder"], inputs_as_kwargs=True
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
        noisy_actions, timesteps, global_cond=obs_cond
    )

    return F.mse_loss(noise_pred, noise)


def create_single_dataset(config, obs_keys, hdf5_path, demo_limit=None,
                          obs_key_map=None, cache_mode_override=None):
    """
    Create a single SequenceDataset for one domain.

    Uses the training hyper-parameters from *config* (seq_length,
    frame_stack, cache mode, etc.) but overrides the HDF5 path
    and optionally limits the number of demos.

    If *obs_key_map* is provided (actual-HDF5 → canonical-model, same
    direction as ``ObsKeyRemapWrapper``), the SequenceDataset reads the
    actual HDF5 keys and the returned dataset renames them to canonical.
    """
    hdf5_obs_keys = _translate_obs_keys(obs_keys, obs_key_map)
    cache = cache_mode_override or config.train.hdf5_cache_mode
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
        hdf5_cache_mode=cache,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=config.train.hdf5_filter_key,
        demo_limit=demo_limit,
    )
    if obs_key_map:
        ds = _RemappedDataset(ds, obs_key_map)
    return ds


def run_cotrain_epoch(
    model,
    source_loader,
    target_loader,
    epoch,
    alpha,
    num_steps=None,
    obs_normalization_stats=None,
):
    """
    Run one epoch of vanilla co-training.

    L_total = α · L(θ; D_source) + (1 − α) · L(θ; D_target)

    Args:
        model: algo instance (DiffusionPolicyUNet)
        source_loader: DataLoader for the source domain
        target_loader: DataLoader for the target domain
        epoch: current epoch number
        alpha: weighting coefficient for the source loss
        num_steps: number of gradient steps per epoch
        obs_normalization_stats: optional observation normalisation stats
    """
    epoch_timestamp = time.time()
    model.set_train()

    if num_steps is None:
        num_steps = max(len(source_loader), len(target_loader))

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])

    src_iter = iter(source_loader)
    tgt_iter = iter(target_loader)

    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # Load batches
        t = time.time()
        try:
            src_batch = next(src_iter)
        except StopIteration:
            src_iter = iter(source_loader)
            src_batch = next(src_iter)

        try:
            tgt_batch = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(target_loader)
            tgt_batch = next(tgt_iter)
        timing_stats["Data_Loading"].append(time.time() - t)

        # Process batches
        t = time.time()
        src_input = model.process_batch_for_training(src_batch)
        src_input = model.postprocess_batch_for_training(
            src_input, obs_normalization_stats=obs_normalization_stats
        )
        tgt_input = model.process_batch_for_training(tgt_batch)
        tgt_input = model.postprocess_batch_for_training(
            tgt_input, obs_normalization_stats=obs_normalization_stats
        )
        timing_stats["Process_Batch"].append(time.time() - t)

        # Forward + combined backward
        t = time.time()

        loss_source = compute_diffusion_loss(model, src_input)
        loss_target = compute_diffusion_loss(model, tgt_input)
        loss_total = alpha * loss_source + (1.0 - alpha) * loss_target

        # manual backprop (mirrors TorchUtils.backprop_for_loss)
        optimizer = model.optimizers["policy"]
        optimizer.zero_grad()
        loss_total.backward()

        # optional gradient clipping
        max_grad_norm = model.global_config.train.max_grad_norm
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.nets.parameters(), max_grad_norm)

        # compute grad norms for logging
        grad_norms = 0.0
        for p in model.nets.parameters():
            if p.grad is not None:
                grad_norms += p.grad.data.norm(2).pow(2).item()

        optimizer.step()

        # EMA update
        if model.ema is not None:
            model.ema.step(model.nets)

        timing_stats["Train_Batch"].append(time.time() - t)
        model.on_gradient_step()

        # Logging
        t = time.time()
        step_log = OrderedDict(
            Loss=loss_total.item(),
            Source_Loss=loss_source.item(),
            Target_Loss=loss_target.item(),
            Policy_Grad_Norms=grad_norms,
        )
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # Aggregate epoch metrics
    step_log_dict = {}
    for entry in step_log_all:
        for k, v in entry.items():
            step_log_dict.setdefault(k, []).append(v)
    step_log_all = {k: float(np.mean(v)) for k, v in step_log_dict.items()}

    for k in timing_stats:
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.0
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.0

    return step_log_all


# BEA (Best-Effort Adaptation) helpers

def run_bea_q_update_pass(model, data_loader, obs_normalization_stats=None):
    """
    Phase 1 of BEA: update q weights with model h fixed.

    Supports two modes selected by ``config.algo.bea.q.mode``:

      ``"full"`` (paper-faithful) — collect all per-sample losses in a single
      forward pass, then solve the global convex optimization problem for q
      exactly via closed-form + bisection.

      ``"minibatch"`` — iterate over mini-batches, apply a projected
      subgradient step to q for each batch, then project onto the
      bounded simplex.

    Args:
        model: BEADiffusionPolicyUNet instance (with q, p0, d initialized).
        data_loader: DataLoader over the combined MetaDataset.
        obs_normalization_stats: optional obs normalization stats dict.

    Returns:
        q_update_stats (dict): statistics about the q update for logging.
    """
    cfg = model.algo_config.bea
    mode = cfg.q.mode

    model.set_eval()
    num_updated = 0
    all_losses_list = []
    all_indices_list = []

    # Compute R_DP(h) once (h is fixed during the q-update phase).
    # gamma is the weight-decay coefficient from the optimizer config.
    gamma = model._get_gamma()
    if gamma > 0:
        with torch.no_grad():
            r_dp_value = model.compute_r_dp().item()
    else:
        r_dp_value = 0.0

    for batch in data_loader:
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(
            input_batch, obs_normalization_stats=obs_normalization_stats
        )
        indices, per_sample_losses = model.compute_losses_for_q_update(input_batch)

        if mode == "minibatch":
            model.update_q(indices, per_sample_losses, r_dp_value=r_dp_value)

        num_updated += len(indices)
        all_indices_list.append(indices.detach().cpu())
        all_losses_list.append(per_sample_losses.detach().cpu())

    all_indices = torch.cat(all_indices_list)
    all_losses = torch.cat(all_losses_list)

    if mode == "full":
        # Assemble a full loss vector indexed 0..m+n-1.  Samples may appear
        # in multiple batches (unlikely with drop_last=False, shuffle=False)
        # so we average in that case.
        total = model.m + model.n
        loss_vec = torch.zeros(total, dtype=torch.float32)
        count_vec = torch.zeros(total, dtype=torch.float32)
        loss_vec.scatter_add_(0, all_indices.long(), all_losses.float())
        count_vec.scatter_add_(0, all_indices.long(), torch.ones_like(all_losses))
        count_vec.clamp_(min=1.0)
        loss_vec /= count_vec

        model.solve_q_global(loss_vec, r_dp_value=r_dp_value)
    else:
        model.project_q_simplex()

    m, n = model.m, model.n
    q = model.q

    stats = {
        "q_update_mode": mode,
        "q_update_num_samples": num_updated,
        "q_update_mean_loss": all_losses.mean().item(),
        "q_global_mean": q.mean().item(),
        "q_global_std": q.std().item(),
        "q_global_min": q.min().item(),
        "q_global_max": q.max().item(),
        "q_source_mean": q[:m].mean().item() if m > 0 else 0.0,
        "q_target_mean": q[m:].mean().item() if n > 0 else 0.0,
        "q_source_nonzero_frac": (q[:m] > 1e-8).float().mean().item() if m > 0 else 0.0,
        "q_total_mass": q.sum().item(),
        "r_dp": r_dp_value,
        "q_inf": q.max().item(),
    }
    return stats


def run_bea_epoch(
    model,
    data_loader,
    epoch,
    num_steps=None,
    obs_normalization_stats=None,
):
    """
    Phase 2 of BEA: update model h with q fixed (q-weighted ERM).

    Standard training loop but using BEADiffusionPolicyUNet.train_on_batch,
    which computes q-weighted per-sample loss.

    Args:
        model: BEADiffusionPolicyUNet instance.
        data_loader: DataLoader over the combined MetaDataset.
        epoch: current epoch number.
        num_steps: gradient steps per epoch (None = full dataset pass).
        obs_normalization_stats: optional obs normalization stats dict.

    Returns:
        step_log_all (dict): aggregated epoch metrics.
    """
    epoch_timestamp = time.time()
    model.set_train()

    if num_steps is None:
        num_steps = len(data_loader)

    # Scale the optimizer's weight_decay by ||q||_inf for this epoch.
    # The joint objective has gamma * ||q||_inf * R_DP(h), so during the
    # h-update (q fixed) the effective weight decay is gamma * ||q||_inf.
    optimizer = model.optimizers["policy"]
    gamma = model._get_gamma()
    q_inf = model.q.max().item()
    orig_wd = optimizer.param_groups[0]["weight_decay"]
    scaled_wd = gamma * q_inf
    for pg in optimizer.param_groups:
        pg["weight_decay"] = scaled_wd

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])

    data_loader_iter = iter(data_loader)
    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # Load batch
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)
        timing_stats["Data_Loading"].append(time.time() - t)

        # Process batch
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(
            input_batch, obs_normalization_stats=obs_normalization_stats
        )
        timing_stats["Process_Batch"].append(time.time() - t)

        # q-weighted forward + backward
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=False)
        timing_stats["Train_Batch"].append(time.time() - t)
        model.on_gradient_step()

        # Logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # Restore original weight_decay
    for pg in optimizer.param_groups:
        pg["weight_decay"] = orig_wd

    # Aggregate epoch metrics
    step_log_dict = {}
    for entry in step_log_all:
        for k, v in entry.items():
            step_log_dict.setdefault(k, []).append(v)
    step_log_all = {k: float(np.mean(v)) for k, v in step_log_dict.items()}

    for k in timing_stats:
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.0
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.0

    return step_log_all


# MSBEA (Multi-Source BEA) helpers

def run_msbea_q_update_pass(model, data_loader, obs_normalization_stats=None):
    """
    MSBEA weight-update pass: update w (source domain weights), then
    q_tilde (within-source sample weights) and q_T (target weights).

    1. Collect all per-sample losses in a full forward pass.
    2. Update w using full-dataset gradients.
    3. Update q_tilde and q_T with per-batch subgradient steps.
    4. Project q_tilde and q_T onto their constraint sets.

    Returns:
        stats (dict): update statistics for logging.
    """
    model.set_eval()
    all_losses_list = []
    all_indices_list = []

    for batch in data_loader:
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(
            input_batch, obs_normalization_stats=obs_normalization_stats
        )
        indices, per_sample_losses = model.compute_losses_for_q_update(input_batch)
        all_indices_list.append(indices.detach().cpu())
        all_losses_list.append(per_sample_losses.detach().cpu())

    all_indices = torch.cat(all_indices_list)
    all_losses = torch.cat(all_losses_list)

    # Assemble full loss vector indexed 0..m_total+n-1
    total = model.m_total + model.n
    loss_vec = torch.zeros(total, dtype=torch.float32)
    count_vec = torch.zeros(total, dtype=torch.float32)
    loss_vec.scatter_add_(0, all_indices.long(), all_losses.float())
    count_vec.scatter_add_(0, all_indices.long(), torch.ones_like(all_losses))
    count_vec.clamp_(min=1.0)
    loss_vec /= count_vec

    # Phase 1b: update w (needs full loss vector)
    model.update_w(loss_vec)

    # Phase 1c: update q_tilde and q_T (minibatch subgradient)
    for batch in data_loader:
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(
            input_batch, obs_normalization_stats=obs_normalization_stats
        )
        indices, per_sample_losses = model.compute_losses_for_q_update(input_batch)
        model.update_q_msbea(indices, per_sample_losses)

    # Project onto constraint sets
    model.project_q_simplex()

    # Collect stats
    q_eff = model.get_q_eff()
    m = model.m_total
    n = model.n
    stats = {
        "w_values": model.w.tolist(),
        "q_eff_source_mean": q_eff[:m].mean().item(),
        "q_eff_source_std": q_eff[:m].std().item(),
        "q_eff_target_mean": q_eff[m:].mean().item() if n > 0 else 0.0,
        "q_eff_target_std": q_eff[m:].std().item() if n > 0 else 0.0,
        "q_eff_mean": q_eff.mean().item(),
        "q_eff_mass": q_eff.sum().item(),
        "q_eff_inf": q_eff.max().item(),
        "mean_loss": all_losses.mean().item(),
    }

    # w gradient diagnostics
    if hasattr(model, '_last_w_grad') and model._last_w_grad is not None:
        stats["w_grad"] = model._last_w_grad.tolist()
        stats["w_eta_eff"] = model._last_w_eta

    for s, sz in enumerate(model.source_sizes):
        mask = model.domain_id == s
        stats[f"w_{s}"] = model.w[s].item()
        qe_s = q_eff[:m][mask]
        stats[f"q_eff_src{s}_mean"] = qe_s.mean().item()
        stats[f"q_eff_src{s}_std"] = qe_s.std().item()
        stats[f"q_eff_src{s}_min"] = qe_s.min().item()
        stats[f"q_eff_src{s}_max"] = qe_s.max().item()
        d_s = model.d[:m][mask]
        stats[f"d_src{s}_mean"] = d_s.mean().item()
        stats[f"d_src{s}_std"] = d_s.std().item()

    return stats


# Main training function

def train(config, device, args, resume=False):
    """
    Main training entry point.

    Dispatches to the appropriate training loop based on args.cotraining:
      - "off":     standard single-domain BC via TrainUtils.run_epoch
      - "vanilla": alpha-weighted co-training via run_cotrain_epoch
      - "ot":      OT alignment via TrainUtils.run_epoch_for_ot_policy
      - "mmd":     MMD alignment via TrainUtils.run_epoch_for_ot_policy
      - "bea":     alternating q-update + q-weighted ERM

    Also handles dataset setup, rollout environments, checkpointing, and
    logging for all five modes.

    Args:
        config: robomimic Config object (locked)
        device: torch device
        args:   parsed CLI arguments (includes cotraining, alpha)
        resume: if True, resume from the latest checkpoint in the run directory
    """
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(2)

    print("\n--- New Training Run with Config ---")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, time_dir = get_exp_dir(config, resume=resume)

    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    ObsUtils.initialize_obs_utils_with_config(config)

    vanilla_mode = args.cotraining == "vanilla"
    bea_mode = args.cotraining == "bea"
    msbea_mode = args.cotraining == "msbea"
    mmd_mode = args.cotraining == "mmd"
    ot_mode = args.cotraining == "ot"
    paired_mode = ot_mode or mmd_mode

    # Environment metadata & shapes
    from robomimic.utils.python_utils import deep_update

    env_meta_list = []
    shape_meta_list = []
    if isinstance(config.train.data, str):
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                "Dataset at provided path {} not found!".format(dataset_path)
            )
        print("\n--- Loaded Environment Metadata ---")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
        env_meta["lang"] = dataset_cfg.get("lang", "dummy")

        # Per-dataset env_meta_update_dict takes priority (for co-training),
        # then fall back to the global one.
        per_ds_update = dataset_cfg.get("env_meta_update_dict", None)
        if per_ds_update is not None:
            deep_update(env_meta, per_ds_update)
        else:
            deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        ds_obs_key_map = dataset_cfg.get("obs_key_map", None)
        effective_obs_keys = _translate_obs_keys(config.all_obs_keys, ds_obs_key_map)
        if ds_obs_key_map:
            for actual_key, canonical_key in ds_obs_key_map.items():
                if actual_key not in ObsUtils.OBS_KEYS_TO_MODALITIES:
                    ObsUtils.OBS_KEYS_TO_MODALITIES[actual_key] = \
                        ObsUtils.OBS_KEYS_TO_MODALITIES[canonical_key]
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_config=dataset_cfg,
            action_keys=config.train.action_keys,
            all_obs_keys=effective_obs_keys,
            verbose=True,
        )
        if ds_obs_key_map:
            shape_meta["all_shapes"] = OrderedDict(
                (ds_obs_key_map.get(k, k), v)
                for k, v in shape_meta["all_shapes"].items()
            )
            shape_meta["all_obs_keys"] = [
                ds_obs_key_map.get(k, k) for k in shape_meta["all_obs_keys"]
            ]
            print("  [ObsKeyMap] dataset {}: {}".format(
                dataset_cfg["path"], ds_obs_key_map))
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        env_meta = env_meta_list[0].copy()
        env_meta["env_name"] = config.experiment.env
        env_meta_list = [env_meta]
        print("\n--- Replacing Env to {} ---".format(env_meta["env_name"]))

    # Rollout environments
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        rollout_domains = config.experiment.rollout.get("domains", None)

        for env_i in range(len(env_meta_list)):
            if rollout_domains is not None:
                domain_label = "src" if env_i == 0 else "tgt"
                if domain_label not in rollout_domains:
                    continue
            dataset_cfg = config.train.data[env_i]
            if not dataset_cfg.get("eval", True):
                continue
            env_meta = env_meta_list[env_i]
            shape_meta = shape_meta_list[env_i]
            env_names = [env_meta["env_name"]]
            if (env_i == 0) and (config.experiment.additional_envs is not None):
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            def create_env(env_name):
                env_kwargs = dict(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"] or shape_meta["use_depths"],
                )
                env = EnvUtils.create_env_from_metadata(**env_kwargs)
                env = EnvUtils.wrap_env_from_config(env, config=config)
                return env

            obs_key_map = dataset_cfg.get("obs_key_map", None)
            if obs_key_map is None:
                obs_key_map = _build_obs_key_map(env_meta, config)

            for env_name in env_names:
                env = create_env(env_name)
                if obs_key_map:
                    env = ObsKeyRemapWrapper(env, obs_key_map)
                env_key = (
                    os.path.splitext(os.path.basename(dataset_cfg["path"]))[0]
                    if not dataset_cfg.get("key", None)
                    else dataset_cfg["key"]
                )
                envs[env_key] = env
                print(env)

    print("")

    # Datasets & DataLoaders
    obs_keys = shape_meta_list[0]["all_obs_keys"]

    if paired_mode:
        # OT co-training: BC dataset + paired OT dataset
        src_ds_cfg = config.train.data[0]
        tgt_ds_cfg = config.train.data[1]
        src_path = os.path.expanduser(src_ds_cfg["path"])
        tgt_path = os.path.expanduser(tgt_ds_cfg["path"])
        src_demo_limit = src_ds_cfg.get("demo_limit", None)
        tgt_demo_limit = tgt_ds_cfg.get("demo_limit", None)

        _paired_label = "MMD" if mmd_mode else "OT"
        print("\n--- {} Co-training Setup ---".format(_paired_label))
        print("  mode   : {}".format(_paired_label.lower()))
        print("  source : key={}, demos={}, path={}".format(
            src_ds_cfg.get("key", "src"), src_demo_limit or "all", src_path))
        print("  target : key={}, demos={}, path={}".format(
            tgt_ds_cfg.get("key", "tgt"), tgt_demo_limit or "all", tgt_path))
        print("")

        ot_masks = {}
        if src_demo_limit is not None:
            ot_masks["ot_src"] = src_demo_limit
        if tgt_demo_limit is not None:
            ot_masks["ot_tgt"] = tgt_demo_limit
        with config.values_unlocked():
            config.train.ot.dataset_masks = ot_masks or None

        # -- BC trainset (obs_key_map-aware) --
        src_bc = create_single_dataset(
            config, obs_keys, hdf5_path=src_path,
            demo_limit=src_demo_limit,
            obs_key_map=src_ds_cfg.get("obs_key_map", None),
        )
        tgt_bc = create_single_dataset(
            config, obs_keys, hdf5_path=tgt_path,
            demo_limit=tgt_demo_limit,
            obs_key_map=tgt_ds_cfg.get("obs_key_map", None),
        )
        ds_weights = [src_ds_cfg.get("weight", 1.0),
                      tgt_ds_cfg.get("weight", 1.0)]
        trainset = MetaDataset(
            datasets=[src_bc, tgt_bc],
            ds_weights=ds_weights,
            normalize_weights_by_ds_size=config.train.normalize_weights_by_ds_size,
        )
        validset = None
        train_sampler = trainset.get_dataset_sampler()
        print("\n--- BC Training Dataset ---")
        print(trainset)
        print("")

        # -- OT paired dataset (obs_key_map-aware) --
        per_ds_obs_key_map = {}
        if src_ds_cfg.get("obs_key_map", None):
            per_ds_obs_key_map["ot_src"] = src_ds_cfg["obs_key_map"]
        if tgt_ds_cfg.get("obs_key_map", None):
            per_ds_obs_key_map["ot_tgt"] = tgt_ds_cfg["obs_key_map"]
        ot_dataset = get_ot_dataset(
            config, obs_keys=obs_keys,
            per_ds_obs_key_map=per_ds_obs_key_map if per_ds_obs_key_map else None,
        )
        ot_loader = DataLoader(
            dataset=ot_dataset,
            batch_size=int(0.5 * config.train.ot.batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=config.train.num_data_workers,
        )
        print("\n--- {} Paired Dataset ---".format(_paired_label))
        print(ot_dataset)
        print("")

        ot_params = dict(
            emb_scale=config.algo.ot.emb_scale,
            cost_scale=config.algo.ot.cost_scale,
            reg=config.algo.ot.reg,
            tau1=config.algo.ot.tau1,
            tau2=config.algo.ot.tau2,
            scale=config.algo.ot.scale,
            heuristic=config.algo.ot.heuristic,
            label=config.algo.ot.label,
        )

        obs_normalization_stats = None
        if config.train.hdf5_normalize_obs:
            obs_normalization_stats = trainset.get_obs_normalization_stats()
        action_normalization_stats = trainset.get_action_normalization_stats()

        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True,
        )

        source_loader = None
        target_loader = None

    elif bea_mode:
        # BEA: combined source+target in a single MetaDataset
        src_ds_cfg = config.train.data[0]
        tgt_ds_cfg = config.train.data[1]
        src_path = os.path.expanduser(src_ds_cfg["path"])
        tgt_path = os.path.expanduser(tgt_ds_cfg["path"])
        src_demo_limit = src_ds_cfg.get("demo_limit", None)
        tgt_demo_limit = tgt_ds_cfg.get("demo_limit", None)
        src_key = src_ds_cfg.get("key", "src")
        tgt_key = tgt_ds_cfg.get("key", "tgt")

        print("\n--- BEA Co-training Setup ---")
        print("  mode   : bea (Best-Effort Adaptation)")
        print("  source : key={}, demos={}, path={}".format(
            src_key, src_demo_limit or "all", src_path))
        print("  target : key={}, demos={}, path={}".format(
            tgt_key, tgt_demo_limit or "all", tgt_path))
        print("")

        # hdf5_cache_mode="all" is not supported by MetaDataset (it merges
        # action normalization stats after init, which invalidates the cache).
        # Force "low_dim" to avoid a hard assertion error.
        effective_cache_mode = config.train.hdf5_cache_mode
        if effective_cache_mode == "all":
            print("[BEA] WARNING: hdf5_cache_mode='all' not supported with "
                  "MetaDataset. Falling back to 'low_dim'.")
            effective_cache_mode = "low_dim"

        source_dataset = create_single_dataset(
            config, obs_keys, hdf5_path=src_path, demo_limit=src_demo_limit,
            obs_key_map=src_ds_cfg.get("obs_key_map", None),
            cache_mode_override=effective_cache_mode,
        )
        target_dataset = create_single_dataset(
            config, obs_keys, hdf5_path=tgt_path, demo_limit=tgt_demo_limit,
            obs_key_map=tgt_ds_cfg.get("obs_key_map", None),
            cache_mode_override=effective_cache_mode,
        )

        m_samples = len(source_dataset)
        n_samples = len(target_dataset)
        print("\n--- Source Dataset ({}, {} demos, {} samples) ---".format(
            src_key, src_demo_limit or "all", m_samples))
        print(source_dataset)
        print("\n--- Target Dataset ({}, {} demos, {} samples) ---".format(
            tgt_key, tgt_demo_limit or "all", n_samples))
        print(target_dataset)

        # MetaDataset combines both into a single index space:
        #   indices 0..m-1  => source
        #   indices m..m+n-1 => target
        ds_weights = [src_ds_cfg.get("weight", 0.5),
                      tgt_ds_cfg.get("weight", 0.5)]
        trainset = MetaDataset(
            datasets=[source_dataset, target_dataset],
            ds_weights=ds_weights,
            normalize_weights_by_ds_size=config.train.normalize_weights_by_ds_size,
        )
        trainset.ds_labels = [src_key, tgt_key]
        trainset.ds_labels_to_ids = {src_key: 0, tgt_key: 1}

        obs_normalization_stats = None
        if config.train.hdf5_normalize_obs:
            obs_normalization_stats = trainset.get_obs_normalization_stats()
        action_normalization_stats = trainset.get_action_normalization_stats()

        # For the q-update pass we iterate over the full dataset sequentially
        # (no weighted sampling, no dropping last incomplete batch).
        bea_q_update_loader = DataLoader(
            dataset=trainset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_data_workers,
            drop_last=False,
        )
        # For the h-update pass we use standard shuffled batches.
        train_sampler = trainset.get_dataset_sampler()
        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True,
        )

        print("\n--- Combined BEA Dataset ({} + {} = {} samples) ---".format(
            m_samples, n_samples, len(trainset)))
        print(trainset)
        print("")

        source_loader = None
        target_loader = None
        validset = None

    elif msbea_mode:
        # MSBEA: multi-source BEA with source-domain weights
        # All data entries except the last are sources; last is target.
        # Demo limits come from the config ("demo_limit" per entry).
        num_data = len(config.train.data)
        assert num_data >= 3, (
            f"MSBEA requires at least 2 source + 1 target data entries, "
            f"got {num_data}"
        )
        src_cfgs = list(config.train.data)[:-1]
        tgt_cfg = config.train.data[-1]

        print("\n--- MSBEA Co-training Setup ---")
        print("  mode   : msbea (Multi-Source BEA)")
        for i, sc in enumerate(src_cfgs):
            sc_path = os.path.expanduser(sc["path"])
            sc_key = sc.get("key", f"src{i}")
            sc_limit = sc.get("demo_limit", "all")
            print(f"  source {i}: key={sc_key}, demos={sc_limit}, path={sc_path}")
        tgt_path = os.path.expanduser(tgt_cfg["path"])
        tgt_key = tgt_cfg.get("key", "tgt")
        tgt_limit = tgt_cfg.get("demo_limit", "all")
        print(f"  target : key={tgt_key}, demos={tgt_limit}, path={tgt_path}")
        print("")

        effective_cache_mode = config.train.hdf5_cache_mode
        if effective_cache_mode == "all":
            print("[MSBEA] WARNING: hdf5_cache_mode='all' not supported with "
                  "MetaDataset. Falling back to 'low_dim'.")
            effective_cache_mode = "low_dim"

        source_datasets = []
        source_labels = []
        for i, sc in enumerate(src_cfgs):
            sc_path = os.path.expanduser(sc["path"])
            sc_limit = sc.get("demo_limit", None)
            ds = create_single_dataset(
                config, obs_keys, hdf5_path=sc_path,
                demo_limit=sc_limit if sc_limit is not None else None,
                obs_key_map=sc.get("obs_key_map", None),
                cache_mode_override=effective_cache_mode,
            )
            source_datasets.append(ds)
            source_labels.append(sc.get("key", f"src{i}"))
            print(f"\n--- Source {i} Dataset ({source_labels[-1]}, "
                  f"{sc_limit or 'all'} demos, {len(ds)} samples) ---")
            print(ds)

        target_dataset = create_single_dataset(
            config, obs_keys, hdf5_path=tgt_path,
            demo_limit=tgt_cfg.get("demo_limit", None),
            obs_key_map=tgt_cfg.get("obs_key_map", None),
            cache_mode_override=effective_cache_mode,
        )
        print(f"\n--- Target Dataset ({tgt_key}, "
              f"{tgt_limit} demos, {len(target_dataset)} samples) ---")
        print(target_dataset)

        all_datasets = source_datasets + [target_dataset]
        ds_weights = [c.get("weight", 1.0 / num_data) for c in config.train.data]
        trainset = MetaDataset(
            datasets=all_datasets,
            ds_weights=ds_weights,
            normalize_weights_by_ds_size=config.train.normalize_weights_by_ds_size,
        )
        trainset.ds_labels = source_labels + [tgt_key]
        trainset.ds_labels_to_ids = {
            lab: i for i, lab in enumerate(trainset.ds_labels)
        }

        msbea_source_sizes = [len(ds) for ds in source_datasets]
        n_samples = len(target_dataset)
        m_samples = sum(msbea_source_sizes)

        obs_normalization_stats = None
        if config.train.hdf5_normalize_obs:
            obs_normalization_stats = trainset.get_obs_normalization_stats()
        action_normalization_stats = trainset.get_action_normalization_stats()

        bea_q_update_loader = DataLoader(
            dataset=trainset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_data_workers,
            drop_last=False,
        )
        train_sampler = trainset.get_dataset_sampler()
        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True,
        )

        print(f"\n--- Combined MSBEA Dataset ({m_samples} + "
              f"{n_samples} = {len(trainset)} samples) ---")
        print(trainset)
        print("")

        source_loader = None
        target_loader = None
        validset = None

    elif vanilla_mode:
        # Co-training: separate source / target datasets
        src_ds_cfg = config.train.data[0]
        tgt_ds_cfg = config.train.data[1]
        src_path = os.path.expanduser(src_ds_cfg["path"])
        tgt_path = os.path.expanduser(tgt_ds_cfg["path"])
        src_demo_limit = src_ds_cfg.get("demo_limit", None)
        tgt_demo_limit = tgt_ds_cfg.get("demo_limit", None)
        src_key = src_ds_cfg.get("key", "src")
        tgt_key = tgt_ds_cfg.get("key", "tgt")

        print("\n--- Co-training Setup ---")
        print("  mode   : {}".format(args.cotraining))
        print("  alpha  : {}".format(args.alpha))
        print("  source : key={}, demos={}, path={}".format(
            src_key, src_demo_limit or "all", src_path))
        print("  target : key={}, demos={}, path={}".format(
            tgt_key, tgt_demo_limit or "all", tgt_path))
        print("")

        source_dataset = create_single_dataset(
            config, obs_keys, hdf5_path=src_path, demo_limit=src_demo_limit,
            obs_key_map=src_ds_cfg.get("obs_key_map", None),
        )
        target_dataset = create_single_dataset(
            config, obs_keys, hdf5_path=tgt_path, demo_limit=tgt_demo_limit,
            obs_key_map=tgt_ds_cfg.get("obs_key_map", None),
        )

        print("\n--- Source Dataset ({} / {} demos) ---".format(
            src_key, src_demo_limit or "all"))
        print(source_dataset)
        print("\n--- Target Dataset ({} / {} demos) ---".format(
            tgt_key, tgt_demo_limit or "all"))
        print(target_dataset)

        source_loader = DataLoader(
            dataset=source_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_data_workers,
            drop_last=True,
        )
        target_loader = DataLoader(
            dataset=target_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_data_workers,
            drop_last=True,
        )

        # action normalization: merge raw stats from both datasets then convert
        action_normalization_stats = _merge_action_norm_stats(
            source_dataset, target_dataset, config.train.action_config
        )
        # CRITICAL: propagate merged stats back so both datasets normalise
        # actions the same way during training (must match rollout denorm).
        source_dataset.set_action_normalization_stats(action_normalization_stats)
        target_dataset.set_action_normalization_stats(action_normalization_stats)

        obs_normalization_stats = None
        if config.train.hdf5_normalize_obs:
            obs_normalization_stats = source_dataset.get_obs_normalization_stats()

        train_loader = None  # not used
        validset = None

    else:
        # Standard training: single (possibly multi-) dataset
        trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=obs_keys)
        train_sampler = trainset.get_dataset_sampler()
        print("\n--- Training Dataset ---")
        print(trainset)
        print("")
        if validset is not None:
            print("\n--- Validation Dataset ---")
            print(validset)
            print("")

        obs_normalization_stats = None
        if config.train.hdf5_normalize_obs:
            obs_normalization_stats = trainset.get_obs_normalization_stats()
        action_normalization_stats = trainset.get_action_normalization_stats()

        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True,
        )

        source_loader = None
        target_loader = None

    valid_loader = None
    if validset is not None and config.experiment.validate:
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )

    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # Optim-params bookkeeping
    with config.values_unlocked():
        if "optim_params" in config.algo:
            if train_num_steps is not None:
                effective_ds_len = train_num_steps
            elif vanilla_mode:
                effective_ds_len = max(len(source_loader), len(target_loader))
            else:
                effective_ds_len = len(trainset)
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = effective_ds_len
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs

    # Model
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device,
    )

    # BEA / MSBEA weight initialization (must happen before checkpoint load)
    if bea_mode and not resume:
        model.init_bea_weights(m=m_samples, n=n_samples)
    elif msbea_mode and not resume:
        model.init_msbea_weights(source_sizes=msbea_source_sizes, n=n_samples)

    if resume:
        print("*" * 50)
        print("resuming from ckpt at {}".format(latest_model_path))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_path)
        except Exception as e:
            print("got error: {} when loading from {}".format(e, latest_model_path))
            print("trying backup path {}".format(latest_model_backup_path))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_backup_path)
        model.deserialize(ckpt_dict["model"], load_optimizers=True)
        print("*" * 50)

    ckpt_path = config.experiment.ckpt_path
    if (ckpt_path is not None) and (not resume):
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n--- Model Summary ---")
    print(model)
    print("")

    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience.")
    flush_warnings()
    print("*" * 50)
    print("")

    # Training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    start_epoch = 1
    if resume:
        variable_state = ckpt_dict["variable_state"]
        start_epoch = variable_state["epoch"] + 1
        best_valid_loss = variable_state["best_valid_loss"]
        best_return = variable_state["best_return"]
        best_success_rate = variable_state["best_success_rate"]
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)

    for epoch in range(start_epoch, config.train.num_epochs + 1):

        # Train one epoch
        if bea_mode:
            # Phase 1: update q with h fixed (every q.update_freq epochs)
            q_update_freq = config.algo.bea.q.update_freq
            if epoch % q_update_freq == 0 or epoch == start_epoch:
                d_mode = config.algo.bea.d.mode

                if d_mode == "knn":
                    d_stats = model.update_d_knn(
                        data_loader=bea_q_update_loader,
                        obs_normalization_stats=obs_normalization_stats,
                    )
                    for k, v in d_stats.items():
                        data_logger.record("BEA/{}".format(k), v, epoch)
                elif d_mode == "classifier":
                    d_stats = model.update_d_classifier(
                        data_loader=bea_q_update_loader,
                        obs_normalization_stats=obs_normalization_stats,
                    )
                    for k, v in d_stats.items():
                        data_logger.record("BEA/{}".format(k), v, epoch)

                q_stats = run_bea_q_update_pass(
                    model=model,
                    data_loader=bea_q_update_loader,
                    obs_normalization_stats=obs_normalization_stats,
                )
                # Log q-update stats
                for k, v in q_stats.items():
                    if isinstance(v, (int, float)):
                        data_logger.record("BEA/{}".format(k), v, epoch)
                print("[BEA] q-update ({}): mean={:.4f}, std={:.4f}, src={:.4f}, tgt={:.4f}, mass={:.2f}".format(
                    q_stats.get("q_update_mode", "?"),
                    q_stats["q_global_mean"], q_stats["q_global_std"],
                    q_stats["q_source_mean"], q_stats["q_target_mean"],
                    q_stats["q_total_mass"],
                ))

            # Phase 2: update h with q fixed
            step_log = run_bea_epoch(
                model=model,
                data_loader=train_loader,
                epoch=epoch,
                num_steps=train_num_steps,
                obs_normalization_stats=obs_normalization_stats,
            )
        elif msbea_mode:
            q_update_freq = config.algo.bea.q.update_freq
            if epoch % q_update_freq == 0 or epoch == start_epoch:
                d_mode = config.algo.bea.d.mode

                if d_mode == "knn":
                    d_stats = model.update_d_knn(
                        data_loader=bea_q_update_loader,
                        obs_normalization_stats=obs_normalization_stats,
                    )
                    for k, v in d_stats.items():
                        data_logger.record("MSBEA/{}".format(k), v, epoch)
                elif d_mode == "classifier":
                    d_stats = model.update_d_classifier(
                        data_loader=bea_q_update_loader,
                        obs_normalization_stats=obs_normalization_stats,
                    )
                    for k, v in d_stats.items():
                        data_logger.record("MSBEA/{}".format(k), v, epoch)

                q_stats = run_msbea_q_update_pass(
                    model=model,
                    data_loader=bea_q_update_loader,
                    obs_normalization_stats=obs_normalization_stats,
                )
                for k, v in q_stats.items():
                    if isinstance(v, (int, float)):
                        data_logger.record("MSBEA/{}".format(k), v, epoch)
                w_str = ", ".join(f"{v:.4f}" for v in model.w.tolist())
                # w gradient diagnostics
                w_grad_str = ""
                if "w_grad" in q_stats:
                    g = q_stats["w_grad"]
                    g_str = ", ".join(f"{v:.2f}" for v in g)
                    w_grad_str = f", grad_w=[{g_str}], eta_eff={q_stats['w_eta_eff']:.2e}"
                src_details = []
                for s in range(len(model.source_sizes)):
                    src_details.append(
                        f"  src{s}: w={q_stats[f'w_{s}']:.4f}, "
                        f"d={q_stats[f'd_src{s}_mean']:.2f}±{q_stats[f'd_src{s}_std']:.2f}, "
                        f"q_eff={q_stats[f'q_eff_src{s}_mean']:.4f}±{q_stats[f'q_eff_src{s}_std']:.4f} "
                        f"[{q_stats[f'q_eff_src{s}_min']:.4f}, {q_stats[f'q_eff_src{s}_max']:.4f}]"
                    )
                print(
                    f"[MSBEA] w=[{w_str}]{w_grad_str}\n"
                    f"  q_eff: src={q_stats['q_eff_source_mean']:.4f}±{q_stats['q_eff_source_std']:.4f}, "
                    f"tgt={q_stats['q_eff_target_mean']:.4f}±{q_stats['q_eff_target_std']:.4f}, "
                    f"mass={q_stats['q_eff_mass']:.2f}\n"
                    + "\n".join(src_details)
                )

            step_log = run_bea_epoch(
                model=model,
                data_loader=train_loader,
                epoch=epoch,
                num_steps=train_num_steps,
                obs_normalization_stats=obs_normalization_stats,
            )
        elif paired_mode:
            step_log = TrainUtils.run_epoch_for_ot_policy(
                model=model,
                ot_params=ot_params,
                bc_dataloader=train_loader,
                ot_dataloader=ot_loader,
                epoch=epoch,
                num_steps=train_num_steps,
                obs_normalization_stats=obs_normalization_stats,
            )
        elif vanilla_mode:
            step_log = run_cotrain_epoch(
                model=model,
                source_loader=source_loader,
                target_loader=target_loader,
                epoch=epoch,
                alpha=args.alpha,
                num_steps=train_num_steps,
                obs_normalization_stats=obs_normalization_stats,
            )
        else:
            step_log = TrainUtils.run_epoch(
                model=model,
                data_loader=train_loader,
                epoch=epoch,
                num_steps=train_num_steps,
                obs_normalization_stats=obs_normalization_stats,
            )
        model.on_epoch_end(epoch)

        # Checkpoint logic
        epoch_ckpt_name = "model_epoch_{}".format(epoch)
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (
                config.experiment.save.every_n_seconds is not None
            ) and (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (
                config.experiment.save.every_n_epochs is not None
            ) and (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))

        # Log current d stats every epoch (d only changes on q-update epochs)
        if bea_mode and config.algo.bea.d.mode != "d_hat" and model.m > 0:
            d_src = model.d[:model.m]
            d_info = {
                "d_source_mean": d_src.mean().item(),
                "d_source_std": d_src.std().item(),
                "d_source_min": d_src.min().item(),
                "d_source_max": d_src.max().item(),
            }
            for k, v in d_info.items():
                data_logger.record("BEA/{}".format(k), v, epoch)
            print("[BEA] d: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
                d_info["d_source_mean"], d_info["d_source_std"],
                d_info["d_source_min"], d_info["d_source_max"],
            ))

        # visualize OT transport plan if present
        pi = step_log.pop("pi", None)
        if pi is not None:
            plt.figure(figsize=(5, 5))
            plt.imshow(pi, cmap='viridis', aspect='auto')
            plt.colorbar(label="Value")
            plt.title("OT Transport Plan (epoch {})".format(epoch))
            pi_dir = os.path.join(video_dir, "pi")
            os.makedirs(pi_dir, exist_ok=True)
            fig_path = os.path.join(pi_dir, "pi_epoch_{}.png".format(epoch))
            plt.savefig(fig_path)
            plt.close()
            import imageio
            pi_img = imageio.imread(fig_path)
            data_logger.record("Train/pi", pi_img, epoch, data_type='image')

        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Validation (standard, OT, MMD, and BEA modes)
        if valid_loader is not None:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)
            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Rollouts
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (
            should_save_ckpt and ckpt_reason == "time"
        )
        if (
            config.experiment.rollout.enabled
            and (epoch > config.experiment.rollout.warmstart)
            and rollout_check
        ):
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )
            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record(
                            "Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]),
                            v, epoch,
                        )
                    else:
                        data_logger.record(
                            "Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True
                        )
                print(
                    "\nEpoch {} Rollouts took {}s (avg) with results:".format(
                        epoch, rollout_logs["time"]
                    )
                )
                print("Env: {}".format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            should_save_ckpt = (
                config.experiment.save.enabled and updated_stats["should_save_ckpt"]
            ) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

            if len(all_rollout_logs) == 1:
                env_name = list(all_rollout_logs.keys())[0]
                sr = all_rollout_logs[env_name]["Success_Rate"]
                epoch_ckpt_name = "model_epoch_{}_{}".format(epoch, sr)
            else:
                parts = ["model_epoch_{}".format(epoch)]
                for env_name in all_rollout_logs:
                    sr = all_rollout_logs[env_name]["Success_Rate"]
                    parts.append("{}_{}".format(env_name, sr))
                epoch_ckpt_name = "_".join(parts)

        # Save checkpoints
        variable_state = dict(
            epoch=epoch,
            best_valid_loss=best_valid_loss,
            best_return=best_return,
            best_success_rate=best_success_rate,
        )

        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        is_last_epoch = epoch == config.train.num_epochs
        should_save_latest = should_save_ckpt or is_last_epoch or (epoch % 10 == 0)
        if should_save_latest:
            print("\nsaving latest model at {}...\n".format(latest_model_path))
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
                variable_state=variable_state,
                ckpt_path=latest_model_path,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )
            shutil.copyfile(latest_model_path, latest_model_backup_path)
            print("\nsaved backup of latest model at {}\n".format(latest_model_backup_path))

        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    data_logger.close()


# Action-normalization helpers

def _merge_action_norm_stats(ds_a, ds_b, action_config):
    """
    Merge action normalization stats from two SequenceDatasets.

    We merge the *raw* action statistics (min, max, mean, sqdiff, n) from
    both datasets first, then convert to (scale, offset) via the standard
    robomimic helper so the format is exactly what downstream code expects.
    """
    from robomimic.utils.dataset import action_stats_to_normalization_stats

    # Gather raw stats from both datasets
    stats_a = ds_a.get_action_stats()
    stats_b = ds_b.get_action_stats()

    # Merge: take wider min/max, combine running stats
    merged = {}
    for key in stats_a:
        if key not in stats_b:
            raise KeyError("Action stats key '{}' missing from second dataset".format(key))
        sa = stats_a[key]
        sb = stats_b[key]
        n_a, n_b = sa["n"], sb["n"]
        n_tot = n_a + n_b
        if n_tot <= 0:
            raise ValueError("Invalid action stats for key '{}': total count <= 0".format(key))

        mean_a, mean_b = sa["mean"], sb["mean"]
        mean_tot = (mean_a * n_a + mean_b * n_b) / n_tot

        # Parallel-variance merge:
        # M2_total = M2_a + M2_b + n_a*(mu_a-mu)^2 + n_b*(mu_b-mu)^2
        sqdiff_tot = (
            sa["sqdiff"] + sb["sqdiff"]
            + n_a * np.square(mean_a - mean_tot)
            + n_b * np.square(mean_b - mean_tot)
        )
        merged[key] = {
            "min": np.minimum(sa["min"], sb["min"]),
            "max": np.maximum(sa["max"], sb["max"]),
            "mean": mean_tot,
            "sqdiff": sqdiff_tot,
            "n": n_tot,
        }

    return action_stats_to_normalization_stats(merged, action_config)


# CLI entry point

def main(args):
    """
    Build config from CLI arguments, apply overrides, and launch training.

    Args:
        args: parsed argparse namespace with config, algo, dataset, name,
              debug, resume, cotraining, alpha.
    """
    register_custom_task_envs(args.task)

    if args.config is not None:
        with open(args.config, "r") as f:
            ext_cfg = json.load(f)
        config = config_factory(ext_cfg["algo_name"])
        config.unlock()
        config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        with config.values_unlocked():
            config.train.data = [{"path": args.dataset}]

    if args.name is not None:
        with config.values_unlocked():
            config.experiment.name = args.name

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    if args.debug:
        config.unlock()
        config.lock_keys()
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10
        config.train.output_dir = "/tmp/tmp_trained_models"

    config.lock()

    try:
        train(config, device=device, args=args, resume=args.resume)
        print("Finished run successfully!")
    except Exception:
        print("Run failed with error:\n{}".format(traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robomimic training with optional co-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BEACON co-training, sim-to-sim (demo limits are set in the config JSON)
  python train.py --config configs/sim/beacon.json --task stack --cotraining bea

  # BEACON / MS-BEACON, sim-to-real
  python train.py --config configs/real/beacon.json   --task stack --cotraining bea
  python train.py --config configs/real/msbeacon.json --task stack --cotraining msbea

  # Baselines (the --cotraining mode must match the config's algo_name)
  python train.py --config configs/sim/target_only.json --task stack
  python train.py --config configs/sim/cotrain.json --task stack --cotraining vanilla --alpha 0.85
  python train.py --config configs/sim/mmd.json     --task stack --cotraining mmd
  python train.py --config configs/sim/uot.json     --task stack --cotraining ot
""",
    )

    # Task selection
    parser.add_argument(
        "--task", type=str, default="stack",
        choices=["stack", "mug", "thread", "luggage"],
        help="Task to train on: 'stack' (cube stacking), 'mug' (mug cleanup), "
             "'thread' (needle threading), or 'luggage' (push football over "
             "luggage ridge). Default: 'stack'.",
    )

    # Standard robomimic arguments
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a config JSON (overrides default algo settings).",
    )
    parser.add_argument(
        "--algo", type=str, default=None,
        help="Algorithm name (only needed when --config is not provided).",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Override experiment name in config.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override dataset path in config (standard mode only).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Quick training run for debugging.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from latest checkpoint.",
    )

    # Co-training arguments
    parser.add_argument(
        "--cotraining", type=str, default="off",
        choices=["off", "vanilla", "ot", "mmd", "bea", "msbea"],
        help="Co-training mode: 'off' (default), 'vanilla', 'ot' (OT-Sim2Real), "
             "'mmd' (MMD energy distance), 'bea' (Best-Effort Adaptation), "
             "or 'msbea' (Multi-Source BEA).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight for source domain loss.  L = α·L_source + (1-α)·L_target  (default: 0.5).",
    )

    args = parser.parse_args()
    main(args)
