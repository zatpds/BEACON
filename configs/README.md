# Configs

One example config per method, written for the block-stacking task. Every config
shares the same backbone, encoder, optimizer, and training schedule; they differ
only in `algo_name`, the method-specific `algo.<method>` block, and `train.data`.
To run another task or domain pair, copy a config and edit the dataset entries
as described below — the method hyperparameters are not tuned per task.

| Config | `algo_name` | `--cotraining` | Method | Setting |
|--------|-------------|----------------|--------|---------|
| `sim/beacon.json` | `diffusion_policy_bea` | `bea` | **BEACON** | sim → sim, `m=500`, `n=20` |
| `sim/cotrain.json` | `diffusion_policy` | `vanilla` | Co-Training (fixed ratio) | sim → sim |
| `sim/mmd.json` | `diffusion_policy_mmd` | `mmd` | MMD feature alignment | sim → sim |
| `sim/uot.json` | `diffusion_policy_ot` | `ot` | UOT feature alignment | sim → sim |
| `sim/target_only.json` | `diffusion_policy` | `off` | Target-Only (also Source-Only, see below) | single domain |
| `real/beacon.json` | `diffusion_policy_bea` | `bea` | **BEACON** | sim → real, `m=500`, `n=40` |
| `real/msbeacon.json` | `diffusion_policy_msbea` | `msbea` | **MS-BEACON** | two sim sources → real |

```bash
python train.py --config configs/sim/beacon.json --task stack --cotraining bea
```

`--cotraining` must match the config's `algo_name` (the algorithm class holds the
method-specific state), and `--task` must match the datasets (it decides which
environments are registered for rollouts). Co-Training additionally takes
`--alpha`, the source share of the loss `L = α·L_src + (1-α)·L_tgt`; we use
`--alpha 0.85`.

## Shared setup

Identical in every config, following the public Diffusion Policy implementation:

| | |
|---|---|
| Observations | one `96x96` RGB camera + proprioception (`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `robot0_gripper_qvel`, `robot0_joint_pos`, `robot0_joint_vel`) |
| Encoder | ResNet-18 + SpatialSoftmax (32 keypoints) → 64-d, random `84x84` crop |
| Policy | 1D conditional U-Net (`256/512/1024`), `T_o=2`, `T_a=8`, `T_p=16`, DDPM 100 steps, EMA 0.75 |
| Optimizer | AdamW, lr `1e-4`, cosine schedule with 500 warm-up steps, L2 `1e-6` |
| Schedule | batch 128, 100 gradient steps per epoch, 600 epochs (700 for sim → real, 300 for target-only) |

## BEACON hyperparameters (`algo.bea`)

| Config key | Paper symbol | sim → sim | sim → real | Meaning |
|------------|--------------|-----------|------------|---------|
| `lambda_1` | λ₁ | `0.01` | `0.01` | L1 deviation of `q` from the reference `p₀` |
| `lambda_2` | λ₂ | `0.01` | `0.01` | L2 regularizer on `q` |
| `d.lambda_d` | λ_d | `0.1` | `0.05` | weight of the discrepancy penalty `Σ qᵢ dᵢ` |
| `d.mode` / `d.k` | — / k | `knn`, `4` | `knn`, `5` | per-sample k-NN discrepancy in encoder space |
| `q.eta` | η_q | `0.01` | `0.01` | projected-subgradient step size for `q` |
| `q.update_freq` | K_q | `1` | `1` | refresh `d` and update `q` every epoch |
| `q.max` | q_max | `5.0` | `5.0` | per-sample upper bound |
| `q.target_min` | q_T^min | `0.05` | `0.10` | floor on target weights |
| `q.sum_constraint_alpha` | α | `0.5` | `0.4` | budget `Σ qᵢ = n + α·m` |
| `p0_target` | p₀ | `1/n` | `1/n` | reference weight on each target sample (`p0_source = 0`) |

The sim-to-real setting uses a smaller `lambda_d` and `alpha` because the larger
gap inflates the per-sample discrepancies. Alternative estimators and q-updates
(`d.mode ∈ {knn, classifier, d_hat}`, `q.mode ∈ {minibatch, full}`) are described
in [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md#implementation-variants).
Note that the `diffusion_policy_bea` config class defaults to `d_hat` + `full`,
so set both explicitly when writing a config from scratch.

MS-BEACON adds `algo.msbea`: `rho_1 = rho_2 = 0.01` (source-level regularizers),
`w.eta = 0.1` (step size for the per-source weights `w`), and `q.eta_T = 0.01`.

## Dataset entries (`train.data`)

```json
{
  "path": "data/stack/rsw_stack_20.hdf5",
  "key": "rsw",
  "weight": 0.15,
  "env_meta_update_dict": { "env_name": "StackWood", "env_kwargs": { "controller_configs": { ... } } }
}
```

- The **first** entry is the source domain, the **last** is the target. MS-BEACON
  takes several source entries followed by the target.
- `weight` is the sampling proportion of the dataset in a batch (`0.85 / 0.15`
  for sim → sim, `0.75 / 0.25` for sim → real); it is unrelated to BEACON's
  learned `q`. Co-Training ignores it and uses `--alpha`.
- `demo_limit` caps the number of demos read from the file.
- `eval: false` marks a dataset whose environment cannot be rolled out in
  simulation (real-robot data).
- `env_meta_update_dict` overrides the environment metadata stored in the HDF5
  before rollout environments are built. The `controller_configs` block pins the
  robosuite 1.5 OSC-pose controller the datasets were collected with; leave it
  as is. `env_name` selects the domain variant used for rollouts.
- `obs_key_map` renames observation keys at load time, so a source rendered from
  one camera can be co-trained with a target from another (see below).

Each dataset directory layout and the generation commands are in
[`../docs/DATA.md`](../docs/DATA.md).

## Adapting a config

**Another task.** Change the dataset paths and the rollout `env_name`:

| `--task` | Source dataset | Target dataset (texture gap) | Rollout `env_name` | `experiment.rollout.horizon` |
|----------|----------------|------------------------------|--------------------|------------------------------|
| `stack` | `data/stack/rs_stack_500.hdf5` | `data/stack/rsw_stack_20.hdf5` | `StackWood` | 250 |
| `mug` | `data/mug/rs_mug_500.hdf5` | `data/mug/rsw_mug_20.hdf5` | `MugCleanupWood` | 450 |
| `thread` | `data/thread/rs_thread_500.hdf5` | `data/thread/rsw_thread_20.hdf5` | `ThreadingWood` | 280 |
| `luggage` | `data/luggage/rs_luggage_500.hdf5` | `data/luggage/rsw_luggage_250.hdf5` | *(omit; same env, different style)* | 250 |

**Texture + viewpoint gap (`rsws`).** The target is rendered from `agentview45`,
so the policy consumes `agentview45_image` and the source is remapped onto it:

```json
"observation": { "modalities": { "obs": { "rgb": ["agentview45_image"], ... } } },
"train": { "data": [
  { "path": "data/stack/rs_stack_500.hdf5",   "key": "rs",   "weight": 0.85,
    "obs_key_map": { "agentview_image": "agentview45_image" }, ... },
  { "path": "data/stack/rsws_stack_20.hdf5", "key": "rsws", "weight": 0.15, ... }
] }
```

For UOT / MMD also point `train.ot.dataset.ot_tgt` and `train.ot.pair_info_path`
at the new target and its pair-info file.

**Source-Only.** Use `sim/target_only.json` with the source dataset as its single
entry (and 600 epochs).

**Target-scaling study.** Point the target entry of `sim/beacon.json` at
`rsw_stack_100.hdf5` and set `demo_limit` to 20 / 40 / 60 / 80.

**Smoke test.** `--debug` runs two short epochs and writes to `/tmp`; use it to
check the datasets, observation keys, and environment metadata before a long run:

```bash
python train.py --config configs/sim/beacon.json --task stack --cotraining bea --debug
```
