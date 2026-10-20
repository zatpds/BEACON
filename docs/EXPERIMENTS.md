# Reproducing the experiments

This page maps the experiments in the paper to configs and commands. Dataset
preparation is covered in [`DATA.md`](DATA.md); the configs and every knob in
them are described in [`../configs/README.md`](../configs/README.md).

## How a run is specified

Three things define a run: the config (datasets, method hyperparameters,
architecture), the task (which environments are registered for rollouts), and the
co-training mode:

```bash
python train.py --config <config.json> --task <stack|mug|thread|luggage> \
    --cotraining <off|vanilla|ot|mmd|bea|msbea>
```

| Method in the paper | Config | `--cotraining` |
|---------------------|--------|----------------|
| Source-Only / Target-Only | `configs/sim/target_only.json` | `off` |
| Co-Training | `configs/sim/cotrain.json` | `vanilla --alpha 0.85` |
| MMD | `configs/sim/mmd.json` | `mmd` |
| UOT | `configs/sim/uot.json` | `ot` |
| **BEACON** | `configs/sim/beacon.json`, `configs/real/beacon.json` | `bea` |
| **MS-BEACON** | `configs/real/msbeacon.json` | `msbea` |

`bea` / `msbea` are the implementation's internal names for BEACON and MS-BEACON;
*BEA* on its own refers to the Best-Effort Adaptation framework the method builds
on. The configs are written for block stacking; the per-task edits (dataset paths,
rollout environment, horizon) are listed in `configs/README.md`.

Always start with a smoke run — it exercises data loading, observation keys, and
environment metadata in about a minute and writes to `/tmp`:

```bash
python train.py --config configs/sim/beacon.json --task stack --cotraining bea --debug
```

## Tasks and domains

| Task | `--task` | Environments | Success |
|------|----------|--------------|---------|
| Block stacking | `stack` | `Stack` / `StackWood` | cube A on cube B (`R`: at any frame; `S`: sustained, `eval.py --strict`) |
| Mug cleanup | `mug` | `MugCleanup` / `MugCleanupWood` | mug upright inside the drawer, drawer closed |
| Threading | `thread` | `Threading` / `ThreadingWood` | rod inserted through the receptacle |

| Domain | Alias | Gap | Camera |
|--------|-------|-----|--------|
| Source | `rs` | canonical robosuite scene | `agentview` |
| Target-T | `rsw` | textures (wood table) | `agentview` |
| Target-T+V | `rsws` | textures + camera extrinsics (`agentview` rotated 45° about the table center) | `agentview45` |
| Target-P | `real` | physical + visual gap, Franka Panda | fixed RGB camera |

Source datasets have `m = 500` MimicGen demonstrations per task. Target datasets
have `n = 20` demonstrations for sim → sim and `n = 40` for sim → real.

## Sim-to-sim

For task `<t>` in `stack`, `mug`, `thread`, with the texture-gap target (`_w`
configs edited as described in `configs/README.md` for other tasks and for the
T+V target):

```bash
# single-domain baselines (Target-Only as shipped; Source-Only by swapping the dataset)
python train.py --config configs/sim/target_only.json --task <t>

# co-training methods — same source and target datasets, same encoder and backbone
python train.py --config configs/sim/cotrain.json --task <t> --cotraining vanilla --alpha 0.85
python train.py --config configs/sim/uot.json     --task <t> --cotraining ot
python train.py --config configs/sim/mmd.json     --task <t> --cotraining mmd
python train.py --config configs/sim/beacon.json  --task <t> --cotraining bea
```

UOT and MMD need the DTW pair-info file for the source/target pair
(`DATA.md` §4.1). All co-training runs must share the same datasets, observation
encoder, and diffusion backbone, differing only in how source data is used — keep
it that way when you add configs, since that is what makes the comparison
meaningful.

**Target scaling** (block stacking): rerun `configs/sim/beacon.json` with the
target entry pointed at `rsw_stack_100.hdf5` (or `rsws_stack_100.hdf5`) and
`demo_limit` set to 20 / 40 / 60 / 80.

## Sim-to-real

Sim source demos are co-trained with `n = 40` real demonstrations. MS-BEACON adds
the texture-gap simulation domain as a second source; the two sources contribute
equal halves of the source data (`250 + 250`):

```bash
python train.py --config configs/real/beacon.json   --task stack --cotraining bea
python train.py --config configs/real/msbeacon.json --task stack --cotraining msbea
```

Baselines for the real target are the `configs/sim/*` configs with the target
entry replaced by `data/stack_R/real_stack_40.hdf5` (with `"eval": false`, since
it cannot be rolled out in simulation) and training rollouts disabled; see
`configs/real/beacon.json` for the pattern.

Policies trained on real data were evaluated on the robot, from 20 initial states
shared across methods. The out-of-distribution evaluation (block stacking and mug
cleanup) trains on real demos whose object placements are confined to designated
regions and evaluates from placements outside them; it differs from the standard
run only in the target dataset. The robot-side controller and evaluation loop are
not part of this repository, so `eval.py` applies to the simulated domains only.

## Evaluation

```bash
python eval.py --agent runs/beacon_stack_sim/<run>/models/model_epoch_600.pth \
    --task stack --eval_env rsw --num_seeds 100 --start_seed 1000 --save_videos

# S (sustained) criterion for block stacking
python eval.py --agent <ckpt.pth> --task stack --eval_env rsw --num_seeds 100 --start_seed 1000 --strict
```

- `--eval_env` (`rs` / `rsw` / `rsws`) selects the environment the policy is
  rolled out in, independently of what it was trained on; this is how the
  transfer numbers are produced.
- Sim-to-sim results use **100 initial states**, identical across the methods
  being compared: keep `--start_seed` and `--num_seeds` fixed, and choose
  `--start_seed` beyond the seeds used by training-time rollouts so the
  evaluation states are held out.
- The metric is the success rate (`summary.json` / `avg_success_rate.npy`).
  Block stacking is reported under both the `R` criterion (success at any frame)
  and the `S` criterion (`--strict`: five consecutive success frames, or success
  on the last frame).
- `--ddim_steps 10` gives roughly 10x faster inference for DDPM-trained
  checkpoints. Results go to `<run_dir>/eval_results/` by default.

## Implementation variants

The discrepancy estimator and the q-update rule are both selectable in
`algo.bea`, covering the variants discussed in the paper's appendix:

| Config field | Values | Meaning |
|--------------|--------|---------|
| `d.mode` | `knn` | per-sample k-NN distance in encoder embedding space, normalized by the within-target k-NN spread (**default**) |
| | `classifier` | per-sample `1 − g(φ(x))` from a class-balanced logistic-regression domain classifier on encoder features |
| | `d_hat` | a single localized labeled discrepancy `d̂` from a target-only reference policy, fixed for the whole run (estimate it with `utils/estimate_dhat.py`, set `d.value`) |
| `q.mode` | `minibatch` | stochastic projected-subgradient update on `q`, followed by projection onto the box and sum constraint (**default**) |
| | `full` | full convex inner solve for `q` at fixed policy (soft-threshold closed form + bisection on the sum constraint) |

Both defaults (`knn` + `minibatch`) are set in the shipped configs. The
`diffusion_policy_bea` config class itself defaults to `d_hat` + `full`, so set
`d.mode` and `q.mode` explicitly in any config written from scratch.

## Outputs

Each run creates `<train.output_dir>/<experiment.name>/<timestamp>/`, containing
`models/` (periodic checkpoints plus `last.pth`), `logs/` (TensorBoard events and
a text log), `videos/` (rollout renders), and a copy of the resolved
`config.json`. `--resume` continues from the latest checkpoint of a run with the
same name, and `--name` overrides the name from the config.

## Analysis

| Script | Produces |
|--------|----------|
| `scripts/plot_q_evolution.py --log <run>/logs/log.txt` | evolution of BEACON's source / target weights `q` over training, and their per-sample distribution from a checkpoint |
| `scripts/feature_align/plot_feature_align.py --manifest <manifest.json>` | UMAP of encoder features, source vs. target, for a set of checkpoints (feature-alignment analysis; see `scripts/feature_align/example_manifest.json`) |
| `utils/estimate_dhat.py` | localized discrepancy estimate `d̂` from a target-only checkpoint |
