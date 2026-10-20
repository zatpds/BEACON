# Data collection and preprocessing

No datasets ship with this repository. Everything under `data/` is produced by
the steps below: simulation demonstrations are generated with
[MimicGen](https://github.com/NVlabs/mimicgen), and real-robot demonstrations are
converted from teleoperation recordings. All commands are run from the repository
root with the `beacon` environment active.

## Naming conventions

Training configs refer to datasets by a short domain alias:

| Prefix | Domain | Camera |
|--------|--------|--------|
| `rs_` | canonical robosuite (ceramic table) | `agentview` |
| `rsw_` | wood-table variant | `agentview` |
| `rsws_` | wood table **and** shifted viewpoint | `agentview45` |
| `real_` | real-robot demonstrations | `agentview` |

Files live in one directory per task, with a `_R` suffix for the directories used
by the sim-to-real experiments:

```
data/
├── stack/      stack_R/       # Stack / StackWood
├── mug/        mug_R/         # MugCleanup / MugCleanupWood
├── thread/                    # Threading / ThreadingWood
└── luggage/    luggage_R/     # LuggagePush (ball rollover)
```

The `_R` directories keep their own copy of the source dataset (e.g. both
`data/stack/rs_stack_500.hdf5` and `data/stack_R/rs_stack_500.hdf5`); a single
generation can be copied or symlinked to both locations.

Every dataset is a standard robomimic HDF5 file — `data/demo_<i>/{actions,
states, rewards, dones, obs/*, next_obs/*}`, with `data.attrs["env_args"]`
describing the environment — with `96x96` uint8 images.

## What each experiment needs

| Experiment | Datasets |
|------------|----------|
| Source-Only | `data/<task>/rs_<task>_500.hdf5` |
| Target-Only | `data/<task>/rsw_<task>_20.hdf5` or `rsws_<task>_20.hdf5` (luggage: `rsw_luggage_250`) |
| Sim → sim co-training (`configs/sim/`) | source `rs_<task>_500` + target `rsw_` / `rsws_` variant (+ `pair_info_*.json` for UOT / MMD) |
| Sim → real co-training (`configs/real/`) | `data/<task>_R/rs_<task>_500` + `real_<task>_40` (+ `rsw_<task>_250` for MS-BEACON) |
| Target scaling | `rsw_<task>_100.hdf5` / `rsws_<task>_100.hdf5`, capped with `demo_limit` |

To check what a specific config needs, read its `train.data[*].path` entries.

## 1. Simulation datasets

### 1.1 Source demonstrations

MimicGen generates new demonstrations by transforming a handful of human source
demos. Download them and annotate them with the object-centric subtask
signals MimicGen needs:

```bash
python -m mimicgen.scripts.download_datasets \
    --dataset_type source --tasks stack mug_cleanup threading \
    --download_dir /tmp/mg_source          # -> /tmp/mg_source/source/<task>.hdf5

python -m mimicgen.scripts.prepare_src_dataset \
    --dataset /tmp/mg_source/source/stack.hdf5 \
    --env_interface MG_Stack --env_interface_type robosuite \
    --output data/stack/stack.hdf5
```

Repeat for the other two tasks (`MG_MugCleanup` → `data/mug/mug_cleanup.hdf5`,
`MG_Threading` → `data/thread/threading.hdf5`); these output paths are what the
collection configs in `scripts/rs_collect/core_configs/` expect.

### 1.2 Generate the domain variants

Each collection config pairs a source dataset with a target environment and
camera. `--num_demos` and `--seed` override the values in the config, so the
same config produces the different dataset sizes:

```bash
python -m mimicgen.scripts.generate_dataset \
    --config scripts/rs_collect/core_configs/stack_wood.json \
    --num_demos 20 --seed 1
```

| Collection config | Environment | Camera | Demos | Produces |
|-------------------|-------------|--------|-------|----------|
| `stack.json` | `Stack` | `agentview` + `robot0_eye_in_hand` | 500 | `rs_stack_500.hdf5` |
| `stack_wood.json` | `StackWood` | `agentview` | 250 | `rsw_stack_250.hdf5` (`--num_demos 100 / 20` for the smaller variants) |
| `stack_wood_s.json` | `StackWood` | `agentview45` | 100 | `rsws_stack_100.hdf5` (`--num_demos 20` for `rsws_stack_20`) |
| `mug.json` | `MugCleanup` | `agentview` | 500 | `rs_mug_500.hdf5` |
| `mug_wood.json` | `MugCleanupWood` | `agentview` | 250 | `rsw_mug_250.hdf5` (`--num_demos 20` for `rsw_mug_20`) |
| `mug_wood_s.json` | `MugCleanupWood` | `agentview45` | 20 | `rsws_mug_20.hdf5` |
| `threading.json` | `Threading` | `agentview` | 500 | `rs_thread_500.hdf5` |
| `threading_wood.json` | `ThreadingWood` | `agentview` | 20 | `rsw_thread_20.hdf5` |
| `threading_wood_s.json` | `ThreadingWood` | `agentview45` | 20 | `rsws_thread_20.hdf5` |

None of the shipped configs read the wrist camera; drop `robot0_eye_in_hand`
from `stack.json` unless you want it, since it roughly doubles the dataset size.

Generation writes to `scripts/rs_collect/core_datasets/<experiment>/`, together
with a log and preview videos. The dataset itself is `demo.hdf5`; move and rename
it to the `data/` location above (target demos are not saved with a size in the
filename, so name them according to how many demos you generated).

Generation is the expensive step — a few hours per 500-demo dataset on one
machine — and it is CPU-bound, so run the tasks in parallel if you can.

### 1.3 Stack only: append lift actions

MimicGen's `Stack` demos terminate as soon as cube A is placed on cube B, with
the gripper still around the cube. Policies trained on them learn to stop rather
than retract, which makes rollout success ambiguous. `postprocess_release_lift.py`
replays the tail of each demo to warm up the OSC controller and appends a ramped
lift:

```bash
python scripts/postprocess_release_lift.py \
    --input  scripts/rs_collect/core_datasets/stack_real/stack/demo.hdf5 \
    --output data/stack/rs_stack_500.hdf5 \
    --lift_steps 20 --z_max 0.5
```

The output has actions and states but **no** observations, so re-render them
(§1.4) before training.

### 1.4 (Re-)extracting observations

Use robomimic's replay script when you need cameras that were not rendered
during generation, or after modifying actions/states:

```bash
python -m robomimic.scripts.dataset_states_to_obs \
    --dataset data/stack/rs_stack_500.hdf5 \
    --output_name rs_stack_500_obs.hdf5 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 96 --camera_width 96 --done_mode 2
```

The observation keys must match the `observation.modalities.obs` block of the
config you intend to train with. When a source dataset was rendered with a
different camera than its target (for example `agentview` source vs
`agentview45` target), you do not need to re-render: the config's per-dataset
`obs_key_map` renames source keys at load time, e.g.

```json
{ "path": "data/mug/rs_mug_500.hdf5", "key": "rs",
  "obs_key_map": { "agentview_image": "agentview45_image" } }
```

## 2. Ball rollover task (`luggage`)

`LuggagePush` — the *ball rollover* task — is a custom robosuite environment
(`scripts/rs_collect/luggage_env/`) rather than a MimicGen task, so its demos come
from a scripted state-machine policy instead of demo generation. The gripper has
to get a ball over the central ridge of a closed luggage; the ridge is nearly as
tall as the ball, so the ball must be rolled to the ridge and levered over.

The domain shift here is a change of appearance within the same environment,
selected with `--table_style` / `--luggage_style`:

```bash
cd scripts/rs_collect/luggage_env

# source (rs): ceramic table
MUJOCO_GL=egl python collect_demo.py --n_demos 500 \
    --out ../../../data/luggage/rs_luggage_500.hdf5

# target (rsw): wood table
MUJOCO_GL=egl python collect_demo.py --n_demos 250 --table_style wood \
    --out ../../../data/luggage/rsw_luggage_250.hdf5
```

Two fix-ups are needed before these datasets train efficiently:

```bash
# Images were stored bottom-up by the raw robosuite env; match robomimic's convention.
python utils/fix_sim_image_orientation.py \
    data/luggage/rs_luggage_500.hdf5 data/luggage/rsw_luggage_250.hdf5

# Rewrite gzip/auto-chunked files as contiguous, uncompressed datasets (~13x faster loading).
python utils/repack_hdf5_contiguous.py data/luggage/rs_luggage_500.hdf5
```

Both are idempotent: the orientation fix records a flag in `data.attrs` so a file
is never flipped twice.

## 3. Real-robot datasets

Real demonstrations were teleoperated on a Franka Panda and recorded in the
zarr replay-buffer layout used by the diffusion-policy real-robot stack (that
robot-side control code is not part of this repository). The expected layout is
documented at the top of `utils/zarr_to_hdf5.py`:

```
meta/episode_ends, data/{action, robot_eef_pose, robot_joint, robot_joint_vel, camera_0}
```

Convert to robomimic HDF5:

```bash
python utils/zarr_to_hdf5.py \
    --zarr data/stack_R/stack_R.zarr \
    --out  data/stack_R/real_stack_40.hdf5 \
    --env-name Stack
```

The converter drops idle timesteps (teleoperation contains long pauses that
otherwise dominate the behaviour-cloning loss), center-crops and resizes frames
to `96x96`, derives `robot0_eef_quat` / gripper qpos from the recorded pose and
gripper command, and writes a `mask/train` filter key. `object` and `states` are
zero-filled — the image-based policies do not read them, but keep this in mind if
you add a low-dim baseline.

The real dataset for the out-of-distribution experiment is produced the same way
from a separate teleoperation session in which the objects are placed in the
designated target region only; the OOD *evaluation* initial states are then drawn
from regions disjoint from it. Nothing about the conversion differs — only which
placements were demonstrated.

If the camera framing needs adjusting before the resize, crop first:

```bash
python scripts/crop_obs.py --src data/stack_R/real_stack_40.hdf5 \
    --left 0.10 --right 0.10 --top 0.0 --bottom 0.20
```

`crop_obs.py` also shifts the stored camera intrinsics' principal point so they
stay consistent with the cropped frames.

## 4. Derived files for co-training

### 4.1 DTW pair info (UOT and MMD)

The UOT and MMD configs pair every target demo with the source demos closest to
it under dynamic time warping, and read that pairing from a JSON file named by
`train.ot.pair_info_path` (a single top-level field, not a per-dataset one):

```bash
python utils/gen_dtw.py \
    --src data/stack/rs_stack_500.hdf5 \
    --tgt data/stack/rsw_stack_20.hdf5 \
    --output data/stack/pair_info_rsw.json
```

The alignment uses low-dimensional proprioception (`robot0_eef_pos`,
`robot0_gripper_qpos`), so it is unaffected by the camera difference between
source and target. The script also writes a histogram of the distance
distribution next to the JSON and prints suggested cutoff percentiles.

One pair-info file is needed per (source, target) pair you run UOT or MMD on,
e.g. `data/stack/pair_info_rsw.json` (`rs_stack_500` ↔ `rsw_stack_20`),
`data/stack/pair_info_rsws.json` (`rs_stack_500` ↔ `rsws_stack_20`), or
`data/stack_R/pair_info_rsR.json` (`rs_stack_500` ↔ `real_stack_40`). The
`cutoff` in the UOT config (`algo.ot.cutoff`) is a DTW-distance threshold on
this pairing; take it from the percentiles the script prints when you generate
pair info for a new dataset pair.

BEACON, MS-BEACON, Co-Training, and single-domain training do not need pair info.

### 4.2 Discrepancy estimate (analysis only)

`utils/estimate_dhat.py` estimates the localized source–target discrepancy d̂
from a target-only reference checkpoint. It is reported in the analysis and is
not required for training:

```bash
python utils/estimate_dhat.py \
    --ckpt_path runs/target_only_stack_sim/<run>/models/model_epoch_300.pth \
    --source_data data/stack/rs_stack_500.hdf5 \
    --target_data data/stack/rsws_stack_20.hdf5 \
    --source_obs_key_map '{"agentview_image":"agentview45_image"}' \
    --name stack_rsws --rho_frac 0.05
```

## 5. Checking a dataset before training

```bash
# Structure, demo count/lengths, obs keys and shapes, action ranges, env metadata
python scripts/inspect_hdf5.py data/stack/rsw_stack_20.hdf5

# Sample frames and short clips, to confirm the domain variant looks right
python scripts/get_obs.py --path data/stack/rsws_stack_20.hdf5 \
    --camera_key agentview45_image --num_samples 5 --tag stack_rsws
```

A quick way to confirm a config and its datasets agree, before launching a long
run, is a two-epoch smoke run:

```bash
python train.py --config configs/sim/beacon.json --task stack --cotraining bea --debug
```

`--debug` shortens epochs and rollouts and writes to `/tmp`, so it exercises data
loading, the observation keys, and the environment metadata without producing a
real checkpoint.

## Checklist

1. Source demos prepared: `data/<task>/<task>.hdf5`.
2. Domain variants generated and renamed into `data/<task>/` (and `data/<task>_R/`).
3. Stack datasets passed through the lift post-processing and re-rendered.
4. Ball-rollover (`luggage`) datasets orientation-fixed and repacked.
5. Real datasets converted from zarr and cropped if needed.
6. `pair_info_*.json` generated for every UOT/MMD config you plan to run.
7. `--debug` smoke run passes for one config per experiment family.
