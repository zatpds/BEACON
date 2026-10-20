# BEACON: Cross-Domain Co-Training of Generative Robot Policies via Best-Effort Adaptation

### [Project Page](https://computationalrobotics.seas.harvard.edu/BEACON/) · [Paper](https://arxiv.org/abs/2605.08571)

BEACON trains image-based [Diffusion Policies](https://github.com/real-stanford/diffusion_policy)
that transfer across a domain gap — between simulation variants and from
simulation to a real robot — by co-training on abundant source demonstrations
together with a small target dataset.

It casts cross-domain co-training as a **discrepancy-aware importance
reweighting** problem: the policy and per-sample source weights `q` are learned
jointly, minimizing a surrogate informed by a target-domain generalization bound,
so that source samples compatible with the target are up-weighted and mismatched
ones are suppressed. **MS-BEACON** extends this to multiple source domains with a
second, domain-level weight.

## Methods

All methods share the same backbone, encoder, and datasets, and differ only in
how source data is incorporated:

| Method | `--cotraining` | Description |
|--------|----------------|-------------|
| Source-Only / Target-Only | `off` | Behavior cloning on one dataset |
| Co-Training | `vanilla` | Fixed mixture `L = α·L_src + (1-α)·L_tgt` |
| UOT | `ot` | Unbalanced optimal transport feature alignment |
| MMD | `mmd` | Energy-distance MMD feature alignment |
| **BEACON** | `bea` | Discrepancy-aware `q`-weighted ERM |
| **MS-BEACON** | `msbea` | Multi-source BEACON |

`bea` / `msbea` are the code's internal names for BEACON and MS-BEACON.

## Tasks and domains

| `--task` | Environments | Task |
|----------|--------------|------|
| `stack` | `Stack` / `StackWood` | block stacking |
| `mug` | `MugCleanup` / `MugCleanupWood` | mug cleanup (open drawer, place mug, close drawer) |
| `thread` | `Threading` / `ThreadingWood` | threading |
| `luggage` | `LuggagePush` | ball rollover (additional task, not part of the arXiv preprint) |

Domains, used as filename and config suffixes: **`rs`** source, canonical
robosuite (`agentview`) · **`rsw`** Target-T, texture gap (`agentview`) ·
**`rsws`** Target-T+V, texture and viewpoint gap (`agentview45`) · **`real`**
real robot, visual *and* physical gap. Source datasets have 500 demos per task;
targets have 20 (sim → sim) or 40 (sim → real).

## Installation

Linux with an NVIDIA GPU. CPU works but is impractically slow for image-based
diffusion policies.

```bash
bash setup.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate beacon
export MUJOCO_GL=osmesa   # or 'egl' (headless GPU) / 'glfw'
```

`setup.sh` creates a conda environment named `beacon`, installs PyTorch and the
dependencies, and installs the vendored simulators from `packages/` in editable
mode. `osmesa` is the safe default — `egl` renders faster but on some drivers
fails to resize its framebuffer for rollout videos.

## Data

No datasets ship with this repository. Simulation demonstrations are generated
with [MimicGen](https://github.com/NVlabs/mimicgen) from its public source demos
using the collection configs in `scripts/rs_collect/core_configs/`; the ball
rollover demos come from a scripted policy in `scripts/rs_collect/luggage_env/`;
real demos are converted from teleoperation recordings. Configs expect
robomimic-style HDF5 files under `data/`:

```
data/stack/rs_stack_500.hdf5      # source
data/stack/rsw_stack_20.hdf5      # target, texture gap
data/stack/pair_info_rsw.json     # DTW pair info, UOT / MMD only
data/stack_R/real_stack_40.hdf5   # real-robot target
```

**[`docs/DATA.md`](docs/DATA.md)** covers the full pipeline and how to inspect a
dataset before training.

## Training

```bash
# BEACON, sim-to-sim (texture gap)
python train.py --config configs/sim/beacon.json --task stack --cotraining bea

# BEACON / MS-BEACON, sim-to-real
python train.py --config configs/real/beacon.json   --task stack --cotraining bea
python train.py --config configs/real/msbeacon.json --task stack --cotraining msbea

# baselines
python train.py --config configs/sim/target_only.json --task stack
python train.py --config configs/sim/cotrain.json --task stack --cotraining vanilla --alpha 0.85
python train.py --config configs/sim/mmd.json     --task stack --cotraining mmd
python train.py --config configs/sim/uot.json     --task stack --cotraining ot
```

`--cotraining` must match the config's `algo_name`. Add `--debug` for a one-minute
smoke run, `--resume` to continue, `--name` to override the run name. Checkpoints,
logs, and videos go to the config's `output_dir` under `runs/`.

`configs/` holds one example config per method, written for block stacking; the
method hyperparameters are not tuned per task, so other tasks and domain pairs
only need the dataset entries changed. **[`configs/README.md`](configs/README.md)**
documents every knob and the per-task edits; **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)**
maps the paper's experiments to commands, including the evaluation protocol and
the alternative discrepancy estimators and `q`-updates.

## Evaluation

```bash
python eval.py --agent runs/beacon_stack_sim/<run>/models/model_epoch_600.pth \
    --task stack --eval_env rsw --num_seeds 100 --start_seed 1000 --save_videos
```

`--eval_env` (`rs` / `rsw` / `rsws`) selects the rollout environment independently
of what the policy was trained on, which is how the transfer numbers are produced.
`--strict` applies the sustained-success criterion used for block stacking;
`--ddim_steps N` gives faster inference. Policies trained on real data were
evaluated on the physical robot; that hardware-specific control loop is not part
of this repository.

## License

BEACON's own code is MIT licensed (see [`LICENSE`](LICENSE)). The vendored
packages keep their own licenses — robomimic and robosuite are MIT, while
**mimicgen is under the more restrictive NVIDIA License**; review it before
redistribution or commercial use.

## Acknowledgments

BEACON builds on [robomimic](https://github.com/ARISE-Initiative/robomimic),
[robosuite](https://github.com/ARISE-Initiative/robosuite), and
[mimicgen](https://github.com/NVlabs/mimicgen), with the diffusion-policy
backbone following [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
(Chi et al., RSS 2023), and on the Best-Effort Adaptation framework of Awasthi,
Cortes, and Mohri (2023). See
[`packages/ACKNOWLEDGMENTS.md`](packages/ACKNOWLEDGMENTS.md) for the files we
added or modified in each package.

## Citation

```bibtex
@misc{beacon2026,
  title  = {BEACON: Cross-Domain Co-Training of Generative Robot Policies
            via Best-Effort Adaptation},
  author = {Zhang, Antong and Qi, Han and Yang, Heng},
  year   = {2026},
  eprint = {2605.08571},
  archivePrefix = {arXiv},
  note   = {\url{https://computationalrobotics.seas.harvard.edu/BEACON/}}
}
```
