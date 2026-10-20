# Vendored packages and acknowledgments

BEACON builds on three open-source robot-learning packages, which are vendored
here (under `packages/`) so the repository is self-contained and reproducible.
Each package is redistributed under its own license, included in its directory.
All credit for these packages goes to their original authors; we vendor lightly
modified copies only to add the environments and algorithms used in this work.

Our modifications on top of each upstream package are limited to the files
listed below (everything else is unmodified upstream code).

## robosuite (v1.5.2) — MIT License
Upstream: https://github.com/ARISE-Initiative/robosuite
- `robosuite/environments/manipulation/stack_wood.py` — StackWood domain-shift variant (added)
- `robosuite/environments/manipulation/single_arm_env.py` — table/texture style hooks
- `robosuite/models/assets/arenas/table_arena.xml` — wood-table arena option
- `robosuite/models/robots/manipulators/panda_robot.py`, `robosuite/__init__.py`,
  `robosuite/controllers/config/robots/default_panda.json` — minor registration tweaks

## robomimic (v0.5.0) — MIT License
Upstream: https://github.com/ARISE-Initiative/robomimic
- `robomimic/algo/diffusion_policy_bea.py` — BEACON co-training (added)
- `robomimic/algo/diffusion_policy_msbea.py` — MS-BEACON, the multi-source extension (added)
- `robomimic/algo/diffusion_policy_ot.py` — Unbalanced Optimal Transport co-training (added)
- `robomimic/algo/diffusion_policy_mmd.py` — MMD (energy-distance) co-training (added)
- `robomimic/config/diffusion_policy_{bea,msbea,ot,mmd}_config.py` — matching config classes
- `robomimic/utils/domain_classifier.py` — classifier-based discrepancy estimate (added)
- `robomimic/utils/action_utils.py`, `robomimic/utils/ot_dataset.py`,
  `robomimic/utils/train_utils.py` — co-training data/loss plumbing
- `robomimic/algo/__init__.py`, `robomimic/config/__init__.py` — register the new algorithms

The diffusion-policy backbone (`robomimic/algo/diffusion_policy.py`) follows
Diffusion Policy (Chi et al., RSS 2023, https://github.com/real-stanford/diffusion_policy).

## mimicgen (v1.0.1) — NVIDIA License
Upstream: https://github.com/NVlabs/mimicgen
Note: mimicgen is distributed under the NVIDIA License (see `mimicgen/LICENSE`),
which differs from the MIT license of the other two packages; review it before
redistribution or commercial use.
- `mimicgen/envs/robosuite/stack.py`, `mimicgen/envs/robosuite/mug_cleanup.py`,
  `mimicgen/envs/robosuite/threading.py`, `mimicgen/envs/robosuite/single_arm_env_mg.py`
  — domain-shift (wood-table / camera) variants of the data-generation environments
- `mimicgen/scripts/generate_dataset.py`, `mimicgen/scripts/prepare_src_dataset.py`
  — minor hooks for the variants above
- `mimicgen/scripts/generate_core_configs.py` — default paths pointed at this
  repository's layout (the generated configs are checked in under
  `scripts/rs_collect/core_configs/`)
