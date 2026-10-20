"""
Post-process a robomimic-style HDF5 dataset to append lift-up actions
at the end of each demonstration.

This fixes the issue where MimicGen-generated Stack demos terminate
immediately after placing cubeA on cubeB without lifting the arm away.

The script restores each demo a few steps before the end and replays
those actions to warm up the OSC controller (whose internal target is
not captured by sim state), then appends lift actions.

The output does NOT contain image observations -- run
  python -m robomimic.scripts.dataset_states_to_obs
afterwards to generate them, just like with the original MimicGen output.

Usage:
    python scripts/postprocess_release_lift.py \
        --input  scripts/rs_collect/core_datasets/stack_real/stack_wood/demo.hdf5 \
        --output scripts/rs_collect/core_datasets/stack_real/stack_wood/demo_rl.hdf5 \
        --lift_steps 30 --z_max 0.5
"""

import argparse
import json

import h5py
import numpy as np
from tqdm import tqdm

import robosuite


def _make_env(env_args):
    """Create a headless robosuite env from dataset env_args (no renderer needed)."""
    env_kwargs = dict(env_args.get("env_kwargs", {}))
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["use_object_obs"] = True
    env_kwargs["ignore_done"] = True
    env_name = env_args["env_name"]
    return robosuite.make(env_name, **env_kwargs)


def _reset_to_state(env, model_xml, state_vec):
    """Restore env to a given sim state."""
    xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(state_vec)
    env.sim.forward()


def _get_flat_state(env):
    return np.array(env.sim.get_state().flatten())


N_WARMUP = 5  # replay last N actions to warm up OSC controller


def postprocess(args):
    src = h5py.File(args.input, "r")
    data_grp = src["data"]
    demo_keys = sorted(data_grp.keys(), key=lambda x: int(x.split("_")[1]))

    env_args_str = data_grp.attrs.get("env_args", None)
    assert env_args_str is not None, "Dataset missing env_args attribute"
    env_args = json.loads(env_args_str)

    print(f"Creating headless {env_args['env_name']} environment ...")
    env = _make_env(env_args)

    lift_steps = args.lift_steps
    total_extra = lift_steps

    print(f"Will append {lift_steps} lift steps (z_max={args.z_max}, ramped) "
          f"to each of {len(demo_keys)} demos "
          f"(with {N_WARMUP} warmup steps to initialize controller)")

    dst = h5py.File(args.output, "w")
    dst_data = dst.create_group("data")

    for attr_k in data_grp.attrs:
        dst_data.attrs[attr_k] = data_grp.attrs[attr_k]

    total_samples = 0

    for dk in tqdm(demo_keys, desc="Processing demos"):
        demo = data_grp[dk]
        orig_actions = demo["actions"][:]
        orig_states = demo["states"][:]
        ep_len = orig_actions.shape[0]
        action_dim = orig_actions.shape[1]

        model_xml = demo.attrs.get("model_file", None)
        assert model_xml is not None, f"Demo {dk} missing model_file attribute"

        # Restore state N_WARMUP steps before the end and replay those
        # actions so the OSC controller's internal target is correct.
        # Then snap back to orig_states[-1] so there is no state
        # discontinuity (the controller target survives set_state).
        warmup_start = max(0, ep_len - N_WARMUP)
        _reset_to_state(env, model_xml, orig_states[warmup_start])

        for i in range(warmup_start, ep_len):
            env.step(orig_actions[i])

        env.sim.set_state_from_flattened(orig_states[-1])
        env.sim.forward()

        # Append lift actions with a smooth ramp: accelerate for the
        # first quarter, hold steady in the middle, decelerate in
        # the last quarter.  Start from the last action of the demo
        # so there's no discontinuity (especially in the gripper).
        extra_actions = []
        extra_states = []

        last_action = orig_actions[-1].copy()
        ramp_len = max(1, total_extra // 4)

        for step_i in range(total_extra):
            extra_states.append(_get_flat_state(env))

            if step_i < ramp_len:
                z_cmd = args.z_max * (step_i + 1) / ramp_len
            elif step_i >= total_extra - ramp_len:
                remaining = total_extra - step_i
                z_cmd = args.z_max * remaining / ramp_len
            else:
                z_cmd = args.z_max

            action = last_action.copy()
            action[:6] = 0.0
            action[2] = z_cmd

            env.step(action)
            extra_actions.append(action)

        extra_actions = np.array(extra_actions)
        extra_states = np.array(extra_states)

        new_actions = np.concatenate([orig_actions, extra_actions], axis=0)
        new_states = np.concatenate([orig_states, extra_states], axis=0)

        ep_grp = dst_data.create_group(dk)
        ep_grp.create_dataset("actions", data=new_actions)
        ep_grp.create_dataset("states", data=new_states)

        for key in demo.keys():
            if key in ("actions", "states", "obs"):
                continue
            if isinstance(demo[key], h5py.Group):
                demo.copy(key, ep_grp)
            else:
                ep_grp.create_dataset(key, data=demo[key][:])

        for attr_k in demo.attrs:
            ep_grp.attrs[attr_k] = demo.attrs[attr_k]

        ep_grp.attrs["num_samples"] = new_actions.shape[0]
        total_samples += new_actions.shape[0]

    dst_data.attrs["total"] = total_samples

    for attr_k in src.attrs:
        dst.attrs[attr_k] = src.attrs[attr_k]

    src.close()
    dst.close()

    print(f"\nDone. Wrote {len(demo_keys)} demos ({total_samples} total steps) "
          f"to {args.output}")
    print("Next step: run  python -m robomimic.scripts.dataset_states_to_obs  "
          "on the output to generate observations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append lift-up actions to Stack demonstrations"
    )
    parser.add_argument("--input", required=True, help="Path to input HDF5 dataset")
    parser.add_argument("--output", required=True, help="Path to output HDF5 dataset")
    parser.add_argument("--lift_steps", type=int, default=20,
                        help="Number of steps to lift upward (default: 20)")
    parser.add_argument("--z_max", type=float, default=0.5,
                        help="Peak z action magnitude during lift (default: 0.5)")

    args = parser.parse_args()
    postprocess(args)
