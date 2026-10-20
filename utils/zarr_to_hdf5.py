"""
Convert a real-robot teleoperation dataset from zarr to robomimic HDF5.

Expected input layout (the replay-buffer format used by the diffusion-policy
real-robot stack), where N is the number of timesteps across all episodes:

    meta/episode_ends            (E,)       exclusive end index of each episode
    data/action                  (N, 7)     6-DoF delta pose + gripper (-1/+1)
    data/robot_eef_pose          (N, 6)     xyz + rotation vector
    data/robot_joint             (N, 7)
    data/robot_joint_vel         (N, 7)
    data/camera_0                (N, H, W, 3)

Timesteps whose arm action is ~zero are dropped, since teleoperation contains
long idle stretches that bias the behaviour-cloning loss. Images are
center-cropped to a square and resized to 96x96 to match the sim datasets.
Fields with no real-robot counterpart (`object`, `states`) are written as zeros
and are unused by the image-based policies; reward/done are set on the final
timestep of each episode. All demos are listed under the `mask/train` filter key.

Usage:
    python utils/zarr_to_hdf5.py \
        --zarr data/stack_R/stack_R.zarr \
        --out data/stack_R/real_stack_40.hdf5 \
        --env-name Stack
"""

import argparse
import json
import numpy as np
import zarr
import h5py
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm

IMG_SIZE = 96
ZERO_ACTION_THRESH = 1e-3


def center_crop_and_resize(imgs, target_size=IMG_SIZE):
    """center-crop a (T, H, W, 3) array to square, then resize"""
    H, W = imgs.shape[1], imgs.shape[2]
    crop = (W - H) // 2
    imgs_cropped = imgs[:, :, crop : W - crop, :]
    out = np.empty((len(imgs), target_size, target_size, 3), dtype=np.uint8)
    for i in range(len(imgs)):
        out[i] = cv2.resize(
            imgs_cropped[i], (target_size, target_size),
            interpolation=cv2.INTER_AREA,
        )
    return out


def rotvec_to_quat_xyzw(rotvec):
    """(N, 3) rotvec -> (N, 4) quaternion in xyzw """
    return Rotation.from_rotvec(rotvec).as_quat()


def gripper_action_to_qpos(grip_actions):
    """
    Map binary grip action (-1=open, +1=closed) to approximate Panda gripper qpos.
    finger1 in [0, 0.04], finger2 in [-0.04, 0].
    """
    finger1 = 0.02 - 0.02 * grip_actions
    finger2 = -0.02 + 0.02 * grip_actions
    return np.stack([finger1, finger2], axis=-1)


def shift_by_one(arr):
    return np.concatenate([arr[1:], arr[-1:]], axis=0)


def main(zarr_path, out_path, env_name):
    z = zarr.open(zarr_path, "r")
    ep_ends = z["meta"]["episode_ends"][:]
    n_eps = len(ep_ends)
    total_samples = int(ep_ends[-1])

    print(f"Converting {n_eps} episodes, {total_samples} total timesteps")

    env_args = {
        "env_name": env_name,
        "env_version": "1.5.2",
        "type": 1,
        "env_kwargs": {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": True,
            "control_freq": 20,
            "robots": ["Panda"],
            "camera_depths": False,
            "camera_heights": IMG_SIZE,
            "camera_widths": IMG_SIZE,
            "render_gpu_device_id": 0,
            "reward_shaping": False,
            "camera_names": ["agentview"],
        },
    }

    with h5py.File(out_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env_args"] = json.dumps(env_args, indent=4)

        demo_keys = []
        total_kept = 0
        for ep_i in tqdm(range(n_eps), desc="Episodes"):
            start = 0 if ep_i == 0 else int(ep_ends[ep_i - 1])
            end = int(ep_ends[ep_i])
            T = end - start

            actions_raw = np.squeeze(z["data"]["action"][start:end])
            eef_pose_raw = np.squeeze(z["data"]["robot_eef_pose"][start:end])
            joints_raw = np.squeeze(z["data"]["robot_joint"][start:end])
            joint_vels_raw = np.squeeze(z["data"]["robot_joint_vel"][start:end])

            # Filter out idle timesteps (arm action L2 norm ~ 0)
            arm_norm = np.linalg.norm(actions_raw[:, :6], axis=1)
            keep = arm_norm > ZERO_ACTION_THRESH
            n_before = len(actions_raw)
            actions = actions_raw[keep]
            eef_pose = eef_pose_raw[keep]
            joints = joints_raw[keep]
            joint_vels = joint_vels_raw[keep]
            T = len(actions)
            print(f"  ep {ep_i}: {n_before} -> {T} (removed {n_before - T} idle frames)")

            # Read & process images only for kept indices
            keep_indices = np.where(keep)[0] + start
            cam0 = z["data"]["camera_0"].get_orthogonal_selection(keep_indices)
            agentview = center_crop_and_resize(cam0)

            eef_pos = eef_pose[:, :3].astype(np.float64)
            eef_quat = rotvec_to_quat_xyzw(eef_pose[:, 3:6]).astype(np.float64)
            joint_pos = joints.astype(np.float64)
            joint_vel = joint_vels.astype(np.float64)
            joint_pos_cos = np.cos(joint_pos)
            joint_pos_sin = np.sin(joint_pos)
            gripper_qpos = gripper_action_to_qpos(actions[:, 6]).astype(np.float64)
            gripper_qvel = np.zeros((T, 2), dtype=np.float64)

            if T == 0:
                print(f"  WARNING: ep {ep_i} has 0 frames after filtering, skipping")
                continue

            object_state = np.zeros((T, 23), dtype=np.float64)
            states = np.zeros((T, 45), dtype=np.float64)
            rewards = np.zeros(T, dtype=np.float64)
            rewards[-1] = 1.0
            dones = np.zeros(T, dtype=np.int64)
            dones[-1] = 1

            total_kept += T
            demo_name = f"demo_{len(demo_keys)}"
            demo_keys.append(demo_name)
            dg = data_grp.create_group(demo_name)
            dg.attrs["num_samples"] = T

            dg.create_dataset("actions", data=actions.astype(np.float64))
            dg.create_dataset("rewards", data=rewards)
            dg.create_dataset("dones", data=dones)
            dg.create_dataset("states", data=states)

            obs_data = {
                "agentview_image": agentview,
                "robot0_eef_pos": eef_pos,
                "robot0_eef_quat": eef_quat,
                "robot0_gripper_qpos": gripper_qpos,
                "robot0_gripper_qvel": gripper_qvel,
                "robot0_joint_pos": joint_pos,
                "robot0_joint_pos_cos": joint_pos_cos,
                "robot0_joint_pos_sin": joint_pos_sin,
                "robot0_joint_vel": joint_vel,
                "object": object_state,
            }

            obs = dg.create_group("obs")
            nobs = dg.create_group("next_obs")
            for key, val in obs_data.items():
                obs.create_dataset(key, data=val)
                nobs.create_dataset(key, data=shift_by_one(val))

        data_grp.attrs["total"] = total_kept

        mask_grp = f.create_group("mask")
        mask_grp.create_dataset("train", data=np.array(demo_keys, dtype="S"))

    print(f"\nSaved {out_path}")
    print(f"  {len(demo_keys)} demos, {total_kept} timesteps (from {total_samples}, removed {total_samples - total_kept} idle frames)")

    with h5py.File(out_path, "r") as f:
        d0 = f["data/demo_0"]
        print(f"\nVerification (demo_0):")
        print(f"  actions:           {d0['actions'].shape}")
        print(f"  agentview_image:   {d0['obs/agentview_image'].shape}")
        print(f"  eef_pos:           {d0['obs/robot0_eef_pos'].shape}")
        print(f"  eef_quat:          {d0['obs/robot0_eef_quat'].shape}")
        print(f"  gripper_qpos:      {d0['obs/robot0_gripper_qpos'].shape}")
        print(f"  joint_pos:         {d0['obs/robot0_joint_pos'].shape}")
        print(f"  joint_vel:         {d0['obs/robot0_joint_vel'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a real-robot .zarr dataset to robomimic .hdf5")
    parser.add_argument("--zarr", default="data/luggage_R/luggage.zarr", help="Input .zarr path")
    parser.add_argument("--out", default=None, help="Output .hdf5 path (default: alongside the zarr)")
    parser.add_argument("--env-name", default="LuggagePush", help="env_name stored in env_args")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = args.zarr.rstrip("/").rsplit(".zarr", 1)[0] + ".hdf5"

    main(args.zarr, out_path, args.env_name)
