"""
Evaluation script for diffusion policy on Stack, MugCleanup, Threading, and
LuggagePush tasks.

Supported tasks (--task):
  stack (default)     Cube stacking (Stack / StackWood environments)
  mug                 Mug cleanup into drawer (MugCleanup / MugCleanupWood)
  thread              Needle threading (Threading / ThreadingWood environments)
  luggage             Push football over luggage ridge (LuggagePush; the wood
                      variant is the same env with table_style/luggage_style
                      kwargs rather than a separate env name)

Supported --eval_env values (with short aliases):
  rs                  Default environment (Stack / MugCleanup / Threading / LuggagePush)
  rsw                 Domain-shifted variant (StackWood / MugCleanupWood /
                      ThreadingWood / LuggagePush+wood)
  rsws                Same as rsw but with agentview45 camera (not available for luggage)

Outputs:
  rewards/reward_seed{seed}.npy   Per-seed reward curves, shape (N+1,)
  videos/test_{seed}.mp4          Per-seed rollout videos (if --save_videos)
  objects_initial_configuration.npy  Initial object poses
  max_reward.npy                  Per-seed max rewards, shape (num_seeds,)
  avg_success_rate.npy            Scalar success rate
  summary.json                    Human-readable summary
  videos/successes.mp4            Compilation of all successful episodes (if --save_videos)

Output directory:
  --output_dir is resolved relative to the agent's run directory when it is a
  relative path. The run directory is taken to be the grandparent of the
  checkpoint (i.e. the parent of the models/ folder). For example, given
      --agent runs/beacon_stack_sim/<run>/models/model_epoch_600.pth
      --output_dir eval_ep350
  results will be written to
      runs/beacon_stack_sim/<run>/eval_ep350
  Pass an absolute path to override this behavior.

Usage examples:

  python eval.py --agent model.pth --task stack --eval_env rs
  python eval.py --agent model.pth --task stack --eval_env rsw --save_videos
  python eval.py --agent model.pth --task mug --eval_env rs --output_dir eval_mug_rs
  python eval.py --agent model.pth --task mug --eval_env rsws --output_dir eval_mug_ws
  python eval.py --agent model.pth --task thread --eval_env rsw --save_videos
"""

import argparse
import json
import os
import time
from copy import deepcopy

import imageio
import numpy as np
import torch
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvType
from robomimic.envs.wrappers import EnvWrapper


# Environment type constants
ROBOSUITE_TYPE = EnvType.ROBOSUITE_TYPE  # 1


def register_custom_task_envs(task):
    """Import side-effecting modules that register custom robosuite envs.

    Some tasks (e.g. ``luggage``) are implemented as in-repo robosuite
    environments that are only added to ``REGISTERED_ENVS`` when their module
    is imported.  Robomimic creates rollout/eval envs via ``robosuite.make``,
    so we must import the module before any env is created.
    """
    if task == "luggage":
        # Importing the module registers ``LuggagePush`` with robosuite.
        import scripts.rs_collect.luggage_env.luggage_push  # noqa: F401

# Canonical short names for eval environments
EVAL_ENV_ALIASES = {
    "robosuite": "rs",
    "robosuite_wood": "rsw",
    "rs_wood": "rsw",
    "rs_wood_s": "rsws",
    "rs_wood_45": "rsws",
}


# Rollout env helpers

class ObsKeyRemapWrapper:
    """Thin wrapper that renames observation dict keys during rollout.

    Used when the evaluation env produces camera obs under a different key
    than what the policy expects (e.g. agentview45_image -> agentview_image).
    """

    def __init__(self, env, key_map):
        self.env = env
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
        return f"ObsKeyRemap({self.env}, {self._key_map})"


# Environment metadata builders

_TASK_ENV_NAMES = {
    "stack": "Stack",
    "mug": "MugCleanup",
    "thread": "Threading",
    "luggage": "LuggagePush",
}


def build_robosuite_env_meta(task="stack", camera_heights=96, camera_widths=96,
                             camera_names=None):
    """Build a robosuite env_meta dict from scratch for a given task.

    This provides default controller configuration for the Panda robot
    with OSC_POSE control, matching the standard robomimic setup.

    Args:
        task:           'stack' or 'mug'
        camera_heights: pixel height for all cameras
        camera_widths:  pixel width for all cameras
        camera_names:   list of camera names (default: agentview + eye_in_hand)

    Returns:
        env_meta dict suitable for EnvUtils.create_env_from_metadata
    """
    if camera_names is None:
        camera_names = ["agentview", "robot0_eye_in_hand"]
    return {
        "env_name": _TASK_ENV_NAMES[task],
        "type": ROBOSUITE_TYPE,
        "env_kwargs": {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": True,
            "control_freq": 20,
            "robots": ["Panda"],
            "camera_depths": False,
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
            "render_gpu_device_id": 0,
            "controller_configs": {
                "type": "BASIC",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2,
                "body_parts": {
                    "right": {
                        "type": "OSC_POSE",
                        "input_max": 1,
                        "input_min": -1,
                        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                        "kp": 150,
                        "damping_ratio": 1,
                        "impedance_mode": "fixed",
                        "kp_limits": [0, 300],
                        "damping_ratio_limits": [0, 10],
                        "position_limits": None,
                        "orientation_limits": None,
                        "uncouple_pos_ori": True,
                        "input_type": "delta",
                        "input_ref_frame": "base",
                        "interpolation": None,
                        "ramp_ratio": 0.2,
                        "gripper": {
                            "type": "GRIP"
                        },
                    }
                },
            },
            "reward_shaping": False,
            "camera_names": camera_names,
        },
    }


# Object pose extraction

def _unwrap_robosuite(env):
    """Unwrap robomimic wrappers to reach the inner robosuite env with .sim."""
    rs_env = env.env if hasattr(env, "env") else env
    while hasattr(rs_env, "env") and not hasattr(rs_env, "sim"):
        rs_env = rs_env.env
    return rs_env


def get_object_poses_stack(env):
    """Extract cubeA and cubeB poses from a robosuite Stack environment.

    Returns:
        (cubeA_pose, cubeB_pose): each shape (7,) = [x, y, z, qw, qx, qy, qz]
    """
    try:
        rs_env = _unwrap_robosuite(env)
        cubeA_body = rs_env.sim.model.body_name2id(rs_env.cubeA.root_body)
        cubeB_body = rs_env.sim.model.body_name2id(rs_env.cubeB.root_body)

        cubeA_pos = rs_env.sim.data.body_xpos[cubeA_body].copy()
        cubeA_quat = rs_env.sim.data.body_xquat[cubeA_body].copy()
        cubeB_pos = rs_env.sim.data.body_xpos[cubeB_body].copy()
        cubeB_quat = rs_env.sim.data.body_xquat[cubeB_body].copy()

        return (
            np.concatenate([cubeA_pos, cubeA_quat]),
            np.concatenate([cubeB_pos, cubeB_quat]),
        )
    except Exception as e:
        print(f"Warning: Could not extract Stack poses: {e}")
        return np.zeros(7), np.zeros(7)


def get_object_poses_mug(env):
    """Extract cleanup_object and drawer poses from a MugCleanup environment.

    Returns:
        (object_pose, drawer_pose): each shape (7,) = [x, y, z, qw, qx, qy, qz]
    """
    try:
        rs_env = _unwrap_robosuite(env)
        obj_body = rs_env.obj_body_id["object"]
        drawer_body = rs_env.obj_body_id["drawer"]

        obj_pos = rs_env.sim.data.body_xpos[obj_body].copy()
        obj_quat = rs_env.sim.data.body_xquat[obj_body].copy()
        drawer_pos = rs_env.sim.data.body_xpos[drawer_body].copy()
        drawer_quat = rs_env.sim.data.body_xquat[drawer_body].copy()

        return (
            np.concatenate([obj_pos, obj_quat]),
            np.concatenate([drawer_pos, drawer_quat]),
        )
    except Exception as e:
        print(f"Warning: Could not extract MugCleanup poses: {e}")
        return np.zeros(7), np.zeros(7)


def get_object_poses_luggage(env):
    """Extract football (ball) and luggage poses from a LuggagePush env.

    Returns:
        (ball_pose, luggage_pose): each shape (7,) = [x, y, z, qw, qx, qy, qz]
    """
    try:
        rs_env = _unwrap_robosuite(env)
        ball_pos = rs_env.sim.data.body_xpos[rs_env.ball_body_id].copy()
        ball_quat = rs_env.sim.data.body_xquat[rs_env.ball_body_id].copy()
        lug_pos = rs_env.sim.data.body_xpos[rs_env.luggage_body_id].copy()
        lug_quat = rs_env.sim.data.body_xquat[rs_env.luggage_body_id].copy()
        return (
            np.concatenate([ball_pos, ball_quat]),
            np.concatenate([lug_pos, lug_quat]),
        )
    except Exception as e:
        print(f"Warning: Could not extract LuggagePush poses: {e}")
        return np.zeros(7), np.zeros(7)


def get_object_poses(env, task="stack"):
    """Extract object poses from the environment based on the task."""
    if task == "mug":
        return get_object_poses_mug(env)
    if task == "luggage":
        return get_object_poses_luggage(env)
    return get_object_poses_stack(env)


# Dense reward

def get_dense_reward(env, task="stack"):
    """Get dense reward from the environment.

    For Stack: uses staged_rewards() normalized to [0, 1].
    For MugCleanup: only sparse binary reward is available; returns env reward.
    """
    try:
        rs_env = _unwrap_robosuite(env)
        if task in ("mug", "luggage"):
            # These envs expose only a sparse success reward.
            return float(rs_env.reward())
        r_reach, r_lift, r_stack = rs_env.staged_rewards()
        return float(max(r_reach, r_lift, r_stack) / 2.0)
    except Exception as e:
        print(f"Warning: Could not get dense reward: {e}")
        return 0.0


# Success check

def check_success(env):
    """Check if the task is successful via the robomimic EnvBase.is_success() interface."""
    try:
        success_dict = env.is_success()
        return bool(success_dict.get("task", False))
    except Exception:
        return False


# Single episode evaluation

def run_single_evaluation(
    policy,
    env,
    seed,
    max_steps,
    task="stack",
    video_path=None,
    camera_name="agentview",
    video_height=256,
    video_width=256,
    use_dense_reward=True,
    strict=False,
):
    """
    Run a single evaluation episode and optionally record a video.

    Args:
        policy:          RolloutPolicy instance
        env:             robomimic-wrapped environment
        seed:            random seed for deterministic initial configuration
        max_steps:       maximum number of environment steps
        task:            'stack' or 'mug'
        video_path:      if provided, save video to this path
        camera_name:     camera name for video rendering
        video_height:    video frame height in pixels
        video_width:     video frame width in pixels
        use_dense_reward: if True, use dense reward
        strict:          if True, require 5 consecutive success frames or
                         success on the last frame

    Returns:
        save_rewards: np.array (N+1,) — reward at each step, starting with 0
        max_reward:   float — maximum reward achieved
        success:      bool — whether the task was completed successfully
        obj1_pose:    np.array (7,) — initial pose of first object
        obj2_pose:    np.array (7,) — initial pose of second object
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy.start_episode()
    obs = env.reset()

    obj1_pose, obj2_pose = get_object_poses(env, task=task)

    imgs = []
    save_rewards = [0.0]
    success = False
    consecutive_success = 0
    SUSTAINED_SUCCESS_FRAMES = 5
    last_frame_success = False

    if video_path is not None:
        frame = env.render(
            mode="rgb_array", height=video_height, width=video_width,
            camera_name=camera_name,
        )
        imgs.append(frame)

    step_idx = 0
    done = False

    try:
        while not done:
            act = policy(ob=obs)
            next_obs, reward, terminated, info = env.step(act)

            if use_dense_reward:
                reward = get_dense_reward(env, task=task)
            else:
                reward = float(reward)

            save_rewards.append(reward)

            if not success:
                frame_success = check_success(env)
                if strict:
                    if frame_success:
                        consecutive_success += 1
                    else:
                        consecutive_success = 0
                    if consecutive_success >= SUSTAINED_SUCCESS_FRAMES:
                        success = True
                    last_frame_success = frame_success
                else:
                    success = frame_success

            if video_path is not None:
                frame = env.render(
                    mode="rgb_array", height=video_height, width=video_width,
                    camera_name=camera_name,
                )
                imgs.append(frame)

            step_idx += 1
            if step_idx >= max_steps:
                done = True

            obs = deepcopy(next_obs)

    except env.rollout_exceptions as e:
        print(f"Warning: rollout exception for seed {seed}: {e}")
    except Exception as e:
        print(f"Warning: unexpected exception for seed {seed}: {e}")

    if strict and not success and last_frame_success:
        success = True

    if video_path is not None and len(imgs) > 0:
        imageio.mimwrite(video_path, imgs, fps=20)

    save_rewards = np.array(save_rewards, dtype=np.float32)
    max_reward = float(np.max(save_rewards))

    return save_rewards, max_reward, success, obj1_pose, obj2_pose


# Success video compilation

def compile_success_video(success_video_paths, output_path, fps=20):
    """Concatenate all successful episode videos into a single compilation.

    Reads each video, appends all frames sequentially, and writes one
    combined video. Inserts a brief black-frame separator between episodes.

    Args:
        success_video_paths: list of paths to individual success episode videos
        output_path: path to write the combined video
        fps: frames per second for the output video
    """
    if not success_video_paths:
        print("No successful episodes to compile into video.")
        return

    all_frames = []
    for vpath in success_video_paths:
        if not os.path.exists(vpath):
            continue
        reader = imageio.get_reader(vpath)
        frames = [frame for frame in reader]
        reader.close()
        if not frames:
            continue

        if all_frames:
            h, w = frames[0].shape[:2]
            separator = np.zeros((h, w, 3), dtype=np.uint8)
            all_frames.extend([separator] * 5)

        all_frames.extend(frames)

    if all_frames:
        imageio.mimwrite(output_path, all_frames, fps=fps)
        print(f"Compiled {len(success_video_paths)} successful episodes "
              f"into {output_path} ({len(all_frames)} frames)")
    else:
        print("No frames found in successful episode videos.")


# DDIM inference override

def override_scheduler_to_ddim(policy, num_inference_steps):
    """Replace the model's noise scheduler with DDIM for faster inference.

    A DDPM-trained model can use a DDIM scheduler at inference time because
    they share the same forward-process noise schedule.  DDIM removes the
    stochastic noise injection at each denoising step, allowing much larger
    step sizes (e.g. 10 steps instead of 100) with comparable quality.

    Modifies the model **in-place**: swaps noise_scheduler and patches
    algo_config so that _get_action_trajectory reads the correct flags.

    Args:
        policy: RolloutPolicy wrapping a DiffusionPolicyUNet (or subclass)
        num_inference_steps: number of DDIM denoising steps (e.g. 10)
    """
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    model = policy.policy
    old = model.noise_scheduler

    ddim = DDIMScheduler(
        num_train_timesteps=old.config.num_train_timesteps,
        beta_schedule=old.config.beta_schedule,
        clip_sample=old.config.clip_sample,
        prediction_type=old.config.prediction_type,
        set_alpha_to_one=True,
        steps_offset=0,
    )
    model.noise_scheduler = ddim

    # Patch algo_config flags so _get_action_trajectory uses DDIM path
    with model.algo_config.values_unlocked():
        model.algo_config.ddpm.enabled = False
        model.algo_config.ddim.enabled = True
        model.algo_config.ddim.num_inference_timesteps = num_inference_steps

    print(f"[DDIM override] Replaced noise scheduler: "
          f"{old.__class__.__name__} -> DDIMScheduler, "
          f"inference steps: {num_inference_steps}")


# Environment type resolution

def resolve_eval_env_type(eval_env_str, task="stack"):
    """Convert CLI --eval_env string to (env_name_override, extras).

    Normalizes aliases (e.g. "rs_wood" -> "rsw") and returns:
      - env_name_override: e.g. "StackWood" / "MugCleanup", or None for default
      - extras dict with optional keys: camera_names, obs_key_map, video_camera_name

    Args:
        eval_env_str: raw CLI string
        task: 'stack' or 'mug'

    Returns:
        (env_name_override, extras)
    """
    s = EVAL_ENV_ALIASES.get(eval_env_str.strip().lower(), eval_env_str.strip().lower())

    if task == "mug":
        if s == "rs":
            return "MugCleanup", {}
        elif s == "rsw":
            return "MugCleanupWood", {}
        elif s == "rsws":
            return "MugCleanupWood", {
                "camera_names": ["agentview45", "robot0_eye_in_hand"],
                "video_camera_name": "agentview45",
            }
        else:
            raise ValueError(
                f"Unknown --eval_env '{eval_env_str}' for task 'mug'. "
                "Use 'rs', 'rsw', or 'rsws'."
            )

    if task == "thread":
        # Threading keeps its native D0 layout (high mimicgen yield). To
        # avoid the robot arm occluding the ring hole from the side-view
        # camera, the Threading env internally overrides the `agentview45`
        # camera pose to a mirrored position on the -y side of the table
        # (see mimicgen/envs/robosuite/threading.py:Threading._load_model).
        # So the "side view" here is the mirrored agentview45, not a
        # different task layout.
        if s == "rs":
            return "Threading", {}
        elif s == "rsw":
            return "ThreadingWood", {}
        elif s == "rsws":
            return "ThreadingWood", {
                "camera_names": ["agentview45", "robot0_eye_in_hand"],
                "video_camera_name": "agentview45",
            }
        else:
            raise ValueError(
                f"Unknown --eval_env '{eval_env_str}' for task 'thread'. "
                "Use 'rs', 'rsw', or 'rsws'."
            )

    if task == "luggage":
        # The wood/domain-shifted variant is the SAME env (LuggagePush) with
        # different appearance kwargs, not a separate env name. We inject those
        # kwargs via ``env_kwargs_update`` (applied to env_metadata in main()).
        if s == "rs":
            return "LuggagePush", {}
        elif s == "rsw":
            return "LuggagePush", {
                "env_kwargs_update": {
                    "table_style": "wood",
                    "luggage_style": "gray",
                },
            }
        else:
            raise ValueError(
                f"Unknown --eval_env '{eval_env_str}' for task 'luggage'. "
                "Use 'rs' (ceramic/black) or 'rsw' (wood/gray). "
                "The 'rsws' agentview45 variant is not available for luggage."
            )

    # task == "stack" (default)
    if s == "rs":
        return None, {}
    elif s == "rsw":
        return "StackWood", {}
    elif s == "rsws":
        return "StackWood", {
            "camera_names": ["agentview45", "robot0_eye_in_hand"],
            "video_camera_name": "agentview45",
        }
    else:
        raise ValueError(
            f"Unknown --eval_env '{eval_env_str}'. "
            "Use 'rs', 'rsw', or 'rsws'."
        )


# Main evaluation

def main(args):
    """
    Load a checkpoint, create the evaluation environment, run episodes,
    and save results (rewards, videos, summary).

    Prints a real-time running success rate after each episode.
    When --save_videos is enabled, also compiles all successful episode
    videos into a single compilation video.

    Args:
        args: parsed argparse namespace
    """
    eval_start_time = time.time()

    # Register custom task environments (e.g. LuggagePush)
    register_custom_task_envs(args.task)

    # Resolve evaluation environment
    env_name_from_type, eval_extras = resolve_eval_env_type(args.eval_env, task=args.task)
    if args.env is None and env_name_from_type is not None:
        args.env = env_name_from_type

    video_camera_name = eval_extras.get("video_camera_name", args.camera_name)

    # Resolve output directory
    # Convention: agent path is <run_dir>/models/<ckpt>.pth, so run_dir is
    # the grandparent of the checkpoint. If --output_dir is a relative path,
    # resolve it inside run_dir so the user only needs to pass e.g.
    # "eval_ep350" instead of repeating the full run path.
    if not os.path.isabs(args.output_dir):
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.agent)))
        args.output_dir = os.path.join(run_dir, args.output_dir)
        print(f"Resolved --output_dir relative to run_dir: {args.output_dir}")

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    rewards_dir = os.path.join(args.output_dir, "rewards")
    os.makedirs(rewards_dir, exist_ok=True)
    if args.save_videos:
        videos_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

    # Load checkpoint
    print("Loading policy from checkpoint...")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=args.agent)

    # Co-training checkpoints store env_metadata / shape_metadata as lists;
    # flatten to a single dict for the evaluation path.
    if isinstance(ckpt_dict.get("env_metadata"), list):
        chosen = None
        if env_name_from_type is not None:
            for meta in ckpt_dict["env_metadata"]:
                if meta.get("env_name") == env_name_from_type:
                    chosen = meta
                    break
        ckpt_dict["env_metadata"] = chosen or ckpt_dict["env_metadata"][0]

    if isinstance(ckpt_dict.get("shape_metadata"), list):
        ckpt_dict["shape_metadata"] = ckpt_dict["shape_metadata"][0]

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict, device=device, verbose=True,
    )

    # Optional: override to DDIM for faster inference
    if args.ddim_steps is not None:
        override_scheduler_to_ddim(policy, num_inference_steps=args.ddim_steps)

    # Override env_metadata if needed
    eval_label = args.eval_env
    if "camera_names" in eval_extras:
        old_meta = ckpt_dict["env_metadata"]
        new_meta = deepcopy(old_meta)
        new_meta["env_kwargs"]["camera_names"] = eval_extras["camera_names"]
        ckpt_dict["env_metadata"] = new_meta
        print(f"Overriding camera_names to {eval_extras['camera_names']}")

    # Inject arbitrary env_kwargs (e.g. luggage wood/gray appearance variant).
    if "env_kwargs_update" in eval_extras:
        new_meta = deepcopy(ckpt_dict["env_metadata"])
        new_meta["env_kwargs"].update(eval_extras["env_kwargs_update"])
        ckpt_dict["env_metadata"] = new_meta
        print(f"Overriding env_kwargs with {eval_extras['env_kwargs_update']}")

    # Horizon
    max_steps = args.horizon
    if max_steps is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        max_steps = config.experiment.rollout.horizon

    # Create environment
    print("Creating environment...")
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=args.save_videos,
        verbose=True,
    )

    obs_key_map = eval_extras.get("obs_key_map", {})
    if obs_key_map:
        env = ObsKeyRemapWrapper(env, obs_key_map)
        print(f"Applied obs key remapping: {obs_key_map}")

    # Storage
    obj1_poses = []
    obj2_poses = []
    save_max_reward = []
    successes = []
    success_video_paths = []

    seed_start = args.start_seed
    seed_end = args.start_seed + args.num_seeds

    print("\n--- Starting evaluation ---")
    print(f"  Task             : {args.task}")
    print(f"  Evaluating on    : {eval_label}")
    print(f"  Seeds            : {seed_start} to {seed_end - 1} ({args.num_seeds} total)")
    print(f"  Max steps/ep     : {max_steps}")
    print(f"  Save videos      : {args.save_videos}")
    print(f"  Video resolution : {args.video_height}x{args.video_width}")
    print(f"  Camera           : {video_camera_name}")
    print(f"  Reward type      : {'dense' if args.dense_reward else 'sparse'}")
    print(f"  Strict success   : {args.strict}")
    if args.ddim_steps is not None:
        print(f"  Scheduler        : DDIM ({args.ddim_steps} steps, overridden)")
    else:
        print(f"  Scheduler        : from checkpoint (no override)")
    print(f"  Output directory : {args.output_dir}")

    # Main evaluation loop
    num_successes = 0
    pbar = tqdm(range(seed_start, seed_end), desc="Evaluating")

    for test_index in pbar:
        video_path = None
        if args.save_videos:
            video_path = os.path.join(videos_dir, f"test_{test_index}.mp4")

        save_rewards, max_reward, success, obj1_pose, obj2_pose = (
            run_single_evaluation(
                policy=policy,
                env=env,
                seed=test_index,
                max_steps=max_steps,
                task=args.task,
                video_path=video_path,
                camera_name=video_camera_name,
                video_height=args.video_height,
                video_width=args.video_width,
                use_dense_reward=args.dense_reward,
                strict=args.strict,
            )
        )

        np.save(
            os.path.join(rewards_dir, f"reward_seed{test_index}.npy"),
            save_rewards,
        )

        obj1_poses.append(obj1_pose)
        obj2_poses.append(obj2_pose)
        save_max_reward.append(max_reward)
        successes.append(success)

        if success:
            num_successes += 1
            if video_path is not None:
                success_video_paths.append(video_path)

        # Real-time success rate in progress bar
        n_done = test_index - seed_start + 1
        current_rate = num_successes / n_done
        pbar.set_postfix(
            success=f"{num_successes}/{n_done}",
            rate=f"{current_rate * 100:.1f}%",
        )

    # Aggregate & save
    obj1_poses = np.asarray(obj1_poses)
    obj2_poses = np.asarray(obj2_poses)
    save_max_reward = np.asarray(save_max_reward)
    successes = np.asarray(successes)

    np.save(
        os.path.join(args.output_dir, "objects_initial_configuration.npy"),
        {"obj1_pose": obj1_poses, "obj2_pose": obj2_poses},
        allow_pickle=True,
    )
    np.save(
        os.path.join(args.output_dir, "max_reward.npy"),
        save_max_reward, allow_pickle=True,
    )

    avg_success_rate = float(successes.mean())
    np.save(
        os.path.join(args.output_dir, "avg_success_rate.npy"),
        np.array(avg_success_rate, dtype=np.float32),
    )

    # Compile success video
    if args.save_videos and success_video_paths:
        compile_success_video(
            success_video_paths,
            os.path.join(videos_dir, "successes.mp4"),
            fps=20,
        )

    # Summary
    elapsed_min = (time.time() - eval_start_time) / 60.0
    summary = {
        "task": args.task,
        "eval_env": eval_label,
        "num_seeds": args.num_seeds,
        "start_seed": args.start_seed,
        "end_seed": seed_end - 1,
        "max_steps": max_steps,
        "reward_type": "dense" if args.dense_reward else "sparse",
        "success_rate": avg_success_rate,
        "avg_max_reward": float(save_max_reward.mean()),
        "num_successes": int(successes.sum()),
        "elapsed_minutes": round(elapsed_min, 2),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Evaluation complete ---")
    print(f"  Evaluated on     : {eval_label}")
    print(f"  Total episodes   : {args.num_seeds}")
    print(f"  Successes        : {int(successes.sum())}")
    print(f"  Success rate     : {avg_success_rate * 100:.2f}%")
    print(f"  Avg max reward   : {save_max_reward.mean():.4f}")
    print(f"  Elapsed time     : {elapsed_min:.1f} min")
    print(f"\nSaved files:")
    print(f"  - {rewards_dir}/reward_seed*.npy ({args.num_seeds} files)")
    if args.save_videos:
        print(f"  - {videos_dir}/test_*.mp4 ({args.num_seeds} files)")
        if success_video_paths:
            print(f"  - {videos_dir}/successes.mp4 "
                  f"({len(success_video_paths)} episodes compiled)")
    print(f"  - {args.output_dir}/objects_initial_configuration.npy")
    print(f"  - {args.output_dir}/max_reward.npy (shape: {save_max_reward.shape})")
    print(f"  - {args.output_dir}/avg_success_rate.npy")
    print(f"  - {args.output_dir}/summary.json")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation for diffusion policy (Stack, MugCleanup, "
                    "Threading, and LuggagePush tasks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tasks:
  stack      Cube stacking (Stack / StackWood environments)
  mug        Mug cleanup into drawer (MugCleanup / MugCleanupWood)
  thread     Needle threading (Threading / ThreadingWood environments)
  luggage    Push football over luggage ridge (LuggagePush; wood variant is the
             same env with table_style/luggage_style kwargs)

Environment aliases:
  rs         Default env (Stack / MugCleanup / Threading / LuggagePush)
  rsw        Domain-shifted variant (StackWood / MugCleanupWood / ThreadingWood
             / LuggagePush+wood)
  rsws       Same as rsw but with agentview45 camera (not available for luggage)

  Legacy aliases (still accepted): rs_wood -> rsw, rs_wood_s -> rsws,
  robosuite -> rs

Examples:
  python eval.py --agent model.pth --task stack --eval_env rs
  python eval.py --agent model.pth --task stack --eval_env rsws --save_videos
  python eval.py --agent model.pth --task mug --eval_env rs
  python eval.py --agent model.pth --task mug --eval_env rsw --save_videos
  python eval.py --agent model.pth --task thread --eval_env rsw --save_videos
  python eval.py --agent model.pth --task luggage --eval_env rs --save_videos
  python eval.py --agent model.pth --task luggage --eval_env rsw --save_videos
""",
    )

    parser.add_argument(
        "--task", type=str, default="stack",
        choices=["stack", "mug", "thread", "luggage"],
        help="Task to evaluate: 'stack' (cube stacking), 'mug' (mug cleanup), "
             "'thread' (needle threading), or 'luggage' (push football over "
             "luggage ridge). Default: 'stack'.",
    )
    parser.add_argument(
        "--agent", type=str, required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--eval_env", type=str, required=True,
        help="Environment to evaluate in: 'rs', 'rsw', 'rsws' "
             "(legacy names like 'rs_wood', 'robosuite' also accepted)",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=5000,
        help="Number of evaluation seeds (default: 5000)",
    )
    parser.add_argument(
        "--start_seed", type=int, default=0,
        help="Starting seed number (default: 0)",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Max steps per episode (default: from checkpoint config)",
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Override environment name (e.g. 'Stack', 'StackWood', "
             "'MugCleanup', or 'Threading')",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results",
        help="Directory to save results. If a relative path, it is resolved "
             "inside the agent's run directory (the grandparent of the "
             "checkpoint, i.e. parent of models/). Pass an absolute path to "
             "override. Default: 'eval_results' (i.e. <run_dir>/eval_results).",
    )
    parser.add_argument(
        "--save_videos", action="store_true",
        help="Save video for each seed + compile successes into one video",
    )
    parser.add_argument(
        "--camera_name", type=str, default="agentview",
        help="Camera to use for video rendering (default: agentview)",
    )
    parser.add_argument(
        "--video_height", type=int, default=256,
        help="Video frame height (default: 256)",
    )
    parser.add_argument(
        "--video_width", type=int, default=256,
        help="Video frame width (default: 256)",
    )
    parser.add_argument(
        "--dense_reward", action="store_true",
        help="Use dense (shaped) reward instead of sparse binary reward",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Strict success: require 5 consecutive success frames or success "
             "on the last frame (filters out transient bounces)",
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=None,
        help="Override noise scheduler to DDIM with this many denoising steps "
             "(e.g. 10). Gives ~10x faster inference with minimal quality loss. "
             "Works with any DDPM-trained checkpoint.",
    )

    args = parser.parse_args()

    try:
        main(args)
        print("Finished evaluation successfully!")
    except Exception:
        import traceback
        print("Eval failed with error:\n{}".format(traceback.format_exc()))
