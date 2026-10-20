"""
Collect scripted demonstrations for the LuggagePush environment.

A small state-machine policy drives the (closed) gripper behind the football in
the left half of the luggage and pushes it forward over the central ridge into
the right half.  Successful episodes are written to a robomimic-format HDF5
dataset (data/demo_i/{actions, states, obs/*}, with env metadata) and a preview
MP4 is rendered.

The domain variants of this task are appearance changes of the same environment,
selected with --table_style (ceramic / wood) and --luggage_style (black / gray).

Usage:
    MUJOCO_GL=egl python collect_demo.py --n_demos 20
    MUJOCO_GL=egl python collect_demo.py --n_demos 5 --render_video
    MUJOCO_GL=egl python collect_demo.py --n_demos 250 --table_style wood \
        --out ../../../data/luggage/rsw_luggage_250.hdf5
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robosuite as suite  # noqa: E402
from robosuite.utils import transform_utils as T  # noqa: E402
from luggage_push import LuggagePush  # noqa: E402,F401  (registers env)

HERE = os.path.dirname(os.path.abspath(__file__))


def face_ball_quat(angle_deg=90.0):
    """Target eef orientation: rotate the gripper from straight-down toward the
    ball by ``angle_deg`` about the world-x axis, so the approach axis swings
    from -z (down) toward +y (facing the ball).  90 deg => fully horizontal.
    """
    ang = np.deg2rad(angle_deg)
    z_ax = np.array([0.0, np.sin(ang), -np.cos(ang)])   # down -> +y
    x_ax = np.array([1.0, 0.0, 0.0])
    y_ax = np.cross(z_ax, x_ax)
    R = np.column_stack([x_ax, y_ax, z_ax])
    return T.mat2quat(R)


def facedown_yaw_quat(yaw_deg=90.0):
    """Top-down gripper twisted about the vertical axis by ``yaw_deg`` so its
    flat face is presented to the ball along the push (+y) direction.  Unlike a
    horizontal gripper, this pose is comfortably reachable over the luggage.
    """
    ang = np.deg2rad(yaw_deg)
    c, s = np.cos(ang), np.sin(ang)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return T.mat2quat(Rz @ R_down)


def make_controller_config():
    """OSC_POSE in the WORLD frame so scripted deltas are intuitive."""
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "OSC_POSE",
                "input_max": 1, "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150, "damping_ratio": 1, "impedance_mode": "fixed",
                "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
                "position_limits": None, "orientation_limits": None,
                "uncouple_pos_ori": True, "input_type": "delta",
                "input_ref_frame": "world", "interpolation": None, "ramp_ratio": 0.2,
                "gripper": {"type": "GRIP"},
            }
        },
    }


def make_env(camera_h, camera_w, horizon, seed=None,
             table_style="ceramic", luggage_style="black"):
    cfg = make_controller_config()
    env = suite.make(
        "LuggagePush",
        robots="Panda",
        controller_configs=cfg,
        seed=seed,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        camera_names=["agentview"],
        camera_heights=camera_h,
        camera_widths=camera_w,
        control_freq=20,
        horizon=horizon,
        ignore_done=True,
        hard_reset=True,
        render_gpu_device_id=0,
        table_style=table_style,
        luggage_style=luggage_style,
    )
    return env, cfg


class PushPolicy:
    """State machine that rolls the ball sideways (+y) over the ridge in TWO
    steps, pushing with the gripper's flat FACE (a 90 deg vertical-axis twist).

    Step 1: from the saddle between the two handle bumps, roll the ball over the
            right bump so it sits between that bump and the ridge.
    Step 2: re-approach and roll it over the ridge into the right half.
    """

    def __init__(self, table_z=0.8, yaw_deg=90.0, z_high=None, lift_max=0.08):
        self.table_z = table_z
        # twist the gripper 90 deg about vertical so its broad FACE (not the thin
        # edge) is presented to the ball along the push direction
        self.q_face = facedown_yaw_quat(yaw_deg)
        self.z_high = (table_z + 0.18) if z_high is None else z_high
        self.lift_max = lift_max
        self.mid_y = -0.05             # rest spot between right bump and ridge
        # per-step push contact: step 1 pushes low behind the ball (flat roll
        # over the small bump); step 2 contacts higher to clear the right bump
        # and shove the ball's upper-back over the tall ridge
        self.push_z = {1: table_z + 0.07, 2: table_z + 0.065}
        self.reset()

    def reset(self):
        self.phase = "detour"
        self.step = 1                  # which of the two rolls we are doing
        self.bx = None                 # locked push-lane x
        self.by = None                 # locked behind-y
        self.push_steps = 0
        self._init = False

    def _behind(self, ball):
        # how far behind the ball (in -y) to line up.  For the first roll, shrink
        # it for balls near the left wall so the gripper never descends outside
        # the luggage (inner wall ~ -0.27 in the luggage frame).
        if self.step == 1:
            return float(np.clip(ball[1] + 0.27, 0.06, 0.13))
        return 0.07

    def __call__(self, obs):
        eef = obs["robot0_eef_pos"]
        quat = obs["robot0_eef_quat"]
        ball = obs["ball_pos"]
        a = np.zeros(7)
        a[6] = 1.0  # keep gripper closed (a compact pushing tool)

        # orientation: keep the face turned toward the ball throughout
        a[3:6] = np.clip(2.0 * T.get_orientation_error(self.q_face, quat), -1.0, 1.0)

        # adaptive start: if the ball already sits between the right bump and the
        # ridge, a single roll over the ridge suffices
        if not self._init:
            self._init = True
            if ball[1] > -0.10:
                self.step = 2

        kp = 12.0
        z_high = self.z_high
        push_z = self.push_z[self.step]
        behind = self._behind(ball)
        side_x = ball[0] + 0.16

        if self.phase == "detour":
            target = np.array([side_x, ball[1] - behind, z_high])
            if np.linalg.norm(eef[:2] - target[:2]) < 0.03:
                self.bx, self.by = float(ball[0]), float(ball[1] - behind)
                self.phase = "behind"
        elif self.phase == "behind":
            target = np.array([self.bx, self.by, z_high])
            if abs(eef[0] - self.bx) < 0.02:
                self.phase = "lower"
        elif self.phase == "lower":
            target = np.array([self.bx, self.by, push_z])
            if abs(eef[2] - push_z) < 0.02:
                self.phase = "push"
        elif self.phase == "push":
            self.push_steps += 1
            if self.step == 1:
                # roll over the right bump to the spot before the ridge.  The
                # gripper starts behind bump_b, so climb slightly as it shoves so
                # the face rides over bump_b (instead of catching on it) and then
                # over the right bump.  Track just ahead of the ball so the face
                # never outruns it and loses contact (the lower base gives the OSC
                # less authority far out, so the ball can lag).
                lift = float(np.clip((self.push_steps - 4) * 0.004, 0.0, 0.04))
                ty = min(self.mid_y + 0.04, float(ball[1]) + 0.05)
                target = np.array([self.bx, ty, push_z + lift])
                if ball[1] > self.mid_y - 0.03:
                    self.phase, self.bx, self.by = "raise", None, None
            else:
                # roll over the ridge: shove forward, then progressively lift so
                # the face scoops the ball up the crest (the ball starts right at
                # the ridge with no runway, so ramp the lift over time).  Keep the
                # forward target reachable for the lowered base and a modest lift
                # so the face drives the ball OVER rather than scooping above it.
                lift = float(np.clip((self.push_steps - 8) * 0.005, 0.0, self.lift_max))
                target = np.array([self.bx, 0.22, push_z + lift])
        else:  # "raise": lift clear, then re-approach for the second roll
            target = np.array([eef[0], ball[1] - behind, z_high])
            if eef[2] > z_high - 0.03:
                self.phase, self.step, self.push_steps = "detour", 2, 0

        a[:3] = np.clip(kp * (target - eef), -1.0, 1.0)
        return a


def rollout(env, policy, max_steps, settle_steps=40):
    """Run one episode; return dict of trajectory data and success flag."""
    env.reset()
    # let the ball settle onto the rails before recording
    for _ in range(settle_steps):
        env.step(np.zeros(env.action_dim))
    policy.reset()

    obs = env._get_observations(force_update=True)
    obs_keys = [k for k in obs.keys() if not k.endswith("_image") or k == "agentview_image"]

    traj = {"actions": [], "states": [], "obs": {k: [] for k in obs_keys}, "rewards": []}
    success = False
    for t in range(max_steps):
        state = env.sim.get_state().flatten()
        action = policy(obs)

        for k in obs_keys:
            traj["obs"][k].append(np.array(obs[k]))
        traj["states"].append(state)
        traj["actions"].append(action)

        obs, reward, done, info = env.step(action)
        traj["rewards"].append(reward)
        if env._check_success():
            success = True
            # retract upward so the ball settles cleanly in the right half;
            # record these frames for a clean trajectory ending
            for _ in range(12):
                state = env.sim.get_state().flatten()
                up = np.array([0, 0, 1.0, 0, 0, 0, 1.0])
                for k in obs_keys:
                    traj["obs"][k].append(np.array(obs[k]))
                traj["states"].append(state)
                traj["actions"].append(up)
                obs, reward, done, info = env.step(up)
                traj["rewards"].append(reward)
            break

    for k in obs_keys:
        traj["obs"][k] = np.stack(traj["obs"][k], 0)
    # robosuite/MuJoCo render camera images origin-at-bottom (upside down).
    # robomimic's EnvRobosuite flips them back to right-side-up at rollout/eval
    # time, so we store the SAME (flipped) convention here -- otherwise the
    # training images would be vertically mirrored relative to what the policy
    # sees during rollout, and relative to the real-robot dataset.
    for k in obs_keys:
        if k.endswith("_image"):
            traj["obs"][k] = traj["obs"][k][:, ::-1]
    traj["actions"] = np.stack(traj["actions"], 0)
    traj["states"] = np.stack(traj["states"], 0)
    traj["rewards"] = np.array(traj["rewards"])
    return traj, success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_demos", type=int, default=20)
    p.add_argument("--out", type=str, default=os.path.join(HERE, "demos", "luggage_push_demo.hdf5"))
    p.add_argument("--camera_h", type=int, default=96)
    p.add_argument("--camera_w", type=int, default=96)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--max_attempts", type=int, default=None)
    p.add_argument("--render_video", action="store_true")
    p.add_argument("--video_demos", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--table_style", type=str, default="ceramic", choices=["ceramic", "wood"])
    p.add_argument("--luggage_style", type=str, default="black", choices=["black", "gray"])
    args = p.parse_args()

    import h5py

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    env, ctrl_cfg = make_env(args.camera_h, args.camera_w, args.max_steps + 40, seed=args.seed,
                             table_style=args.table_style, luggage_style=args.luggage_style)
    policy = PushPolicy(table_z=float(env.table_offset[2]))

    env_args = {
        "env_name": "LuggagePush",
        "env_version": suite.__version__,
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
            "camera_heights": args.camera_h,
            "camera_widths": args.camera_w,
            "render_gpu_device_id": 0,
            "controller_configs": ctrl_cfg,
            "reward_shaping": False,
            "camera_names": ["agentview"],
            "table_style": args.table_style,
            "luggage_style": args.luggage_style,
        },
    }

    f = h5py.File(args.out, "w")
    grp = f.create_group("data")
    grp.attrs["env_args"] = json.dumps(env_args)

    max_attempts = args.max_attempts if args.max_attempts is not None else args.n_demos * 4
    n_collected, total_samples, attempts = 0, 0, 0
    video_frames = []

    while n_collected < args.n_demos and attempts < max_attempts:
        attempts += 1
        traj, success = rollout(env, policy, args.max_steps)
        if not success:
            print(f"  attempt {attempts}: FAILED (len {len(traj['actions'])})")
            continue

        ep = grp.create_group(f"demo_{n_collected}")
        ep.attrs["num_samples"] = traj["actions"].shape[0]
        ep.attrs["model_file"] = env.sim.model.get_xml()
        ep.create_dataset("actions", data=traj["actions"].astype(np.float64))
        ep.create_dataset("states", data=traj["states"].astype(np.float64))
        ep.create_dataset("rewards", data=traj["rewards"].astype(np.float64))
        obs_grp = ep.create_group("obs")
        for k, v in traj["obs"].items():
            if v.dtype == object:
                continue
            # Store images contiguous & uncompressed (matches the stack/mug
            # datasets). gzip with h5py's auto-chunking produces a pathological
            # layout (e.g. (46, 24, 24, 1)) that tiles each frame into dozens of
            # tiny chunks, so every random sequence-window read has to inflate
            # ~48 chunks -> ~13x slower dataloading, which dominates training.
            obs_grp.create_dataset(k, data=v)

        if args.render_video and n_collected < args.video_demos:
            # obs images are already stored right-side-up (see rollout()).
            video_frames.append(traj["obs"]["agentview_image"])

        total_samples += traj["actions"].shape[0]
        n_collected += 1
        print(f"  demo_{n_collected - 1}: SUCCESS  ({traj['actions'].shape[0]} steps)")

    grp.attrs["total"] = total_samples
    f.close()
    print(f"\nWrote {n_collected} demos ({total_samples} steps) to {args.out}")
    print(f"Success rate: {n_collected}/{attempts}")

    if args.render_video and video_frames:
        import imageio
        base = os.path.splitext(args.out)[0] + "_preview"
        try:
            writer = imageio.get_writer(base + ".mp4", fps=20)
            vid_path = base + ".mp4"
        except Exception:
            writer = imageio.get_writer(base + ".gif", fps=20)
            vid_path = base + ".gif"
        with writer as w:
            for frames in video_frames:
                for fr in frames:
                    w.append_data(fr)
                for _ in range(10):  # short pause between demos
                    w.append_data(frames[-1])
        print(f"Wrote preview video to {vid_path}")


if __name__ == "__main__":
    main()
