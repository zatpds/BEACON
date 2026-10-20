"""
LuggagePush -- robosuite environment that mirrors a real push task.

Scene
-----
An open hard-shell luggage (47 cm x 34 cm) lies flat on the table as a
*stationary* fixture.  A raised central ridge (~9.5 cm) splits it into a LEFT
and a RIGHT half.  The LEFT half contains the retractable-handle rails, so it is
not flat: a mini football (22 cm long, 14 cm diameter, prolate ellipsoid with
four colored panels) rests on / between the rails.  The robot must push the ball
to the ridge-side edge of the left half and drive it over the ridge into the
right half.

The luggage interior is black; the ball is purple / red / orange / green; and
the ball is simulated as a true ellipsoid so its rolling physics are accurate.

Importing this module registers the ``LuggagePush`` environment with robosuite,
so afterwards you can call ``robosuite.make("LuggagePush", ...)``.
"""

import os
from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import find_elements, xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
import xml.etree.ElementTree as ET

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


class FootballObject(MujocoXMLObject):
    """Mini football: prolate ellipsoid with four colored visual panels."""

    def __init__(self, name):
        super().__init__(
            os.path.join(ASSET_DIR, "football.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0001")],
            obj_type="all",
            # The colored panels are authored explicitly as visual geoms, so we
            # must NOT auto-duplicate the (dark) collision ellipsoid into a
            # visual geom that would cover them.
            duplicate_collision_geoms=False,
        )


class LuggageObject(MujocoXMLObject):
    """Open luggage fixture.  Loaded with no joints so it is welded to world."""

    def __init__(self, name):
        super().__init__(
            os.path.join(ASSET_DIR, "luggage.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class LuggagePush(ManipulationEnv):
    """
    Push a football over the central ridge of an open luggage.

    Args:
        robots: robot spec (single arm), e.g. "Panda".
        See robosuite.environments.manipulation.stack.Stack for the full list of
        shared keyword arguments -- this env exposes the same set.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        table_style="ceramic",
        luggage_style="black",
        base_below_table=0.02,
    ):
        # table
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))

        # mount the robot base this far below the table top (real-rig offset)
        self.base_below_table = base_below_table

        # appearance variants (for visual / domain diversity)
        self.table_style = table_style       # "ceramic" (default) or "wood"
        self.luggage_style = luggage_style   # "black" (default) or "gray"

        # reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        # geometry constants (luggage local frame, see luggage.xml)
        # The luggage opens left-to-right: the ridge runs along x at y = 0, the
        # ball starts in the LEFT half (y < 0) and is pushed across into the
        # RIGHT half (y > 0).
        self.ridge_y = 0.0           # ridge sits at luggage-center y
        self.ridge_top_z = 0.104     # height of ridge crest above luggage bottom
        self.floor_top_z = 0.02
        # shift the luggage toward the far (upper) table edge
        self.luggage_offset = np.array([-0.10, 0.0, 0.0])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    #  Reward
    def reward(self, action=None):
        """
        Sparse reward of 1.0 when the ball is pushed over the ridge into the
        right half.  When ``reward_shaping`` is on, a dense term encourages
        moving the ball toward (and past) the ridge.
        """
        success = self._check_success()

        if self.reward_shaping:
            ball_pos = self.sim.data.body_xpos[self.ball_body_id]
            # progress along +y relative to the ridge (in world coords)
            progress = ball_pos[1] - (self.table_offset[1] + self.ridge_y)
            # reach term: encourage gripper to get near the ball
            eef = min(
                np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - ball_pos)
                for arm in self.robots[0].arms
            )
            r_reach = (1 - np.tanh(10.0 * eef)) * 0.25
            r_push = 0.5 * (1 + np.tanh(8.0 * progress))
            reward = max(r_reach + r_push, 2.0 if success else 0.0)
        else:
            reward = 1.0 if success else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / (2.0 if self.reward_shaping else 1.0)
        return reward

    #  Model
    def _load_model(self):
        super()._load_model()

        # robot base: keep the default x/y stand-off but drop the base so it sits
        # `base_below_table` metres beneath the table top (matches the real rig).
        xpos = np.array(
            self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]),
            dtype=float,
        )
        base_z = self.table_offset[2] - self.base_below_table
        # set_base_xpos places the robot's *bottom* point; add the (negative)
        # bottom_offset so the base body lands exactly at `base_z`.
        xpos[2] = base_z + self.robots[0].robot_model.bottom_offset[2]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # optional wood tabletop (default ceramic)
        if self.table_style == "wood":
            self._apply_wood_table(mujoco_arena)

        # objects
        self.luggage = LuggageObject(name="luggage")
        # optional deep-gray luggage shell (default black interior)
        if self.luggage_style == "gray":
            self._apply_gray_luggage(self.luggage)
        # weld the (jointless) luggage flush on top of the table
        lug_pos = self.table_offset + self.luggage_offset
        self.luggage.get_obj().set("pos", "{} {} {}".format(*lug_pos))

        self.ball = FootballObject(name="football")

        # placement sampler: ball is dropped onto the LEFT half (y < 0), resting
        # on the two handle bumps with its long axis ~along x so it can be pushed
        # sideways (+y) over the ridge.
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.ball)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="BallSampler",
                mujoco_objects=self.ball,
                # spread across the reachable LEFT compartment: from the saddle
                # between the bumps through to between the right bump and the
                # ridge, widely varied along the bump length (x).  (Further left,
                # past the left bump, the ball rolls into the narrow well against
                # the wall where the arm cannot get behind it.)
                x_range=[-0.10, 0.10],
                y_range=[-0.17, -0.07],
                rotation=[-0.15, 0.15],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset + self.luggage_offset,
                z_offset=0.05,            # spawn just above the bumps, then settle
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.luggage, self.ball],
        )

    @staticmethod
    def _apply_wood_table(mujoco_arena):
        """Retexture the table top with a natural wood material."""
        tex = ET.Element(
            "texture",
            attrib={
                "file": xml_path_completion("textures/light-wood.png"),
                "type": "cube",
                "name": "tex-wood-table",
            },
        )
        mat = ET.Element(
            "material",
            attrib={
                "name": "table_wood",
                "texture": "tex-wood-table",
                "texrepeat": "2 2",
                "specular": "0.1",
                "shininess": "0.1",
                "reflectance": "0.0",
            },
        )
        mujoco_arena.asset.append(tex)
        mujoco_arena.asset.append(mat)
        mujoco_arena.table_visual.set("material", "table_wood")

    @staticmethod
    def _apply_gray_luggage(luggage):
        """Recolor the (black) luggage shell to a deep gray."""
        scheme = (
            ("wall", "0.36 0.36 0.39 1"),
            ("crest", "0.28 0.28 0.30 1"),
            ("ridge", "0.26 0.26 0.28 1"),
            ("bump", "0.20 0.20 0.22 1"),
            ("floor", "0.30 0.30 0.32 1"),
        )
        geoms = find_elements(luggage.worldbody, tags="geom", return_first=False) or []
        for g in geoms:
            name = g.get("name", "")
            for key, rgba in scheme:
                if key in name:
                    g.set("rgba", rgba)
                    break

    def _setup_references(self):
        super()._setup_references()
        self.ball_body_id = self.sim.model.body_name2id(self.ball.root_body)
        self.luggage_body_id = self.sim.model.body_name2id(self.luggage.root_body)

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

    #  Observables
    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def ball_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.ball_body_id])

            @sensor(modality=modality)
            def ball_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.ball_body_id]), to="xyzw")

            @sensor(modality=modality)
            def ridge_pos(obs_cache):
                # world position of the top-center of the ridge (the target line)
                return (
                    self.table_offset
                    + self.luggage_offset
                    + np.array([0.0, self.ridge_y, self.ridge_top_z])
                )

            @sensor(modality=modality)
            def ball_to_ridge(obs_cache):
                return (
                    obs_cache["ridge_pos"] - obs_cache["ball_pos"]
                    if "ridge_pos" in obs_cache and "ball_pos" in obs_cache
                    else np.zeros(3)
                )

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors = [ball_pos, ball_quat, ridge_pos, ball_to_ridge]
            sensors += [
                self._get_obj_eef_sensor(full_pf, "ball_pos", f"{arm_pf}gripper_to_ball", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    #  Success
    def _check_success(self):
        """Ball has crossed the ridge into the right half and settled."""
        ball_pos = self.sim.data.body_xpos[self.ball_body_id]
        crossed = ball_pos[1] > self.table_offset[1] + self.ridge_y + 0.05
        settled = ball_pos[2] < self.table_offset[2] + self.ridge_top_z + 0.02
        return bool(crossed and settled)

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.ball)
