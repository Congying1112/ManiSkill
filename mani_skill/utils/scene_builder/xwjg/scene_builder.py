import os.path as osp
from pathlib import Path

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


class XWJGSceneBuilder(SceneBuilder):
    def build(self, build_config_idxs: list[int] = None):

        model_dir = Path(osp.dirname(__file__)) / "assets"

        workbench_loader = self.scene.create_mjcf_loader()
        workbench_loader.name = "workbench"
        workbench_builder = workbench_loader.parse(str(model_dir / "desk" / "desk.xml"))["actor_builders"][0]
        workbench_builder.initial_pose = sapien.Pose(
            p=[0, 0, 0], q=euler2quat(0, 0, 0)
        )
        self.desk = workbench_builder.build("desk")
        # assert self.desk is not None, f"Fail to load URDF/MJCF from {str(model_dir / "desk" / "desk.xml")}"

        pan_loader = self.scene.create_mjcf_loader()
        pan_loader.name = "pan"
        pan_builder = pan_loader.parse(str(model_dir / "pan" / "pan.xml"))["actor_builders"][0]
        pan_builder.initial_pose = sapien.Pose(
            p=[0, 0, 0], q=euler2quat(0, 0, 0)
        )
        self.pan = pan_builder.build("pan")
        # assert self.pan is not None, f"Fail to load URDF/MJCF from {str(model_dir / "pan" / "pan.xml")}"


        # ground - a strip with length along +x
        self.ground = build_ground(
            self.scene,
            floor_width=400,
            floor_length=400,
            altitude=-1,
            xy_origin=(0, 0),
        )
        self.scene_objects: list[sapien.Entity] = [self.ground, self.desk, self.pan]

    
    def _generate_qpos(self, robot_uid, env_idx: torch.Tensor):
        b = len(env_idx)
        match robot_uid: 
            case "panda" | "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                if self._enhanced_determinism:
                    qpos = (
                        self._batched_episode_rng[env_idx].normal(
                            0, self.robot_init_qpos_noise, len(qpos)
                        )
                        + qpos
                    )
                else:
                    qpos = (
                        self._episode_rng.normal(
                            0, self.robot_init_qpos_noise, (b, len(qpos))
                        )
                        + qpos
                    )
                qpos[:, -2:] = 0.04
                return qpos
            case "so100":
                qpos = np.array([0, 0, 0, np.pi / 2, np.pi / 2, 0])
                qpos = (
                    self._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
                return qpos
            case "fetch":
                return np.array(
                    [
                        0,
                        0,
                        0,
                        0.386,
                        0,
                        0,
                        0,
                        -np.pi / 4,
                        0,
                        np.pi / 4,
                        0,
                        np.pi / 3,
                        0,
                        0.015,
                        0.015,
                    ]
                )
            case _:
                return np.array([])

    def initialize(self, env_idx: torch.Tensor):
        b = len(env_idx)
        if self.env.robot_uids == "panda_wristcam":
            qpos = self._generate_qpos("panda_wristcam", env_idx)
            self.env.agent.reset(qpos)
            # self.env.env.agent.robot.set_root_pose(sapien.Pose([0.615, 0, 0]))
            # agent_root_pose = self.env.env.agent.robot.get_root_pose()
            # agent_p = agent_root_pose.get_p()
            # agent_q = agent_root_pose.get_q()
            # agent_p[:, :2] += (
            #     torch.rand((b, 2))
            # )
            # agent_q[:, :2] += (
            #     torch.rand((b, 2))
            # )
            
            # print("value to set p", agent_p)

            # self.env.agent.robot.root_pose.p = agent_p
            # print("after set: ", self.env.agent.robot.root_pose.p)
            # import time
            # time.sleep(0.1)
            # print("after delay: ", self.env.agent.robot.root_pose.p)

            # self.env.agent.robot.root_pose.q = agent_q
            # print("after random: ", self.env.agent.robot.get_root_pose())
        elif self.env.robot_uids == "fetch":
            qpos = self._generate_qpos("fetch", env_idx)
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -0.9]))
            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = self._generate_qpos("panda", env_idx)
            agent.env.agents[1].reset(qpos)
            agent.env.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.env.agents[0].reset(qpos)
            agent.env.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == ("panda_wristcam", "panda_wristcam"):
            agent: MultiAgent = self.env.agent
            qpos = self._generate_qpos("panda_wristcam", env_idx)

            agent.env.agents[1].reset(qpos)
            agent.env.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.env.agents[0].reset(qpos)
            agent.env.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == "so100":
            qpos = self._generate_qpos("so100", env_idx)
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )


class XWJGSceneBuilder1(SceneBuilder):
    """A simple scene builder that adds a table to the scene such that the height of the table is at 0, and
    gives reasonable initial poses for robots."""

    def build(self):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"
        # table_model_file = str(model_dir / "table.glb")
        scale = 1.

        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        # builder.add_nonconvex_collision_from_file(
        #     filename=table_model_file,
        #     scale=[scale] * 3,
        #     pose=table_pose,
        # )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
        )
        # builder.add_visual_from_file(
        #     filename=table_model_file, scale=[scale] * 3, pose=table_pose
        # )
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )
        # table = builder.build_kinematic(name="table-workspace")
        # aabb = (
        #     table._objs[0]
        #     .find_component_by_type(sapien.render.RenderBodyComponent)
        #     .compute_global_aabb_tight()
        # )
        # value of the call above is saved below
        aabb = np.array(
            [
                [-0.7402168, -1.2148621, -0.91964257],
                [0.4688596, 1.2030163, 3.5762787e-07],
            ]
        )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        # self.table = table
        self.scene_objects: list[sapien.Entity] = [self.ground]
        # self.scene_objects: list[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        # table_height = 0.9196429
        b = len(env_idx)
        # self.table.set_pose(
        #     sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        # )
        if self.env.robot_uids == "panda_wristcam":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in [
            "xarm6_allegro_left",
            "xarm6_allegro_right",
            "xarm6_robotiq",
            "xarm6_nogripper",
        ]:
            qpos = self.env.agent.keyframes["rest"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.522, 0, 0]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == ("panda_wristcam", "panda_wristcam"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif (
            "dclaw" in self.env.robot_uids
            or "allegro" in self.env.robot_uids
            or "trifinger" in self.env.robot_uids
        ):
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass
        elif self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in ["widowxai", "widowxai_wristcam"]:
            qpos = self.env.agent.keyframes["ready_to_grasp"].qpos
            self.env.agent.reset(qpos)
        elif self.env.robot_uids == "so100":
            qpos = np.array([0, 0, 0, np.pi / 2, np.pi / 2, 0])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )
