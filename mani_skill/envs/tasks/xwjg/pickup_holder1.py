from typing import Any, Union
import os.path as osp
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.xwjg import XWJGSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.xwjg.cfgs import XWJG_CONFIGS

PICKUP_HOLDER_DOC_STRING = """**Task Description:**
任务目标为用机器人{robot_id}抓起holder（一个小零件）并将其提起到一定高度。
任务开始时，holder静止放置在桌面上，机器人需要移动到holder旁边，抓取holder并将其提起到指定高度（pickup height）。
任务成功的条件为holder被牢固抓取且提升到目标高度以上，同时机器人保持静止
"""

@register_env("PickupHolder-v1", max_episode_steps=150)
class PickupHolderEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickupHolder-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda_wristcam",
        "fetch",
    ]
    agent: Union[PandaWristCam, Fetch]
    goal_thresh = 0.025
    holder_spawn_half_size = 0.05
    holder_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in XWJG_CONFIGS:
            cfg = XWJG_CONFIGS[robot_uids]
        else:
            cfg = XWJG_CONFIGS["panda_wristcam"]
        self.holder_half_size = cfg["holder_half_size"]
        self.pickup_height = cfg["pickup_height"]
        self.goal_thresh = cfg["goal_thresh"]
        self.holder_spawn_half_size = cfg["holder_spawn_half_size"]
        self.holder_spawn_center = cfg["holder_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        self.agent_pos = cfg["agent_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        positions = self.agent_pos
        # positions += np.random.rand(3) * 0.1  # 示例范围，调整z坐标需谨慎
        # positions[2] = 0  # 保持z坐标不变
        super()._load_agent(options, sapien.Pose(p=positions, q=euler2quat(0, 0, np.pi*3/4 + np.random.rand() * 0.1)))

    def _load_scene(self, options: dict):
        self.xwjg_scene = XWJGSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.xwjg_scene.build()

        model_dir = Path(osp.dirname(__file__)) / "assets" / "MJCF"

        self.holder = self.scene.create_mjcf_loader().parse(
            str(model_dir / "stick_holder" / "stick_holder.xml")
        )["actor_builders"][0].set_initial_pose(
            sapien.Pose(p=[-0.7, -0.5, 0], q=euler2quat(0, 0, 0))
        ).build_dynamic("holder")

        # self.holder = actors.build_cube(
        #     self.scene,
        #     half_size=self.holder_half_size,
        #     color=[1, 1, 0, 1],
        #     name="holder",
        #     initial_pose=sapien.Pose(p=[0, 0, self.holder_half_size]),
        # )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.xwjg_scene.initialize(env_idx)
            xyz = self.holder.pose.p
            # xyz[:, :2] += (
            #     torch.rand((b, 2)) * self.holder_spawn_half_size * 2
            #     - self.holder_spawn_half_size
            # )
            # xyz[:, 0] += self.holder_spawn_center[0]
            # xyz[:, 1] += self.holder_spawn_center[1]
            # xyz[:, 2] = self.holder_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.holder.set_pose(Pose.create_from_pq(xyz, qs))

    def _get_obs_extra(self, info: dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.holder.pose.raw_pose,
                tcp_to_obj_pos=self.holder.pose.p - self.agent.tcp_pose.p,
                gripper_width=self.agent.robot.get_gripper_width(),  # 夹爪宽度
                holder_velocity=self.holder.linear_velocity,  # holder速度
            )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.holder)
        is_pickup = self.holder.pose.p[:,2] > self.pickup_height
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_grasped & is_pickup & is_robot_static,
            "is_grasped": is_grasped,
            "is_pickup": is_pickup,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
         # 1. 改进接近奖励 - 使用更平滑的衰减
        tcp_to_obj_dist = torch.linalg.norm(
            self.holder.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 2.0 * torch.exp(-2.0 * tcp_to_obj_dist)  # 距离为0时奖励2
        reward = reaching_reward

        # 2. 添加朝向奖励 - 鼓励夹爪以正确朝向接近
        tcp_to_obj_vec = self.holder.pose.p - self.agent.tcp_pose.p
        tcp_forward = self.agent.tcp_pose.transform_vectors(
            torch.tensor([1, 0, 0], device=self.device).repeat(len(tcp_to_obj_vec), 1)
        )
        alignment = torch.sum(tcp_forward * tcp_to_obj_vec, dim=1) / (tcp_to_obj_dist + 1e-6)
        alignment_reward = 0.5 * torch.clamp(alignment, 0, 1)
        reward += alignment_reward

        # 3. 改进抓取奖励 - 连续信号
        is_grasped = info["is_grasped"]
        grasper_angle = self.agent.robot.grasper_angle()  # 需要获取夹爪宽度
        grasp_quality = torch.exp(-10 * grasper_angle)  # 夹爪闭合程度
        grasp_reward = 2.0 * is_grasped * grasp_quality
        reward += grasp_reward

        # 4. 改进提升奖励 - 只惩罚高度不足的情况
        height_diff = torch.clamp(self.pickup_height - self.holder.pose.p[:, 2], min=0)
        lift_reward = 2.0 * torch.exp(-5.0 * height_diff)
        reward += lift_reward * is_grasped

        # 5. 添加稳定性奖励
        if is_grasped:
            # 惩罚holder的角速度 - 鼓励稳定抓取
            obj_ang_vel = torch.linalg.norm(self.holder.angular_velocity, dim=1)
            stability_reward = 1.0 * torch.exp(-2.0 * obj_ang_vel)
            reward += stability_reward

        # 6. 改进成功奖励 - 渐进式而不是直接设置
        success_bonus = 3.0 * info["success"].float()  # 稍微降低成功奖励
        reward += success_bonus

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


PickupHolderEnv.__doc__ = PICKUP_HOLDER_DOC_STRING.format(robot_id="panda_wristcam")