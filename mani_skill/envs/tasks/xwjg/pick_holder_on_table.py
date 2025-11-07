from typing import Any, Union, Tuple
import os.path as osp
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat, quat2euler
from pytorch3d.transforms import quaternion_to_axis_angle, matrix_to_euler_angles, quaternion_to_matrix

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, PandaWristCam, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


quat2euler_for_2dtensor = lambda src : torch.tensor([quat2euler(q) for q in src])

def batch_quat2euler_tensor(quat_tensor):
    """批量转换四元数Tensor到欧拉角Tensor"""
    quat_np = quat_tensor.numpy()
    
    # 使用apply_along_axis进行批量处理
    euler_np = np.apply_along_axis(
        lambda q: quat2euler(q), 
        axis=1, 
        arr=quat_np
    )
    
    return torch.tensor(euler_np, dtype=quat_tensor.dtype)

def normAngle_float(a):
    if a <= -np.pi:
        a += 2 * np.pi
        return normAngle(a)
    elif a >= np.pi:
        a -= 2 * np.pi
        return normAngle(a)
    else:
        return a
    
def normAngle(tensor):
    """
    将张量限制在[-pi, pi]区间
    使用模运算实现，支持GPU计算
    """
    # 使用模运算将角度映射到[0, 2pi)区间
    tensor_mod = torch.remainder(tensor, 2 * torch.pi)
    # 将大于pi的值减去2pi，映射到[-pi, pi]区间
    return  torch.where(tensor_mod > torch.pi, tensor_mod - 2 * torch.pi, tensor_mod)

PICK_HOLDER_ON_TABLE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""

TASK_CONFIG = {
    "target_half_size": 0.02,
    "goal_theta_thresh": 0.06,
    "goal_thresh": 0.025,
    "target_spawn_half_size": 0.1,
    "target_spawn_center": (0, 0),
    "max_goal_height": 0.3,
    "pickup_height": 0.3,
}

ROBOT_CONFIGS = {
    "panda_wristcam": {
        "agent_pos": [-0.2, -0.6, 0],
        "sensor_cam_eye_pos": [
            0.3,
            0,
            0.6,
        ],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [
            0.6,
            0.7,
            0.6,
        ],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "fetch": {
        "agent_pos": [-0.2, -0.6, 0],
        "holder_half_size": 0.1,
        "goal_thresh": 0.025,
        "holder_spawn_half_size": 0.1,
        "holder_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "pickup_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
}


@register_env("PickHolderOnTable-v1", max_episode_steps=150)
class PickHolderOnTableEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickHolderOnTable-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda_wristcam",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[PandaWristCam, Fetch, XArm6Robotiq, SO100, WidowXAI]
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.target_half_size = TASK_CONFIG["target_half_size"]
        self.goal_thresh = TASK_CONFIG["goal_thresh"]
        self.goal_theta_thresh = TASK_CONFIG["goal_theta_thresh"]
        self.target_spawn_half_size = TASK_CONFIG["target_spawn_half_size"]
        self.target_spawn_center = TASK_CONFIG["target_spawn_center"]
        self.max_goal_height = TASK_CONFIG["max_goal_height"]

        robot_cfg = ROBOT_CONFIGS[robot_uids] if robot_uids in ROBOT_CONFIGS else ROBOT_CONFIGS["panda_wristcam"]
        self.sensor_cam_eye_pos = robot_cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = robot_cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = robot_cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = robot_cfg["human_cam_target_pos"]
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
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    
    def create_anchor_system(self):
        """创建完整的锚点系统"""
        # 方法1: 纯计算锚点
        self.anchor_positions = []  # 用于记录轨迹等

    def get_anchor_pose1(self):
        """获取锚点的世界坐标 - 正确方法"""
        holder_pose = self.holder.pose

        # 定义锚点相对于holder的局部位姿
        anchor_local_position = np.array([-0.04, 0, 0])
        
        # 定义锚点相对于holder的局部旋转
        from scipy.spatial.transform import Rotation as R
        local_rotation = R.from_euler('z', 90, degrees=True).as_quat()

        # 创建锚点的局部变换矩阵
        anchor_local_matrix = np.eye(4)
        anchor_local_matrix[:3, 3] = anchor_local_position  # 设置平移部分
        anchor_local_matrix[:3, :3] = R.from_quat(local_rotation).as_matrix()  # 设置旋转部分

        # 获取holder的变换矩阵
        holder_matrix = holder_pose.to_transformation_matrix()
        # 计算锚点的世界变换矩阵
        anchor_world_matrix = holder_matrix @ anchor_local_matrix
        # 从变换矩阵提取位姿
        anchor_position = anchor_world_matrix[:3, 3]
        anchor_rotation = sapien.Pose.from_transformation_matrix(anchor_world_matrix).q
        
        return sapien.Pose(anchor_position, anchor_rotation)
    
        # 将局部偏移转换到世界坐标系
        anchor_world_homogeneous = holder_matrix @ np.array([-0.04, 0, 0, 1])
        anchor_world = anchor_world_homogeneous[:, :3]  # 提取前三个分量（x, y, z）
        
        return anchor_world
    
    def get_anchor_pose(self):
        """获取锚点的世界坐标 - 正确方法"""
        holder_pose = self.holder.pose

        # 定义锚点相对于holder的xiangdui位姿
        relative_pose = sapien.Pose(p=[-0.04, 0, 0], q=euler2quat(np.pi, 0, 0))
        
        return holder_pose * relative_pose
    
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.target_half_size,
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0, 0, self.target_half_size]),
        # )
        


        xyz = torch.zeros((3))

        xyz[:2] = (
            torch.rand((2)) * self.target_spawn_half_size * 2
            - self.target_spawn_half_size
        )
        xyz[0] += self.target_spawn_center[0]
        xyz[1] += self.target_spawn_center[1]
        xyz[2] = self.target_half_size
        qs = randomization.random_quaternions(1, lock_x=True, lock_y=True)

        model_dir = Path(osp.dirname(__file__)) / "assets" / "MJCF"
        self.holder = self.scene.create_mjcf_loader().parse(
            str(model_dir / "stick_holder" / "stick_holder.xml")
        )["actor_builders"][0].set_initial_pose(
            Pose.create_from_pq(xyz, qs)
        ).build_dynamic("holder")

        # self.grap_site = actors.build_sphere(
        #     self.scene,
        #     radius=0.04,
        #     color=[1, 0, 0, 1],
        #     name="grap_site",
        #     body_type="kinematic",
        #     add_collision=False,
        #     initial_pose=sapien.Pose(),
        # )
        # self.grap_site.set_parent(self.holder, sapien.Pose([0.1, 0, 0]))
        # self._hidden_objects.append(self.grap_site)


        
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
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))

            xyz[:, :2] = (
                torch.rand((b, 2)) * self.target_spawn_half_size * 2
                - self.target_spawn_half_size
            )
            xyz[:, 0] += self.target_spawn_center[0]
            xyz[:, 1] += self.target_spawn_center[1]
            xyz[:, 2] = self.target_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.holder.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.target_spawn_half_size * 2
                - self.target_spawn_half_size
            )
            goal_xyz[:, 0] += self.target_spawn_center[0]
            goal_xyz[:, 1] += self.target_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            obj_pose=self.holder.pose.raw_pose,
            goal_pos=self.goal_site.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.holder.pose.raw_pose,
                tcp_to_obj_pos=self.holder.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.holder.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.holder.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.holder)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }
    
    def print_pose(self, pose: sapien.Pose):
        euler_angles = quat2euler(pose.q[0])
        # print("Euler angles (radians):", euler_angles)
        # print("Euler angles (degrees):", np.degrees(euler_angles))
        return pose, np.degrees(euler_angles)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        info["reward"] = dict()

        # 夹爪位姿与锚点位姿对齐奖励
        anchor_pose = self.get_anchor_pose()
        tcp_relative_pose = self.agent.tcp_pose * anchor_pose.inv()
        # print("=========================")
        # print("holder_pose", self.print_pose(self.holder.pose))
        # print("anchor_pose", self.print_pose(anchor_pose))
        # print("tcp_pose", self.print_pose(self.agent.tcp_pose))
        # print("delta_pose", self.print_pose(tcp_relative_pose))
        tcp_to_anchor_dist = torch.linalg.norm(tcp_relative_pose.p, axis=1)
        reach_to_anchor_reward = 1 - torch.tanh(5 * tcp_to_anchor_dist)
        info["reward"]["reach_to_anchor_reward"] = reach_to_anchor_reward
        grapper_reward = 1 - torch.tanh(5 * torch.linalg.norm(matrix_to_euler_angles(quaternion_to_matrix(tcp_relative_pose.q), "XYZ")))
        info["reward"]["grapper_reward"] = grapper_reward
        # print("2", reach_to_anchor_reward, grapper_reward)
        reward = reach_to_anchor_reward + grapper_reward



        # tcp_to_anchor_dist = torch.linalg.norm(
        #     anchor_pose.p - self.agent.tcp_pose.p, axis=1
        # )
        # reach_to_anchor_reward = 1 - torch.tanh(5 * tcp_to_anchor_dist)
        # info["reward"]["reach_to_anchor_reward"] = reach_to_anchor_reward
        # print("delta_angle",
        #     matrix_to_euler_angles(quaternion_to_matrix(self.agent.tcp_pose.raw_pose[:, 3:]), "XYZ"),
        #     matrix_to_euler_angles(quaternion_to_matrix(anchor_pose.raw_pose[:, 3:]), "XYZ"),
        #     matrix_to_euler_angles(quaternion_to_matrix(self.agent.tcp_pose.raw_pose[:, 3:]), "XYZ")-
        #     matrix_to_euler_angles(quaternion_to_matrix(anchor_pose.raw_pose[:, 3:]), "XYZ"))
    
        # print("normed_delta_angle", normAngle(
        #     matrix_to_euler_angles(quaternion_to_matrix(self.agent.tcp_pose.raw_pose[:, 3:]), "XYZ")-
        #     matrix_to_euler_angles(quaternion_to_matrix(anchor_pose.raw_pose[:, 3:]), "XYZ")))
        
        # print("normvalue_nromed_delta_angle", torch.linalg.norm(normAngle(
        #     matrix_to_euler_angles(quaternion_to_matrix(self.agent.tcp_pose.raw_pose[:, 3:]), "XYZ")-
        #     matrix_to_euler_angles(quaternion_to_matrix(anchor_pose.raw_pose[:, 3:]), "XYZ"))))
        # delta_angles = normAngle(
        #     matrix_to_euler_angles(quaternion_to_matrix(self.agent.tcp_pose.raw_pose[:, 3:]), "XYZ")
        #     - matrix_to_euler_angles(quaternion_to_matrix(anchor_pose.raw_pose[:, 3:]), "XYZ")
        # )
        # print("delta_angles", delta_angles)
        # grapper_rotation = torch.linalg.norm(delta_angles, dim=1)
        # grapper_reward = 1 - torch.tanh(5 * grapper_rotation)
        # info["reward"]["grapper_reward"] = grapper_reward
        # reward = reach_to_anchor_reward + grapper_reward

        # 物体抓取奖励
        is_grasped = info["is_grasped"]
        info["reward"]["is_grasped"] = is_grasped
        reward += is_grasped

        # 物体放置奖励
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.holder.pose.p, axis=1
        )
        reach_to_goal_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        info["reward"]["reach_to_goal_reward"] = reach_to_goal_reward * is_grasped
        reward += reach_to_goal_reward * is_grasped

        # 放置到位后机器人静止奖励
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda_wristcam", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]
        # print("static_reward", static_reward, "is_obj_placed", info["is_obj_placed"])
        # print("reward1", reward)

        reward[info["success"]] = 10
        info["reward"]["reward"] = reward
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


PickHolderOnTableEnv.__doc__ = PICK_HOLDER_ON_TABLE_DOC_STRING.format(robot_id="PandaWristCam")
