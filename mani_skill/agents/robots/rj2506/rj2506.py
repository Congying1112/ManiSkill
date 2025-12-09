from copy import deepcopy

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig
from transforms3d.euler import euler2quat, quat2euler


@register_agent()
class RJ2506(BaseAgent):
    uid = "rj2506"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/RJ2506/urdf/RJ2506_leftarm_only_noagv.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_hand_finger1=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_finger2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, 0, 0.27, 0.085, 0, 1.6, -1.6, -0.45, 0, 0]),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "body_joint1",
        "body_joint2",
        "left_arm_joint0",
        "left_arm_joint1",
        "left_arm_joint2",
        "left_arm_joint3",
        "left_arm_joint4",
        "left_arm_joint5",
    ]
    gripper_joint_names = [
        "left_hand_finger1_joint",
        "left_hand_finger2_joint",
    ]
    ee_link_name = "left_hand_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100


    @property
    def _sensor_configs(self):
        return [
            # 头上：当前800万(3840*2160)110度，后续将升级为1280万（5120*2880）120度
            CameraConfig(
                uid="cam_head",
                pose=sapien.Pose(p=[0,0,0], q=euler2quat(0, 0, 0)),
                # pose=sapien.Pose(p=[0.3, 0.1, 0.1], q=euler2quat(0, 0.3, 0)),
                # width=3840, height=2160,
                # width=256, height=256,
                width=128, height=128,
                fov= 110 * np.pi / 180,
                near=0.01,
                far=100,
                mount=self.robot.links_map["cam_head"],
            ),
            # 手腕上：800万(3840*2160)110度，后续换为800万（5120*2880）90度
            CameraConfig(
                uid="left_hand_front_cam",
                pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, 0)),
                # pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(1.57, 0, 0)),
                # width=3840, height=2160,
                # width=256, height=256,
                width=128, height=128,
                fov=110 * np.pi / 180,
                near=0.01,
                far=100,
                mount=self.robot.links_map["left_hand_front_cam"],
            ),
            CameraConfig(
                uid="left_hand_back_cam",
                pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, 0)),
                # pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(1.57, 0, 0)),
                # width=3840, height=2160,
                # width=256, height=256,
                width=128, height=128,
                fov=110 * np.pi / 180,
                near=0.01,
                far=100,
                mount=self.robot.links_map["left_hand_back_cam"],
            ),
            # CameraConfig(
            #     uid="right_hand_front_cam",
            #     pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, 0)),
            #     width=3840,
            #     height=2160,
            #     fov=110 * np.pi / 180,
            #     near=0.01,
            #     far=100,
            #     mount=self.robot.links_map["right_hand_front_cam"],
            # ),
        ]

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        # gripper_pd_joint_pos = PDJointPosControllerConfig(
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.1,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"left_hand_finger1_joint": {"joint": "left_hand_finger2_joint"}},
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.left_hand_finger1 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_hand_finger1"
        )
        self.left_hand_finger2 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_hand_finger2"
        )
        # self.finger1pad_link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), "panda_leftfinger_pad"
        # )
        # self.finger2pad_link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), "panda_rightfinger_pad"
        # )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.left_hand_finger1, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.left_hand_finger2, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.left_hand_finger1.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.left_hand_finger2.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def grasper_angle(self):

        # direction to open the gripper
        ldirection = self.left_hand_finger1.pose * self.tcp.pose.inv()
        rdirection = self.left_hand_finger2.pose * self.tcp.pose.inv()

        return torch.linalg.norm(ldirection.p, axis=1) + torch.linalg.norm(rdirection.p, axis=1)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
