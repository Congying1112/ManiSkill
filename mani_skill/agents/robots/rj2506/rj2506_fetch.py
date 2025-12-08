from copy import deepcopy
from typing import Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array
from transforms3d.euler import euler2quat, quat2euler

FETCH_WHEELS_COLLISION_BIT = 30
"""Collision bit of the fetch robot wheel links"""
FETCH_BASE_COLLISION_BIT = 31
"""Collision bit of the fetch base"""


@register_agent()
class RJ2506Fetch(BaseAgent):
    uid = "rj2506_fetch"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/RJ2506/urdf/RJ2506_leftarm_only.urdf"
    urdf_arm_ik_path = f"{PACKAGE_ASSET_DIR}/robots/RJ2506/urdf/RJ2506_leftarm_only.urdf"

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
            pose=sapien.Pose(),
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ),
        test=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array([0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 1.0, 0.2, 0.2, 0]),
        )
    )

    @property
    def _sensor_configs(self):
        return [
            # 头上：当前800万(3840*2160)110度，后续将升级为1280万（5120*2880）120度
            CameraConfig(
                uid="cam_head",
                pose=sapien.Pose(p=[0,0,0], q=euler2quat(0, 0, 0)),
                # pose=sapien.Pose(p=[0.3, 0.1, 0.1], q=euler2quat(0, 0.3, 0)),
                width=3840,
                height=2160,
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
                width=3840,
                height=2160,
                fov=110 * np.pi / 180,
                near=0.01,
                far=100,
                mount=self.robot.links_map["left_hand_front_cam"],
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

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "left_arm_joint0",
            "left_arm_joint1",
            "left_arm_joint2",
            "left_arm_joint3",
            "left_arm_joint4",
            "left_arm_joint5",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = [
            "left_hand_finger1_joint",
            "left_hand_finger2_joint",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "left_hand_tcp"

        self.body_joint_names = [
            "body_joint1",
            "body_joint2"
        ]
        self.body_stiffness = 1e3
        self.body_damping = 1e2
        self.body_force_limit = 100

        # self.base_joint_names = [
        #     "root_x_axis_joint",
        #     "root_y_axis_joint",
        #     "root_z_rotation_joint",
        # ]

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
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
            urdf_path=self.urdf_arm_ik_path,
            root_link_name="base_link",
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
            urdf_path=self.urdf_arm_ik_path,
            root_link_name="base_link",
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
            normalize_action=True,
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
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.01,  # a trick to have force when the object is thin
            0.05,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            mimic={"left_hand_finger2_joint": {"joint": "left_hand_finger1_joint"}},
        )

        # -------------------------------------------------------------------------- #
        # Body
        # -------------------------------------------------------------------------- #
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            -0.1,
            0.1,
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            use_delta=True,
        )

        # useful to keep body unmoving from passed position
        stiff_body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            None,
            None,
            1e5,
            1e5,
            1e5,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        # base_pd_joint_vel = PDBaseForwardVelControllerConfig(
        #     self.base_joint_names,
        #     lower=[-1, -3.14],
        #     upper=[1, 3.14],
        #     damping=1000,
        #     force_limit=500,
        # )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
                # balance_passive_force=False
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
                # balance_passive_force=False
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                # base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_stiff_body=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=stiff_body_pd_joint_pos,
                # base=base_pd_joint_vel,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.finger1_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_hand_finger1"
        )
        self.finger2_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_hand_finger2"
        )
        self.tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        # self.base_link: Link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), "base_link"
        # )
        # self.l_wheel_link: Link = self.robot.links_map["l_wheel_link"]
        # self.r_wheel_link: Link = self.robot.links_map["r_wheel_link"]
        # for link in [self.l_wheel_link, self.r_wheel_link]:
        #     link.set_collision_group_bit(
        #         group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
        #     )
        # self.base_link.set_collision_group_bit(
        #     group=2, bit_idx=FETCH_BASE_COLLISION_BIT, bit=1
        # )

        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )

        self.cam_head_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "cam_head"
        )

        self.queries: dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = -self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

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

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
