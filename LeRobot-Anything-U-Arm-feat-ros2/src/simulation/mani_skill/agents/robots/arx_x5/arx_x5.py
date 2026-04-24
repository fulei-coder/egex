import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent


@register_agent()
class ArxX5(BaseAgent):
    uid = "arx-x5"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/ARX-X5/X5A.urdf"

    # 关节分组（手臂 6 自由度 + 夹爪 2 自由度）
    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    gripper_joint_names = ["joint7", "joint8"]

    # 关键帧（最小：全零）
    keyframes = dict(
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    # 控制器配置（最小：关节位置与关节增量位置）
    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=100,
            damping=10,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=100,
            damping=10,
            use_delta=True,
        )

        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            lower=None,
            upper=None,
            stiffness=100,
            damping=10,
            normalize_action=False,
        )
        gripper_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=100,
            damping=10,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper_active=gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper_active=gripper_pd_joint_delta_pos,
            ),
        )

        return deepcopy_dict(controller_configs)
