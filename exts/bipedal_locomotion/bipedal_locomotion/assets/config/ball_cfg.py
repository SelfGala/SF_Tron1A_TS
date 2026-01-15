import os
import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object import RigidObjectCfg

BALL_CFG = RigidObjectCfg(
    spawn=sim_utils.SphereCfg(
        radius=0.03,  # 网球半径约3cm
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.05,
            angular_damping=0.05,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.7,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{isaaclab.utils.assets.ISAACLAB_NUCLEUS_DIR}/Materials/Base/Metallic/Chrome.mdl",
            project_uvw=True,
        ),
        mass=0.057,  # 网球质量约57g
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)