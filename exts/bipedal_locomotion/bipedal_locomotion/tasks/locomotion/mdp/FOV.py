from __future__ import annotations
import torch
from typing import TYPE_CHECKING

# 引入 USD 核心库
# [修改] 添加 Sdf 导入，用于定义属性类型
from pxr import Gf, UsdGeom, Sdf
import omni.usd

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_apply, quat_inv, quat_mul, quat_from_euler_xyz

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def visualize_fov_and_check_ball(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    camera_offset_pos: tuple = (0.2, 0.0, 0.6), 
    camera_offset_rot: tuple = (0.0, 0.0, 0.0), 
    hfov_deg: float = 87.0, 
    vfov_deg: float = 58.0, 
    max_dist: float = 3.0,  
):
    # 1. 获取数据
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    device = robot.device
    num_envs = len(env_ids)

    # 获取 Robot Base 的位置和姿态
    robot_pos = robot.data.root_pos_w[env_ids]  
    robot_quat = robot.data.root_quat_w[env_ids] 

    # 获取 Ball 的位置
    ball_pos = ball.data.root_pos_w[env_ids]    

    # 2. 计算相机的 World Pose
    cam_offset_pos_tensor = torch.tensor(camera_offset_pos, device=device).repeat(num_envs, 1)
    
    q_rot = quat_from_euler_xyz(
        torch.tensor(camera_offset_rot[0], device=device),
        torch.tensor(camera_offset_rot[1], device=device),
        torch.tensor(camera_offset_rot[2], device=device)
    )
    cam_offset_rot_tensor = q_rot.unsqueeze(0).repeat(num_envs, 1)

    cam_pos_w = robot_pos + quat_apply(robot_quat, cam_offset_pos_tensor)
    cam_quat_w = quat_mul(robot_quat, cam_offset_rot_tensor)

    # 3. 将球的位置转换到相机坐标系
    vec_cam_to_ball_w = ball_pos - cam_pos_w
    vec_cam_to_ball_local = quat_apply(quat_inv(cam_quat_w), vec_cam_to_ball_w)

    x = vec_cam_to_ball_local[:, 0]
    y = vec_cam_to_ball_local[:, 1]
    z = vec_cam_to_ball_local[:, 2]

    # 4. 判断是否在 FOV 内
    tan_h = torch.tan(torch.tensor(hfov_deg / 2 * 3.14159 / 180.0, device=device))
    tan_v = torch.tan(torch.tensor(vfov_deg / 2 * 3.14159 / 180.0, device=device))

    is_in_front = x > 0
    is_within_dist = x < max_dist
    is_within_h = torch.abs(y) <= (x * tan_h)
    is_within_v = torch.abs(z) <= (x * tan_v)

    is_visible = is_in_front & is_within_dist & is_within_h & is_within_v
    
    if len(env_ids) > 0 and env_ids[0] == 0:
        print(f"[DEBUG] Env 0 | Dist: {x[0]:.2f} | X: {x[0]:.2f} Y: {y[0]:.2f} | Visible: {is_visible[0].item()}")

    # 5. 变色逻辑 (直接修改 USD 属性)
    stage = omni.usd.get_context().get_stage()
    
    # 限制循环次数，防止卡顿
    loop_count = min(num_envs, 10) 
    
    for i in range(loop_count):
        env_idx = env_ids[i].item()
        visible = is_visible[i].item()
        
        # 绿色 (可见) / 红色 (不可见)
        color = Gf.Vec3f(0.0, 1.0, 0.0) if visible else Gf.Vec3f(1.0, 0.0, 0.0)
        
        # 手动构造 Prim 路径
        prim_path = f"/World/envs/env_{env_idx}/Ball"
        
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            # [修改] 使用底层 API 直接操作属性，避开 Imageable 报错
            # 尝试获取显示颜色属性 (primvars:displayColor 是标准名称)
            color_attr = prim.GetAttribute("primvars:displayColor")
            
            # 如果属性不存在，则创建它
            if not color_attr.IsValid():
                color_attr = prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
            
            # 设置颜色# filepath: c:\Users\Zibo\Desktop\On-board\tron1-rl-isaaclab\exts\bipedal_locomotion\bipedal_locomotion\tasks\locomotion\mdp\FOV.py
