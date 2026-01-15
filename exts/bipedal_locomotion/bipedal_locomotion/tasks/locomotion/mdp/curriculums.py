from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers import SceneEntityCfg


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    env.command_manager.num_envs
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)

def lin_vel_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    # rwd_threshold: float,
    time_step: float,
    max_lin_vel_x: tuple[float, float],
    max_lin_vel_y: tuple[float, float],
) -> torch.Tensor:
    """Curriculum that increases the velocity command ranges based on time/performance.

    This term increases the velocity command ranges linearly over time.
    """
    # obtain the command term
    term = env.command_manager.get_term(command_name)
    
    # initialize curriculum factor if not present
    if not hasattr(term, "curriculum_factor"):
        term.curriculum_factor = 0.0
        # save initial ranges
        import copy
        term.initial_ranges = copy.deepcopy(term.cfg.ranges)

    # update curriculum factor
    # In this simple implementation, we increment linearly based on time_step
    term.curriculum_factor += time_step
    term.curriculum_factor = min(term.curriculum_factor, 1.0)
    
    #调试
    if env.common_step_counter % 1000 == 0:
        print(f"[Curriculum] step: {env.common_step_counter}, curriculum_factor: {term.curriculum_factor:.3f}, "
              f"lin_vel_x: {term.cfg.ranges.lin_vel_x}, lin_vel_y: {term.cfg.ranges.lin_vel_y}")

    # interpolate ranges
    def interp(start, end, factor):
        return start + (end - start) * factor

    # update x velocity
    term.cfg.ranges.lin_vel_x = (
        interp(term.initial_ranges.lin_vel_x[0], max_lin_vel_x[0], term.curriculum_factor),
        interp(term.initial_ranges.lin_vel_x[1], max_lin_vel_x[1], term.curriculum_factor),
    )
    # update y velocity
    term.cfg.ranges.lin_vel_y = (
        interp(term.initial_ranges.lin_vel_y[0], max_lin_vel_y[0], term.curriculum_factor),
        interp(term.initial_ranges.lin_vel_y[1], max_lin_vel_y[1], term.curriculum_factor),
    )

    return torch.tensor(term.curriculum_factor, device=env.device)