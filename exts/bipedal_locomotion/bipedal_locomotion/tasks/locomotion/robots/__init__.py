import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.rsl_rl_ppo_cfg import SoleFootPPORunnerCfg

from bipedal_locomotion.tasks.locomotion.agents.rsl_rl_ppo_mlp_cfg import SFFlatPPORunnerMlpCfg, SFRoughPPORunnerMlpCfg, SFStairPPORunnerMlpCfg

from . import solefoot_env_cfg

##############################
# Create PPO runners for RSL-RL
##############################

sf_blind_flat_runner_cfg = SoleFootPPORunnerCfg()
sf_blind_flat_runner_cfg.experiment_name = "sf_blind_flat"

sf_blind_rough_runner_cfg = SoleFootPPORunnerCfg()
sf_blind_rough_runner_cfg.experiment_name = "sf_blind_rough"

sf_blind_stairs_runner_cfg = SoleFootPPORunnerCfg()
sf_blind_stairs_runner_cfg.experiment_name = "sf_blind_stairs"

sf_mlp_blind_flat_runner_cfg = SFFlatPPORunnerMlpCfg()
sf_mlp_blind_flat_runner_cfg.experiment_name = "sf_mlp_blind_flat"

sf_mlp_stair_runner_cfg = SFStairPPORunnerMlpCfg()
sf_mlp_stair_runner_cfg.experiment_name = "sf_mlp_stairs"

############################
# SF Blind Flat Environment
############################

gym.register(
    id="Isaac-SF-Blind-Flat-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-SF-Blind-Flat-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": sf_blind_flat_runner_cfg,
    },
)

#############################
# SF Blind Flat Environment v1
#############################

gym.register(
    id="Isaac-SF-Blind-Flat-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": sf_mlp_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-SF-Blind-Flat-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": sf_mlp_blind_flat_runner_cfg,
    },
)


#############################
# SF Blind Rough Environment
#############################

gym.register(
    id="Isaac-SF-Blind-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": sf_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-SF-Blind-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": sf_blind_rough_runner_cfg,
    },
)


#############################
# SF Blind Rough Environment v1
#############################

gym.register(
    id="Isaac-SF-Blind-Rough-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": sf_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-SF-Blind-Rough-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": sf_blind_rough_runner_cfg,
    },
)


##############################
# SF Blind Stair Environment
##############################

gym.register(
    id="Isaac-SF-Blind-Stairs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindStairEnvCfg,
        "rsl_rl_cfg_entry_point": sf_blind_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-SF-Blind-Stairs-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": sf_blind_stairs_runner_cfg,
    },
)


#############################
# SF Blind Stair Environment v1
#############################

gym.register(
    id="Isaac-SF-Blind-Stair-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindStairEnvCfg,
        "rsl_rl_cfg_entry_point": sf_mlp_stair_runner_cfg,
    },
)

gym.register(
    id="Isaac-SF-Blind-Stair-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": solefoot_env_cfg.SFBlindStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": sf_mlp_stair_runner_cfg,
    },
)