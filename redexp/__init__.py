from gymnasium.envs.registration import register

register(
    id="Safe-Dubins3d-NoModelMismatch-v1",
    entry_point="redexp.envs:Dubins3dEnv",
    max_episode_steps=200,
    kwargs={"car": "dubins_3d_omega_0_5", "brt":"dubins_3d_omega_0_5"},
)

register(
    id="Safe-Dubins3d-BadModelMismatch-v1",
    entry_point="redexp.envs:Dubins3dEnv",
    max_episode_steps=200,
    kwargs={"car": "dubins_3d_omega_0_5", "brt":"dubins_3d_omega_0_75"},
)

register(
    id="Safe-Dubins3d-GoodModelMismatch-v1",
    entry_point="redexp.envs:Dubins3dEnv",
    max_episode_steps=200,
    kwargs={"car": "dubins_3d_omega_0_75", "brt":"dubins_3d_omega_0_5"},
)

register(
    id="TurtlebotEnv-NoModelMismatch-v1",
    entry_point="redexp.envs:TurtlebotEnv",
    max_episode_steps=1000,
    kwargs={'model_mismatch': False},
)

register(
    id="TurtlebotEnv-ModelMismatch-v1",
    entry_point="redexp.envs:TurtlebotEnv",
    max_episode_steps=1000,
    kwargs={'model_mismatch': True},
)

register(
    id="TurtlebotEnv-NoModelMismatch-GC-v1",
    entry_point="redexp.envs:TurtlebotEnv",
    max_episode_steps=1000,
    kwargs={'model_mismatch': False, 'goal_conditioned': True},
)

register(
    id="TurtlebotEnv-ModelMismatch-GC-v1",
    entry_point="redexp.envs:TurtlebotEnv",
    max_episode_steps=1000,
    kwargs={'model_mismatch': True, 'goal_conditioned': True},
)

register(
    id="ReachAvoidGame9D-v1",
    entry_point="redexp.envs:ReachAvoidGame9DEnv",
    max_episode_steps=400,
)

register(
    id="Dubins3d-Deepreach-v1",
    entry_point="redexp.envs:Dubins3dDeepreachEnv",
    max_episode_steps=400,
)