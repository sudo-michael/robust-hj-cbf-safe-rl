import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import redexp
from redexp.wrapper.safety_filter import (
    HJCBFSafetyFilter,
    LeastRestrictiveControlSafetyFilter,
)
import numpy as np

env = gym.make("Dubins3d-Deepreach-v1", render_mode='rgb_array')
env = RecordVideo(
    env,
    video_folder="dubins_3d_deepreach",
    name_prefix="test",
    episode_trigger=lambda x: True,
)

env = LeastRestrictiveControlSafetyFilter(env, 'deepreach', 1.0)
# env = HJCBFSafetyFilter(env, 'deepreach', cbf_gamma=1.0, cbf_init_max_slack=0.1)
state, info = env.reset()
done = False
iter = 0
total_reward = 0
while not done:
    action = env.action_space.sample()
    if iter % 2 == 0:
        action = env.action_space.high - 0.05
    else:
        action = env.action_space.low
    iter += 1
    next_state, reward, truncated, terminated, info = env.step(action)
    total_reward += reward
    state = next_state

    print('info')
    print(info)

    if truncated or terminated:
        done = True

    # env.render()
env.close()
