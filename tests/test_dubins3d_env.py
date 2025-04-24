import gymnasium as gym
import redexp
from redexp.wrapper.safety_filter import HJCBFSafetyFilter, LeastRestrictiveControlSafetyFilter
import numpy as np

env = gym.make("Safe-Dubins3d-BadModelMismatch-v1", render_mode='human')
env = LeastRestrictiveControlSafetyFilter(env, 0.05)
# env = HJCBFSafetyFilter(env, 'dubins_3d', cbf_gamma=2.0, cbf_init_max_slack=0.1)
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

    print(info)

    if truncated or terminated:
        done = True

    env.render()