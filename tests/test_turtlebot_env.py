import time
import jax
import numpy as np
import gymnasium as gym
import redexp
from redexp.wrapper.safety_filter import SafetyFilter

env_id = "TurtlebotEnv-NoModelMismatch-v1"
env = gym.make(env_id)
env = SafetyFilter(env, 0.1, turtle_bot=True, use_deepreach=False, use_deepreach_eps=False, use_cbf=False, gamma=0.0)


done = False
obs, info = env.reset()
r = 0
import rospy

while True:
    while not done:
        # action = env.action_space.sample()
        action = np.array([0.6, 0.0]) # v, omega
        next_state, reward, truncated, terminated, info = env.step(action)
        # print(f"{obs=} {next_state=} {reward=}")

        r += reward
        if info.get('cost'):
            print('in target set')
        if truncated or terminated:
            done = True


        # imagine doing training
        time.sleep(0.05)

    if done:
        obs, info = env.reset()
        done = False

        if rospy.is_shutdown():
            break