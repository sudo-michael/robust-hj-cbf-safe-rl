from redexp.wrapper.safety_filter import SafetyFilter
import gymnasium as gym
import time

# env = gym.make('Safe-StaticDubins3d-v0')
env = gym.make('TurtlebotEnv-v0')
env = SafetyFilter(env, 0.05, use_deepreach=False, use_deepreach_eps=0.5, use_cbf=True, gamma=1.0, turtle_bot=True)

obs, _ = env.reset()
done = False

r = 0
i=0
import numpy as np
while not done:
    action = env.action_space.sample()
    if i % 2==0:
        action = np.array([-1.1])
    else:
        action = np.array([1.1])
    i += 1
    next_state, reward, truncated, terminated, info = env.step(action)
    r += reward
    if info.get('cost'):
        print('in target set')
        
    print(env.unwrapped.state, info['value'])
    # input()

    if truncated or  terminated:
        done = True

    print('end step')
    time.sleep(0.1)
    # env.render()

# obs, _ = env.reset()
# done = False
# steps = 0
# tik = time.time()
# TOTAL_STEPS = 10_000
# while steps < TOTAL_STEPS:
#     while not done:
#         action = env.action_space.sample()
#         next_state, reward, truncated, terminated, info = env.step(action)
#         steps += 1
#         if truncated or  terminated:
#             done = True

#     obs, _ = env.reset()
#     done = False

# print(f"{TOTAL_STEPS / (time.time() - tik)} steps per second")