import wandb
wandb.init(project="test")
import gymnasium as gym
import pandas as pd
env = gym.make('CartPole-v1')

df = pd.DataFrame()
for run in range(2):
    done = False

    obs, _ = env.reset()

    traj = [obs.tolist()]
    while not done:
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        obs = next_observation

        traj.append([action])
        traj.append(obs.tolist())
    print(traj)
    len_traj = len(traj)
    traj.extend((1_001 - len_traj) * [[-10000]])

    df[f'traj_{run}'] = traj

tb = wandb.Table(dataframe=df)

wandb.log({"traj": tb})

df = pd.DataFrame()
for run in range(2):
    done = False

    obs, _ = env.reset()

    traj = [obs.tolist()]
    while not done:
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        obs = next_observation

        traj.append([action])
        traj.append(obs.tolist())
    print(traj)
    len_traj = len(traj)
    traj.extend((401 - len_traj) * [[-10000]])

    df[f'traj_{run}'] = traj

tb = wandb.Table(dataframe=df)

wandb.log({"traj_step=2": tb})
