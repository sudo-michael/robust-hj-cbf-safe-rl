from typing import Dict

import gymnasium as gym
import numpy as np


import redexp
from redexp.wrapper.record_costs import RecordEpisodeStatistics
from jaxrl5.agents.agent import save_agent


def evaluate(
    agent,
    env: gym.Env,
    num_episodes: int,
    seed: int,
    step: int,
    path,
) -> Dict[str, float]:
    # if save_video:
    #     env = WANDBVideo(env, name="eval_video", max_videos=1)
    save_agent(agent, path, step)
    env = RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, _ = env.reset(seed=seed)
        done = False
        while not done:
            action, agent = agent.eval_actions(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return {"return": np.mean(env.return_queue), "cost:": np.mean(env.cost_queue)}
