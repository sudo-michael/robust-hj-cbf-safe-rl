#! /usr/bin/env python
import gymnasium as gym
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents import SACLearner
from jaxrl5.data import ReplayBuffer
from jaxrl5.evaluation import evaluate
from jaxrl5.wrappers.single_precision import SinglePrecision
import redexp
from redexp.wrapper.record_costs import RecordEpisodeStatistics
FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "test", "wandb project name.")
flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
config_flags.DEFINE_config_file(
    "config",
    "sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
def main(_):
    wandb.init(project=FLAGS.project_name, monitor_gym=True)
    wandb.config.update(FLAGS)

    from datetime import datetime
    now = datetime.now() 
    date_str = now.strftime("%m_%d_%Y_%H:%M:%S")

    def make_env(train=True):
        env = gym.make(FLAGS.env_name, render_mode="rgb_array")
        env = SinglePrecision(env)

        if train:
            env = RecordEpisodeStatistics(env, deque_size=1)
            if FLAGS.save_video:
                env = gym.wrappers.RecordVideo(env, f"videos/{FLAGS.env_name}_{date_str}")
        return env

    env = make_env()
    eval_env = make_env(train=False)


    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    # observation, _ = env.reset(seed=FLAGS.seed)
    observation, _ = env.reset()
    done = False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        mask = truncated
        done = terminated or truncated

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                costs=0.0,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, _ = env.reset()
            done = False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time", "c": "cost"}
                wandb.log({f"training/{decode[k]}": v}, step=i)
            

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                seed=FLAGS.seed + 42,
                step=i,
                path=f"models/{FLAGS.env_name}_{date_str}"
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)
    
    eval_info = evaluate(
        agent,
        eval_env,
        num_episodes=FLAGS.eval_episodes,
        seed=FLAGS.seed + 42,
        step=i,
        path=f"models/{FLAGS.env_name}_{date_str}"
    )
    for k, v in eval_info.items():
        wandb.log({f"evaluation/{k}": v}, step=i)
            

if __name__ == "__main__":
    app.run(main)
