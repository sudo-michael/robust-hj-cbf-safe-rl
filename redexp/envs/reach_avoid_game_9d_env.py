import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

from redexp.config.dubins_3d import OBSTACLE_RADIUS
from redexp.utils import normalize_angle
from redexp.brts.dubins_3d import (
    dubins_3d_omega_0_5,
    grid,
)
from deepreach.deepreach_model import DeepreachModel


class ReachAvoidGame9DEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode=None,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.render_mode = render_mode

        # hardcoded "runs/1_evader_2_pursuer/"
        self.dr = DeepreachModel("models/1_evader_2_pursuer_5/", 150_000, "reach_avoid")

        self.evader = dubins_3d_omega_0_5
        self.evader.speed = 1.1
        self.evader.wMax = 1.1
        self.persuer_1 = dubins_3d_omega_0_5
        self.persuer_1.speed = 0.5
        self.persuer_1.wMax = 0.7
        self.persuer_2 = dubins_3d_omega_0_5
        self.persuer_2.speed = 0.5
        self.persuer_2.wMax = 0.7

        self.evader_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.persuer_1_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.persuer_2_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.dt = 0.05
        self.action_space = gym.spaces.Box(
            low=np.array([-self.evader.wMax], dtype=np.float32),
            high=np.array([self.evader.wMax], dtype=np.float32),
            dtype=np.float32,
        )

        # fmt: off
        self.observation_space = gym.spaces.Box(
            low=np.array([-4, -4, -1.0, -1.0, # evader
                          -4, -4, -1.0, -1.0, # persuer 1
                          -4, -4, -1.0, -1.0, # persuer 2
            ], dtype=np.float32),
            high=np.array([4, 4, 1.0, 1.0, 
                           4, 4, 1.0, 1.0, 
                           4, 4, 1.0, 1.0, 
            ], dtype=np.float32),
            dtype=np.float32,
        )
        # fmt: on

        self.goal_location = np.array([2.0, 2.0], dtype=np.float32)
        self.goal_r = 0.3
        self.obstacle_location = np.array([0.0, 0.0], dtype=np.float32)
        self.obstacle_r = OBSTACLE_RADIUS
        self.grid = grid

    def reset(self, seed=None, options={}):
        self.evader_state = np.array([-2.0, -2.0, np.pi/4], dtype=np.float32)
        self.persuer_1_state = np.array([-3, 0, -np.pi/2], dtype=np.float32)
        self.persuer_2_state = np.array([0, -3, -np.pi], dtype=np.float32)

        self.state = np.array([
            self.evader_state[0],
            self.evader_state[1],
            self.persuer_1_state[0],
            self.persuer_1_state[1],
            self.persuer_2_state[0],
            self.persuer_2_state[1],
            self.evader_state[2],
            self.persuer_2_state[2],
            self.persuer_2_state[2],
        ])

        if self.render_mode == "human":
            self._render_frame()

        value, opt_evader_ctrl, opt_persuer_1_ctrl, opt_persuer_2_ctrl = (
            self.dr.value_and_opt_ctrl(self.state
            )
        )

        self.traj_min_brt_value = value

        return self._get_obs(), {"cost": 0}

    def step(self, action):
        value, opt_evader_ctrl, opt_persuer_1_ctrl, opt_persuer_2_ctrl = (
            self.dr.value_and_opt_ctrl(
                np.concatenate(
                    [
                        self.evader_state[:2],
                        self.persuer_1_state[:2],
                        self.persuer_2_state[:2],
                        [self.evader_state[2]],
                        [self.persuer_1_state[2]],
                        [self.persuer_2_state[2]],
                    ]
                )
            )
        )
        self.evader_state = (
            self.evader.dynamics_non_hcl(0, self.evader_state, action)
            * self.dt
            + self.evader_state
        )
        self.evader_state[2] = normalize_angle(self.evader_state[2])

        self.persuer_1_state = (
            self.persuer_1.dynamics_non_hcl(0, self.persuer_1_state, opt_persuer_1_ctrl)
            * self.dt
            + self.persuer_1_state
        )
        self.persuer_1_state[2] = normalize_angle(self.persuer_1_state[2])

        self.persuer_2_state = (
            self.persuer_2.dynamics_non_hcl(0, self.persuer_2_state, opt_persuer_2_ctrl)
            * self.dt
            + self.persuer_2_state
        )
        self.persuer_2_state[2] = normalize_angle(self.persuer_2_state[2])

        self.state = np.array([
            self.evader_state[0],
            self.evader_state[1],
            self.persuer_1_state[0],
            self.persuer_1_state[1],
            self.persuer_2_state[0],
            self.persuer_2_state[1],
            self.evader_state[2],
            self.persuer_2_state[2],
            self.persuer_2_state[2],
        ])

        info = {}

        obs = self._get_obs()
        reward = -np.linalg.norm(self.evader_state[:2] - self.goal_location)

        terminated = False

        cost = np.linalg.norm(self.evader_state[:2] - self.obstacle_location) < (
            self.obstacle_r + self.evader.r
        )

        cost = cost or np.linalg.norm(
            self.evader_state[:2] - self.persuer_1_state[:2]
        ) < (self.persuer_1.r + self.evader.r)

        cost = cost or np.linalg.norm(
            self.evader_state[:2] - self.persuer_2_state[:2]
        ) < (self.persuer_2.r + self.evader.r)

        if cost > 0:
            terminated = True
        info["cost"] = cost

        if np.linalg.norm(self.evader_state[:2] - self.goal_location) < (
            self.goal_r + self.evader.r
        ):
            terminated = True
            info["reach_goal"] = True
        info["reach_goal"] = False

        self.traj_min_brt_value = min(
            self.traj_min_brt_value, value
        )
        info["traj_min_brt_value"] = self.traj_min_brt_value

        return obs, reward, terminated, False, info

    def _get_obs(self):
        return np.array(
            [
                self.evader_state[0],
                self.evader_state[1],
                np.cos(self.evader_state[2]),
                np.sin(self.evader_state[2]),
                self.persuer_1_state[0],
                self.persuer_1_state[1],
                np.cos(self.persuer_1_state[2]),
                np.sin(self.persuer_1_state[2]),
                self.persuer_2_state[0],
                self.persuer_2_state[1],
                np.cos(self.persuer_2_state[2]),
                np.sin(self.persuer_2_state[2]),
            ]
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        self._render_frame()
        self.fig.canvas.flush_events()
        plt.pause(1 / self.metadata["render_fps"])

    def _render_frame(self):
        self.ax.clear()

        def add_circle(state, r, color="green"):
            self.ax.add_patch(plt.Circle(state[:2], radius=r, color=color))

            dir = state[:2] + r * np.array([np.cos(state[2]), np.sin(state[2])])

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_circle(self.evader_state, self.evader.r, color="blue")
        add_circle(self.persuer_1_state, self.persuer_1.r, color="red")
        add_circle(self.persuer_2_state, self.persuer_2.r, color="red")
        goal = plt.Circle(self.goal_location[:2], radius=self.goal_r, color="g")
        self.ax.add_patch(goal)
        obstacle = plt.Circle(self.obstacle_location, radius=self.obstacle_r, color="r")
        self.ax.add_patch(obstacle)

        # X, Y = np.meshgrid(
        #     np.linspace(self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0]),
        #     np.linspace(self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1]),
        #     indexing="ij",
        # )

        # index = self.grid.get_index(self.state)

        # self.ax.contour(
        #     X,
        #     Y,
        #     self.true_brt[:, :, index[2]],
        #     # self.brt[:, :, index[2]],
        #     levels=[0.1],
        # )
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        return img

    def close(self):
        plt.close()
