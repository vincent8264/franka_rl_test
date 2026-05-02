import numpy as np
import gymnasium as gym
from franka import FrankaEnv

class FrankaGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.env = FrankaEnv(render_mode=render_mode)

        obs = self.env._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )
        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        self.current_step = 0
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space {self.action_space}")

        controls = self._denormalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(controls)

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        # Action space: [dx, dy, dz, d_yaw, gripper_state]
        # dx, dy, dz: position deltas in meters
        # d_yaw: yaw rotation delta in radians
        # gripper_state: 0 (closed) or 1 (open)
        
        low_limits = np.array([
            -0.01,  # dx
            -0.01,  # dy  
            -0.01,  # dz
            -0.1,   # d_yaw (radians)
            0.0     # gripper_state
        ])
        
        high_limits = np.array([
            0.01,   # dx
            0.01,   # dy
            0.01,   # dz
            0.1,    # d_yaw (radians)
            1.0     # gripper_state
        ])

        # Map action from [-1, 1] to [low_limits, high_limits]
        scaled = low_limits + (action + 1.0) * 0.5 * (high_limits - low_limits)
        
        return scaled.astype(np.float32)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
