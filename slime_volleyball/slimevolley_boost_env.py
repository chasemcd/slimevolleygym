"""
Port of Neural Slime Volleyball to Python Gym Environment

David Ha (2020)

Original version:

https://otoro.net/slimevolley
https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
https://github.com/hardmaru/neuralslimevolley

No dependencies apart from Numpy and Gym
"""

import typing

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import cv2  # installed with gym anyways

from slime_volleyball.core import constants
from slime_volleyball.core import game
from slime_volleyball.baseline_policy import BaselinePolicy
from slime_volleyball.core import utils
from slime_volleyball.slimevolley_env import SlimeVolleyEnv

try:
    from slime_volleyball import rendering
except ImportError as e:
    print(
        f"Unable to import rendering. This means you won't be able to render the game: {e}"
    )

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)


class SlimeVolleyBoostEnv(SlimeVolleyEnv):
    """
    Augmentation of the SlimeVolleyEnv to include a boost action.
    """

    metadata = {
        "render.modes": ["human", "rgb_array", "state"],
        "video.frames_per_second": 50,
    }

    action_table = [
        [0, 0, 0, 0],  # NOOP
        [1, 0, 0, 0],  # LEFT (forward)
        [1, 0, 1, 0],  # UPLEFT (forward jump)
        [0, 0, 1, 0],  # UP (jump)
        [0, 1, 1, 0],  # UPRIGHT (backward jump)
        [0, 1, 0, 0],  # RIGHT (backward)
        # Boost versions
        [0, 0, 0, 1],  # NOOP (boost)
        [1, 0, 0, 1],  # LEFT (forward)
        [1, 0, 1, 1],  # UPLEFT (forward jump)
        [0, 0, 1, 1],  # UP (jump)
        [0, 1, 1, 1],  # UPRIGHT (backward jump)
        [0, 1, 0, 1],  # RIGHT (backward)
        [0, 0, 0, 1],  # NOOP (boost)
    ]

    default_config = {
        "from_pixels": False,
        "survival_reward": False,
        "max_steps": 3000,
        # if true, LHS actions are swapped to be intuitive for humans, otherwise directions are swapped
        "human_inputs": False,
    }

    def __init__(
        self,
        config: dict[str, typing.Any] | None = None,
        render_mode: str | None = None,
    ):
        """
        Reward modes:

        net score = right agent wins minus left agent wins

        0: returns net score (basic reward)
        1: returns 0.01 x number of timesteps (max 3000) (survival reward)
        2: sum of basic reward and survival reward

        0 is suitable for evaluation, while 1 and 2 may be good for training

        Setting multiagent to True puts in info (4th thing returned in stop)
        the otherObs, the observation for the other agent. See multiagent.py

        Setting self.from_pixels to True makes the observation with multiples
        of 84, since usual atari wrappers downsample to 84x84
        """
        super(SlimeVolleyBoostEnv, self).__init__()

        if config is None:
            config = self.default_config

        self._agent_ids = set(["agent_left", "agent_right"])
        self.t = 0
        self.max_steps = config.get("max_steps", 3000)
        self.from_pixels = config.get(
            "from_pixels", self.default_config["from_pixels"]
        )
        self.survival_reward = config.get(
            "survival_reward", self.default_config["survival_reward"]
        )
        self.human_inputs = config.get(
            "human_inputs", self.default_config["human_inputs"]
        )

        if self.from_pixels:
            constants.setPixelObsMode()
            observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(constants.PIXEL_HEIGHT, constants.PIXEL_WIDTH, 3),
                dtype=np.uint8,
            )
        else:
            high = np.array([np.finfo(np.float32).max] * 16)
            observation_space = spaces.Dict(
                {"obs": spaces.Box(-high, high, shape=(16,))}
            )

        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Discrete(len(self.action_table))
                for agent_id in self._agent_ids
            }
        )
        self.observation_space = spaces.Dict(
            {agent_id: observation_space for agent_id in self._agent_ids}
        )

        self.canvas = None
        self.previous_rgbarray = None

        self.game = game.SlimeVolleyGame(self.np_random)
        self.ale = (
            self.game.agent_right
        )  # for compatibility for some models that need the self.ale.lives() function

        self.policy = BaselinePolicy()  # the “bad guy”

        self.viewer = None

        # another avenue to override the built-in AI's action, going past many env wraps:
        self.otherAction = None

        self.render_mode = render_mode

    def seed(self, seed=None):
        self.ale = (
            self.game.agent_right
        )  # for compatibility for some models that need the self.ale.lives() function

    def get_obs(self):
        if self.from_pixels:
            obs = self.render(mode="state")
            self.canvas = obs
            return {
                "agent_left": cv2.flip(obs, 1),
                "agent_right": obs,
            }  # always observe from the same angle

        return {
            "agent_left": {
                "obs": self.game.agent_left.get_observation(boost=True)
            },
            "agent_right": {
                "obs": self.game.agent_right.get_observation(boost=True)
            },
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[dict[str, np.array], dict[str, typing.Any]]:
        self.init_game_state()

        return self.get_obs(), {}

    def discrete_to_box(self, n):
        # convert discrete action n into the actual triplet action
        if n is None:
            return n
        if isinstance(
            n, (list, tuple, np.ndarray)
        ):  # original input for some reason, just leave it:
            if len(n) in [3, 4]:
                return n
        assert (int(n) == n) and (n >= 0)  # and (n < 7)
        return self.action_table[n]

    @staticmethod
    def invert_action(action: list) -> list:
        left, right, up, boost = action
        return [right, left, up, boost]

    def step(self, actions):
        """
        note: although the action space is multi-binary, float vectors
        are fine (refer to setAction() to see how they get interpreted)
        """
        self.t += 1

        # If it's being treated as a single agent env, right agent is being
        # controlled and self.policy will control the left.
        if isinstance(actions, int):
            actions = {"agent_right": actions}

        left_agent_action = self.discrete_to_box(actions.get("agent_left"))
        right_agent_action = self.discrete_to_box(actions.get("agent_right"))

        if left_agent_action is None:  # override baseline policy
            obs = self.game.agent_left.get_observation()
            left_agent_action = self.policy.predict(obs)

        if self.human_inputs:
            left_agent_action = self.invert_action(left_agent_action)

        self.game.agent_left.set_action(left_agent_action)
        self.game.agent_right.set_action(right_agent_action)

        reward_right = self.game.step()
        survival_reward = 0.01 if self.survival_reward else 0.0

        # include survival bonus
        rewards = {
            "agent_left": -reward_right + survival_reward,
            "agent_right": reward_right + survival_reward,
        }

        obs = self.get_obs()

        terminateds, truncateds = self.get_terminateds_truncateds()

        return obs, rewards, terminateds, truncateds, {}
