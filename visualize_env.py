import numpy as np

from slime_volleyball.core import constants
from slime_volleyball.slimevolley_env import SlimeVolleyEnv
from slime_volleyball.baseline_policy import BaselinePolicy


if __name__ == "__main__":
    """
    Example of how to use Gym env, in single or multiplayer setting

    Humans can override controls:

    left Agent:
    W - Jump
    A - Left
    D - Right

    right Agent:
    Up Arrow, Left Arrow, Right Arrow
    """

    if constants.RENDER_MODE:
        from pyglet.window import key
        from time import sleep

    manualAction = [0, 0, 0]  # forward, backward, jump
    otherManualAction = [0, 0, 0]
    manualMode = False
    otherManualMode = False

    # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
    def key_press(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:
            manualAction[0] = 1
        if k == key.RIGHT:
            manualAction[1] = 1
        if k == key.UP:
            manualAction[2] = 1
        if k == key.LEFT or k == key.RIGHT or k == key.UP:
            manualMode = True

        if k == key.D:
            otherManualAction[0] = 1
        if k == key.A:
            otherManualAction[1] = 1
        if k == key.W:
            otherManualAction[2] = 1
        if k == key.D or k == key.A or k == key.W:
            otherManualMode = True

    def key_release(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:
            manualAction[0] = 0
        if k == key.RIGHT:
            manualAction[1] = 0
        if k == key.UP:
            manualAction[2] = 0
        if k == key.D:
            otherManualAction[0] = 0
        if k == key.A:
            otherManualAction[1] = 0
        if k == key.W:
            otherManualAction[2] = 0

    policy = BaselinePolicy()  # defaults to use RNN Baseline for player

    env = SlimeVolleyEnv({"survival_reward": True})

    if constants.RENDER_MODE:
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release

    obs, _ = env.reset(seed=np.random.randint(0, 10000))
    steps = 0
    total_reward = 0

    terminateds = truncateds = {"__all__": False}
    while not terminateds["__all__"] and not truncateds["__all__"]:
        obs_right, obs_left = obs["agent_right"], obs["agent_left"]

        if manualMode:  # override with keyboard
            right_action = manualAction
        else:
            right_action = policy.predict(obs_right)

        # if otherManualMode:
        #     left_action = otherManualAction
        # else:
        #     left_action = policy.predict(obs_left)

        actions = {
            "agent_right": right_action
        }  # {"agent_left": left_action, "agent_right": right_action}
        obs, reward, terminateds, truncateds, _ = env.step(actions)

        # if any([r > 0 or r < 0 for r in reward.values()]):
        #     print("reward", reward)
        #     manualMode = False
        #     otherManualMode = False

        total_reward += reward["agent_right"]
        steps += 1

        if constants.RENDER_MODE:
            env.render()
            sleep(0.01)

        # make the game go slower for human players to be fair to humans.
        if manualMode or otherManualMode:
            if constants.PIXEL_MODE:
                sleep(0.01)
            else:
                sleep(0.02)

    env.close()
    print("cumulative score", total_reward, "steps", steps)
