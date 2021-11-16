
from libultimate import Controller, Action, Fighter, Stage, TrainingMode
from yuzulib import Runner
from .screen import Screen
import gym
import numpy as np
import cv2
import time

action_list = [
    Action.ACTION_JAB,
    Action.ACTION_RIGHT_TILT,
    Action.ACTION_LEFT_TILT,
    Action.ACTION_UP_TILT,
    Action.ACTION_DOWN_TILT,
    Action.ACTION_RIGHT_SMASH,
    Action.ACTION_LEFT_SMASH,
    Action.ACTION_UP_SMASH,
    Action.ACTION_DOWN_SMASH,
    Action.ACTION_NEUTRAL_SPECIAL,
    Action.ACTION_RIGHT_SPECIAL,
    Action.ACTION_LEFT_SPECIAL,
    Action.ACTION_UP_SPECIAL,
    Action.ACTION_DOWN_SPECIAL,
    Action.ACTION_GRAB,
    Action.ACTION_SHIELD,
    Action.ACTION_JUMP,
    Action.ACTION_SHORT_HOP,
    #Action.ACTION_UP_TAUNT,
    #Action.ACTION_DOWN_TAUNT,
    #Action.ACTION_LEFT_TAUNT,
    #Action.ACTION_RIGHT_TAUNT,
    Action.ACTION_SPOT_DODGE,
    Action.ACTION_RIGHT_ROLL,
    Action.ACTION_LEFT_ROLL,
    Action.ACTION_RIGHT_DASH,
    Action.ACTION_LEFT_DASH,
    Action.ACTION_RIGHT_WALK,
    Action.ACTION_LEFT_WALK,
    Action.ACTION_CROUCH,
    #Action.ACTION_RIGHT_CRAWL,
    #Action.ACTION_LEFT_CRAWL,
    Action.ACTION_RIGHT_STICK,
    Action.ACTION_LEFT_STICK,
    Action.ACTION_UP_STICK,
    Action.ACTION_DOWN_STICK,
    Action.ACTION_NO_OPERATION
]

class UltimateEnv(gym.Env):
    def __init__(self, game_path: str, dlc_dir: str):
        super().__init__()
        self.game_path = game_path
        self.dlc_dir = dlc_dir
        self.action_space = gym.spaces.Discrete(len(action_list)) 
        self.controller, self.screen, self.training_mode = self._setup()
        self.prev_observation, self.prev_info = self._reset()

    def _setup(self):
        runner = Runner(self.game_path, self.dlc_dir)
        runner.run()
        controller = Controller()
        controller.move_to_home()

        training_mode = TrainingMode(
            controller=controller,
            stage=Stage.STAGE_FINAL_DESTINATION, 
            player=Fighter.FIGHTER_MARIO,
            cpu=Fighter.FIGHTER_DONKEY_KONG,
            cpu_level=7,
        )
        print("Training Mode")
        training_mode.start()
        time.sleep(1)
        screen = Screen()
        screen.run()
        time.sleep(1)
        return controller, screen, training_mode

    def _reset(self):
        # click reset button
        self.training_mode.reset()
        return self._observe()

    def _step(self, action: Action):
        self.controller.act(action)
        observation, info = self._observe()
        reward = self._get_reward(info, self.prev_info)
        self.done = self._is_done(info, self.prev_info)
        self.prev_observation = observation
        self.prev_info = info
        return observation, reward, self.done, info

    def _render(self, mode='human', close=False):
        print("You can see screen at http://localhost:8081/vnc.html")
        if mode == 'human':
            cv2.imshow('test', self.prev_observation)
        else:
            print("Info: {}".format(self.prev_info))

    def _observe(self):
        frame, fps = self.screen.get()
        # resolution = 512x512 grayscale, 
        observation = frame
        # remove background color
        damage = self._get_damage(observation)
        kill = self._get_kill(observation)
        # get damege
        return observation, {"damage": damage, "kill": kill}

    def _is_done(self, observation, prev_observation):
        # if 
        done = False
        return done

    def _get_reward(self, observation, prev_observation):
        reward = 0
        return reward

    def _get_damage(self, observation):
        # read damage from observation
        damage = []
        return damage

    def _get_kill(self, observation):
        # read damage from observation
        kill = []
        return kill