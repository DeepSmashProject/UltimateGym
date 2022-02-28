from .screen import Screen
from yuzulib.game.ssbu import Action, UltimateController
import gym
import numpy as np
import cv2
import time
from collections import deque
import torch
from .model import Net, NetV3, NetV5
# window start 211 138 853 487
# window + screen -> screen only -> 
# player1 damage 1 point (382,552,24,30) -> (171, 414, 24, 30)
# player1 damage 10 point (360,552,24,30) -> (149, 414, 24, 30)
# player1 damage 100 point (338,552,24,30) -> (127, 414, 24, 30)
# player2 damage point (564,552,24,30) -> (353, 414, 24, 30)
# player2 damage point (542,552,24,30) -> (331, 414, 24, 30)
# player2 damage point (520,552,24,30) -> (309, 414, 24, 30)
# -10 width -10 height
# player1 damage 1 point (387,557,14,20) -> (176, 419, 14, 20)
# player1 damage 10 point (365,557,14,20) -> (154, 419, 14, 20)
# player1 damage 100 point (343,557,14,20) -> (132, 419, 14, 20)
# player2 damage point (569,557,14,20) -> (358, 419, 14, 20)
# player2 damage point (547,557,14,20) -> (336, 419, 14, 20)
# player2 damage point (525,557,14,20) -> (314, 419, 14, 20)


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
    Action.ACTION_UP_RIGHT_SPECIAL,
    Action.ACTION_UP_LEFT_SPECIAL,
    Action.ACTION_GRAB,
    Action.ACTION_SHIELD,
    Action.ACTION_JUMP,
    Action.ACTION_SHORT_HOP,
    Action.ACTION_UP_TAUNT,
    Action.ACTION_DOWN_TAUNT,
    Action.ACTION_LEFT_TAUNT,
    Action.ACTION_RIGHT_TAUNT,
    Action.ACTION_SPOT_DODGE,
    Action.ACTION_RIGHT_ROLL,
    Action.ACTION_LEFT_ROLL,
    Action.ACTION_RIGHT_DASH,
    Action.ACTION_LEFT_DASH,
    Action.ACTION_RIGHT_WALK,
    Action.ACTION_LEFT_WALK,
    Action.ACTION_CROUCH,
    Action.ACTION_RIGHT_CRAWL,
    Action.ACTION_LEFT_CRAWL,
    Action.ACTION_RIGHT_STICK,
    Action.ACTION_LEFT_STICK,
    Action.ACTION_UP_STICK,
    Action.ACTION_DOWN_STICK,
    Action.ACTION_NO_OPERATION
]

class Buffer:
    def __init__(self, capacity) -> None:
        self.data = []
        self.capacity = capacity

    def add(self, value):
        self.data.append(value)
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def is_full_value(self, value):
        return self.data.count(value) == len(self.data)

    def is_full_capacity(self):
        return len(self.data) == self.capacity

    def get_full_same_value(self):
        if not self.is_full_capacity():
            return -1
        for d in self.data:
            if d != self.data[0]:
                return -1
        return self.data[0]

    def count_value_from_back(self, value):
        count = 0
        for v in reversed(self.data):
            if v == value:
                count += 1
            else:
                break
        return count

    def clear(self):
        self.data = []

class UltimateEnv(gym.Env):
    def __init__(self, fps=60):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(action_list)) 

        # damage predict model
        device = torch.device("cpu")
        self.model = NetV5().to(device)
        self.model.load()

        self.buffer_size = 10
        self.p1_d_buffer = Buffer(self.buffer_size)
        self.p2_d_buffer = Buffer(self.buffer_size)
        self.recent_buffer_size = 3
        self.p1_recent_d_buffer = Buffer(self.recent_buffer_size)
        self.p2_recent_d_buffer = Buffer(self.recent_buffer_size)
        self.p1_damage = 0
        self.p2_damage = 0
        self.p1_damaged_or_killed_flag = False
        self.p2_damaged_or_killed_flag = False
        self.screen = Screen(fps=fps)
        self.screen.run()
        time.sleep(1) # waiting run screen thread
        self.controller = UltimateController()
        self.prev_observation = self.reset()

    def reset(self, without_reset=False):
        # click reset button
        self.p1_d_buffer.clear()
        self.p2_d_buffer.clear()
        self.p1_damaged_or_killed_flag = False
        self.p2_damaged_or_killed_flag = False
        if not without_reset:
            self.p1_damage = 0
            self.p2_damage = 0
            self.controller.mode.training.reset()
        time.sleep(0.5) # waiting for setup
        observation = self._observe()
        return observation

    def step(self, action: Action):
        self.controller.act(action)
        observation = self._observe()
        damage, diff_damage, kill = self._get_damage(observation)
        info = {"damage": damage, "diff_damage": diff_damage, "kill": kill}
        reward = self._reward(info)
        self.done = self._done(info)
        self.prev_observation = observation
        self.prev_info = info
        return observation, reward, self.done, info

    def render(self, mode='human', close=False):
        if mode == 'human':
            print("You can see screen at http://localhost:8081/vnc.html")
        else:
            print("Info: {}".format(self.prev_info))

    def _observe(self):
        frame, fps = self.screen.get()
        # frame (width, height, 1 if gray_scale else 3)
        return frame

    def _done(self, info):
        done = False
        if info["kill"][0] or info["kill"][1]:
            done = True
        return done

    def _reward(self, info):
        p1_diff_damage = info["diff_damage"][0]
        p2_diff_damage = info["diff_damage"][1]
        reward = p2_diff_damage - p1_diff_damage
        return reward

    def _get_damage(self, observation):
        # read damage from observation
        #p1_damage_obs = (observation[414:444, 127:151], observation[414:444, 149:173], observation[414:444, 171:195]) #[y,x]
        #p2_damage_obs = (observation[414:444, 309:333], observation[414:444, 331:355], observation[414:444, 353:377]) #[y,x] 
        p1_damage_obs = (observation[411:441, 124:148], observation[411:441, 146:170], observation[411:441, 168:192]) #[y,x]
        p2_damage_obs = (observation[411:441, 306:330], observation[411:441, 328:352], observation[411:441, 350:374]) #[y,x] 
        p1_damage = self.model.predict_damage(p1_damage_obs)
        p2_damage = self.model.predict_damage(p2_damage_obs)
        self.p1_d_buffer.add(p1_damage)
        self.p2_d_buffer.add(p2_damage)
        self.p1_recent_d_buffer.add(p1_damage)
        self.p2_recent_d_buffer.add(p2_damage)
        p1_diff_damage = 0
        p2_diff_damage = 0
        p1_killed = False
        p2_killed = False
        if p1_damage == 999:
            self.p1_damaged_or_killed_flag = True
        if p2_damage == 999:
            self.p2_damaged_or_killed_flag = True
        if self.p1_recent_d_buffer.get_full_same_value() != -1:
            # kill or not
            if self.p1_damaged_or_killed_flag and self.p1_recent_d_buffer.is_full_value(0) and ((self.p2_damage == 0 and self.p1_d_buffer.count_value_from_back(0) < self.p2_d_buffer.count_value_from_back(0)) or (self.p2_damage != 0 and self.p2_recent_d_buffer.get_full_same_value() != -1 and self.p2_recent_d_buffer.get_full_same_value() != 999)):
                p1_killed = True
                self.p1_damage = 0
                self.p1_damaged_or_killed_flag = False
            # calc damage
            elif not self.p1_recent_d_buffer.is_full_value(999) and self.p1_damage < self.p1_recent_d_buffer.get_full_same_value():
                p1_diff_damage = p1_damage - self.p1_damage
                self.p1_damage = p1_damage
        if self.p2_recent_d_buffer.get_full_same_value() != -1:
            if self.p2_damaged_or_killed_flag and self.p2_recent_d_buffer.is_full_value(0) and ((self.p1_damage == 0 and self.p2_d_buffer.count_value_from_back(0) < self.p1_d_buffer.count_value_from_back(0)) or (self.p1_damage != 0 and self.p1_recent_d_buffer.get_full_same_value() != -1 and self.p1_recent_d_buffer.get_full_same_value() != 999)):
                p2_killed = True
                self.p2_damage = 0
                self.p2_damaged_or_killed_flag = False
            elif not self.p2_recent_d_buffer.is_full_value(999) and self.p2_damage < self.p2_recent_d_buffer.get_full_same_value():
                p2_diff_damage = p2_damage - self.p2_damage
                self.p2_damage = p2_damage
            
        return (self.p1_damage, self.p2_damage), (p1_diff_damage, p2_diff_damage), (p1_killed, p2_killed)

