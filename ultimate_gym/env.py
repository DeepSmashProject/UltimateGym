
from libultimate import Controller, Action, Fighter, Stage, TrainingMode, Runner
from .screen import Screen
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
    def __init__(self, game_path: str, dlc_dir: str, screen, controller, mode, without_setup: bool = False):
        super().__init__()
        self.game_path = game_path
        self.dlc_dir = dlc_dir
        self.action_space = gym.spaces.Discrete(len(action_list)) 

        # damage predict model
        device = torch.device("cpu")
        self.model = NetV5().to(device)
        self.model.load()

        self.buffer_size = 2
        self.p1_d_buffer = deque([], self.buffer_size)
        self.p2_d_buffer = deque([], self.buffer_size)
        self.p1_damage = 0
        self.p2_damage = 0
        self.p1_damaged_or_killed_flag = False
        self.p2_damaged_or_killed_flag = False
        self.screen = screen
        self.screen.run()
        time.sleep(1) # waiting run screen thread
        self.controller = controller
        self.mode = mode
        if not without_setup:
            self._setup()
        self.prev_observation, self.prev_info = self.reset()

    def _setup(self):
        runner = Runner(self.game_path, self.dlc_dir)
        runner.run()
        self.controller.move_to_home()
        self.mode.start()
        print("Training Mode")
        time.sleep(1)

    def reset(self):
        # click reset button
        self.p1_d_buffer.clear()
        self.p2_d_buffer.clear()
        self.p1_damage = 0
        self.p2_damage = 0
        self.p1_damaged_or_killed_flag = False
        self.p2_damaged_or_killed_flag = False
        self.mode.reset()
        time.sleep(1) # waiting for setup
        return self._observe()

    def step(self, action: Action):
        self.controller.act(action)
        observation, info = self._observe()
        reward = self._reward(info)
        self.done = self._done(info)
        self.prev_observation = observation
        self.prev_info = info
        return observation, reward, self.done, info

    def render(self, mode='human', close=False):
        print("You can see screen at http://localhost:8081/vnc.html")
        if mode == 'human':
            cv2.imshow('test', self.prev_observation)
        else:
            print("Info: {}".format(self.prev_info))

    def close(self):
        self.screen.close()

    def _observe(self):
        frame, fps = self.screen.get()
        # resolution = 512x512 grayscale, 
        observation = frame[:, :, :3]
        # remove background color
        damage, diff_damage, kill = self._get_damage(observation)
        #kill = self._get_kill(damage)
        # get damege
        return observation, {"damage": damage, "diff_damage": diff_damage, "kill": kill}

    def _done(self, info):
        done = False
        if info["kill"][0] or info["kill"][1]:
            done = True
        return done

    def _reward(self, info):
        p1_diff_damage = info["diff_damage"][0]
        p2_diff_damage = info["diff_damage"][1]
        reward = (p2_diff_damage - p1_diff_damage) / 20
        return reward

    def _get_damage(self, observation):
        # read damage from observation
        #p1_damage_obs = (observation[414:444, 127:151], observation[414:444, 149:173], observation[414:444, 171:195]) #[y,x]
        #p2_damage_obs = (observation[414:444, 309:333], observation[414:444, 331:355], observation[414:444, 353:377]) #[y,x] 
        p1_damage_obs = (observation[411:441, 124:148], observation[411:441, 146:170], observation[411:441, 168:192]) #[y,x]
        p2_damage_obs = (observation[411:441, 306:330], observation[411:441, 328:352], observation[411:441, 350:374]) #[y,x] 
        p1_damage = self.model.predict_damage(p1_damage_obs)
        p2_damage = self.model.predict_damage(p2_damage_obs)
        self.p1_d_buffer.append(p1_damage)
        self.p2_d_buffer.append(p2_damage)
        p1_diff_damage = 0
        p2_diff_damage = 0
        p1_killed = False
        p2_killed = False
        if self.p1_d_buffer.count(self.p1_damage) == 0 and self.p1_d_buffer.count(999) == 0 and p1_damage != self.p1_damage:
            p1_diff_damage = p1_damage - self.p1_damage
            if p1_diff_damage > 0:
                self.p1_damage = p1_damage
            else:
                p1_diff_damage = 0
            # kill or not
            if self.p1_damaged_or_killed_flag and p1_damage == 0:
                p1_killed = True
        if self.p2_d_buffer.count(self.p2_damage) == 0 and self.p2_d_buffer.count(999) == 0 and p2_damage != self.p2_damage:
            p2_diff_damage = p2_damage - self.p2_damage
            if p2_diff_damage > 0:
                self.p2_damage = p2_damage
            else:
                p2_diff_damage = 0
            # kill or not
            if self.p2_damaged_or_killed_flag and p2_damage == 0:
                p2_killed = True

        # killed flag
        if self.p1_d_buffer.count(999) >= len(self.p1_d_buffer):
            self.p1_damaged_or_killed_flag = True
        if self.p2_d_buffer.count(999) >= len(self.p2_d_buffer):
            self.p2_damaged_or_killed_flag = True
            
        return (self.p1_damage, self.p2_damage), (p1_diff_damage, p2_diff_damage), (p1_killed, p2_killed)

    def _get_kill(self, damage):
        (p1_damage, p2_damage) = damage
        #self.p1_d_buffer.append(p1_damage)
        #self.p2_d_buffer.append(p2_damage)
        if len(self.p1_d_buffer) < self.buffer_size or len(self.p2_d_buffer) < self.buffer_size:
            return (False, False)
        # exist 150 and 0 in 5 queue and majority
        killed = self.p1_d_buffer.count(999)  >= len(self.p1_d_buffer) and self.p2_d_buffer.count(999) >= len(self.p2_d_buffer)
        p1_kill = killed and self.p1_d_buffer.count(999) > self.p2_d_buffer.count(999)
        p2_kill = killed and self.p2_d_buffer.count(999) > self.p1_d_buffer.count(999)
        kill = (p1_kill, p2_kill)

        return kill