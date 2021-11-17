
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
    def __init__(self, game_path: str, dlc_dir: str, screen, controller, mode, without_setup: bool = False):
        super().__init__()
        self.game_path = game_path
        self.dlc_dir = dlc_dir
        self.action_space = gym.spaces.Discrete(len(action_list)) 
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
        self.mode.reset()
        return self._observe()

    def step(self, action: Action):
        self.controller.act(action)
        observation, info = self._observe()
        reward = self._reward(info, self.prev_info)
        self.done = self._done(info, self.prev_info)
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
        observation = frame
        # remove background color
        damage = self._get_damage(observation)
        kill = self._get_kill(observation)
        # get damege
        return observation, {"damage": damage, "kill": kill}

    def _done(self, observation, prev_observation):
        # if 
        done = False
        return done

    def _reward(self, observation, prev_observation):
        reward = 0
        return reward

    def _get_damage(self, observation):
        # read damage from observation
        # almost black to black (0,0,0)
        p1_damage_obs = observation[418:440, 178:191] #[y,x]
        p2_damage_obs = observation[418:440, 358:373] #[y,x]
        p1_damage_rgb = self._get_damage_rgb(p1_damage_obs)
        p2_damage_rgb = self._get_damage_rgb(p2_damage_obs)
        p1_damage = self._rgb_to_damage(p1_damage_rgb)
        p2_damage = self._rgb_to_damage(p2_damage_rgb)
        return (p1_damage, p2_damage)

    def _rgb_to_damage(self, rgb):
        print(rgb)
        (r, g, b) = rgb
        # damage color list 0~150%   // color=R+G
        damage_color_list = [510, 500, 480, 455, 420, 390, 360, 330, 300, 290, 275, 265, 255, 248, 237, 227]
        color = int(r) + int(g)
        damage = 0
        if color >= damage_color_list[0]: return 0
        if color <= damage_color_list[len(damage_color_list)-1]: return 150
        idx = np.abs(np.asarray(damage_color_list) - color).argmin()
        if color >= damage_color_list[idx]:
            rate = (color - damage_color_list[idx]) / (damage_color_list[idx-1] - damage_color_list[idx])
            damage = (idx- 1*rate) * 10 
        else:
            rate = (color - damage_color_list[idx+1]) / (damage_color_list[idx] - damage_color_list[idx+1])
            damage = (idx + 1*rate) * 10 
        return damage

    def _get_damage_rgb(self, img):
        lower = np.array([0,0,0]) 
        upper = np.array([100,100,100])
        img_mask = cv2.inRange(img, lower, upper)
        img_mask = cv2.bitwise_not(img_mask,img_mask)
        img = cv2.bitwise_and(img, img, mask=img_mask)
        print(img)
        cv2.imwrite("damage_remove_black.png", img)
        u, counts = np.unique(img[:, :, 0], return_counts=True)
        b = u[np.argmax(counts[1:])]
        u, counts = np.unique(img[:, :, 1], return_counts=True)
        g = u[np.argmax(counts[1:])]
        u, counts = np.unique(img[:, :, 2], return_counts=True)
        r = u[np.argmax(counts[1:])]
        return (r, g, b)

    def _get_kill(self, observation):
        # read damage from observation
        kill = []
        return kill