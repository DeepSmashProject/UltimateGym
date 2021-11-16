
from ultimate_gym import Screen, UltimateEnv
from yuzulib import Runner
from libultimate import Controller, Action
import time
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', help="game path: ex. /path/to/game[v0].nsp")
parser.add_argument('-d', '--dlc', help="dlc dir: ex. /path/to/dlc/")
args = parser.parse_args()

if args.game == "" or args.dlc == "":
    print("Invalid argument")
    os.exit(1)

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
    Action.ACTION_JAMP,
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

env = UltimateEnv(args.game, args.dlc)
for k in range(10):
    obs = env.reset()
    for i in range(10):
        action = random.choice(action_list)
        next_obs, reward, done, info = env.step(action)
        print("episode: {}, step: {}, obs: {}".format(k, i, next_obs[0][0]))
print("finished!")