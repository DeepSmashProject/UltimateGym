
from ultimate_gym import Screen, UltimateEnv
from libultimate import Controller, Action, Fighter, Stage, TrainingMode
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

screen = Screen(fps=30)
controller = Controller()
training_mode = TrainingMode(
    controller=controller,
    stage=Stage.STAGE_FINAL_DESTINATION, 
    player=Fighter.FIGHTER_MARIO,
    cpu=Fighter.FIGHTER_DONKEY_KONG,
    cpu_level=7,
)

env = UltimateEnv(args.game, args.dlc, screen, controller, training_mode, without_setup=False)
for k in range(10):
    done = False
    obs = env.reset()
    step = 0
    while not done:
        action = random.choice(action_list)
        next_obs, reward, done, info = env.step(action)
        print("episode: {}, step: {}, obs: {}, done: {}, damage: {}, diff_damage: {}, kill: {}".format(k, step, next_obs[100][100], done, info["damage"], info["diff_damage"], info["kill"]))
        step += 1
env.close()
print("finished!")