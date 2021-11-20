
from ultimate_gym import Screen, UltimateEnv
from libultimate import Controller, Action, Fighter, Stage, TrainingMode
import time
import random
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', help="game path: ex. /path/to/game[v0].nsp")
parser.add_argument('-d', '--dlc', help="dlc dir: ex. /path/to/dlc/")
args = parser.parse_args()

if args.game == "" or args.dlc == "":
    print("Invalid argument")
    os.exit(1)

action_list = [
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

def take_screenshot(filename, left, top, width, height):
    filename = "p1_1-0.png"
    # delete screenshot
    if os.path.isfile(filename):
        os.remove(filename)

    command = "scrot -o --autoselect '{},{},{},{}' damage_data/{}".format(left, top, width, height, filename)
    proc = subprocess.run(command, shell=True, executable='/bin/bash')
    if proc.returncode == 0:
        print("Take screenshot successfully")
    return filename

data = [
    {"name": "p1", "digit": 1, "left": 382, "top": 552, "width": 24, "height": 30},
    {"name": "p1", "digit": 10, "left": 360, "top": 552, "width": 24, "height": 30},
    {"name": "p1", "digit": 100, "left": 338, "top": 552, "width": 24, "height": 30},
    {"name": "p2", "digit": 1, "left": 564, "top": 552, "width": 24, "height": 30},
    {"name": "p2", "digit": 10, "left": 542, "top": 552, "width": 24, "height": 30},
    {"name": "p2", "digit": 100, "left": 520, "top": 552, "width": 24, "height": 30},
]

def collect():
    for i in range(100):
        time.sleep(3)
        for d in data:
            filename = "{}_{}_{}.png".format(d["name"], i, d["digit"])
            take_screenshot(filename, d["left"], d["top"], d["width"], d["height"])



env = UltimateEnv(args.game, args.dlc, screen, controller, training_mode, without_setup=False)
for k in range(10):
    done = False
    obs = env.reset()
    step = 0
    while not done:
        action = random.choice(action_list)
        next_obs, reward, done, info = env.step(action)
        print("episode: {}, step: {}, obs: {}, done: {}, damage: {}, kill: {}".format(k, step, next_obs[100][100], done, info["damage"], info["kill"]))
        step += 1
env.close()
print("finished!")