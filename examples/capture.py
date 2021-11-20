

from ultimate_gym import Screen, UltimateEnv
from libultimate import Controller, Action, Fighter, Stage, TrainingMode
import time
import random
import argparse
import os
import subprocess

def take_screenshot(filename, left, top, width, height):
    #filename = "p1_1-0.png"
    # delete screenshot
    if os.path.isfile(filename):
        os.remove(filename)

    command = "scrot -o --autoselect '{},{},{},{}' damage_data/{}".format(left, top, width, height, filename)
    proc = subprocess.run(command, shell=True, executable='/bin/bash')
    if proc.returncode == 0:
        print("Take screenshot successfully: ", filename)
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
            filename = "{}_{}_{}.png".format(d["name"],i,d["digit"])
            take_screenshot(filename, d["left"], d["top"], d["width"], d["height"])
collect()