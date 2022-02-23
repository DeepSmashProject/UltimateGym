import pyautogui
from .enums import Action
import time
from yuzulib.controller import Controller
from yuzulib.enums import Button
import os
from pathlib import Path
from .mode import Mode
from .enums import Action, Stage, Fighter

class UltimateController(Controller):
    def __init__(self):
        super(UltimateController, self).__init__()
        self.mode = Mode(self)

    def act(self, action: Action):
        buttons = action["buttons"]
        hold = action["hold"]
        sec = action["sec"]
        wait = action["wait"]
        refresh = action["refresh"]
        self.press(buttons, hold=hold, sec=sec, wait=wait, refresh=refresh)

if __name__ == '__main__':
    controller = UltimateController()
    controller.act(Action.ACTION_JAB)