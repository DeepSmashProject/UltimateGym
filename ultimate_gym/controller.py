from libultimate.client import UltimateClient
from libultimate.enums import Action
from threading import (Event)

class Controller:
    def __init__(self, address="http://localhost:6000") -> None:
        self.client = UltimateClient(address=address, disable_warning=True)

    def act(self, action: Action):
        self.client.act(action)

    def reset_training(self):
        self.client.reset_training()