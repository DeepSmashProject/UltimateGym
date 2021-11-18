import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
import cv2
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7200, 128)
        self.fc2 = nn.Linear(128, 11)
        self.path = Path(os.path.dirname(__file__)).resolve()
        self.model_path = str(self.path)+"/mnist_cnn_v2.pt"
        self.load()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

    def load(self):
        self.load_state_dict(torch.load(self.model_path))

    def predict_damage(self, damage_obs):
        damage = ""
        self.eval()
        for data in damage_obs:
            data = cv2.resize(data, (32, 32))
            #data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data = self.extract_black(data)
            X = [data]
            X = torch.tensor(X) / 255
            X = X.unsqueeze(1)
            output = self(X) # output is 0~10
            pred = output.argmax(dim=1, keepdim=True)[0][0].item()
            if pred != 10: # 10 is no number
                damage += str(pred)
        damage = int(damage) if len(damage) > 0 else 999
        return damage

    def extract_black(self, img):
        lower = np.array([0, 0, 0]) 
        upper = np.array([30, 30, 30])
        img = cv2.inRange(img, lower, upper)
        #img_mask = cv2.bitwise_not(img_mask,img_mask)
        #img = cv2.bitwise_not(img, img, mask=img_mask)
        return img

class NetV3(nn.Module):
    def __init__(self):
        super(NetV3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 11)
        self.path = Path(os.path.dirname(__file__)).resolve()
        self.model_path = str(self.path)+"/mnist_cnn_v3.pt"
        self.load()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    
    def load(self):
        self.load_state_dict(torch.load(self.model_path))

    def predict_damage(self, damage_obs):
        damage = ""
        self.eval()
        for data in damage_obs:
            data = cv2.resize(data, (32, 32))
            #data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data = self.extract_black(data)
            X = [data]
            X = torch.tensor(X) / 255
            X = X.unsqueeze(1)
            output = self(X) # output is 0~10
            pred = output.argmax(dim=1, keepdim=True)[0][0].item()
            if pred != 10: # 10 is no number
                damage += str(pred)
        damage = int(damage) if len(damage) > 0 else 999
        return damage

    def extract_black(self, img):
        lower = np.array([0, 0, 0]) 
        upper = np.array([30, 30, 30])
        img = cv2.inRange(img, lower, upper)
        #img_mask = cv2.bitwise_not(img_mask,img_mask)
        #img = cv2.bitwise_not(img, img, mask=img_mask)
        return img