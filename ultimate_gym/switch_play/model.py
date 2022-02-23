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
        self.path = Path(os.path.dirname(__file__)).joinpath("..").resolve()
        self.model_path = str(self.path)+"/mnist/mnist_cnn_v2.pt"
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
        self.model_path = str(self.path)+"/mnist/mnist_cnn_v3.pt"
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

class NetV4(nn.Module):
    def __init__(self):
        super(NetV4, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 11)
        self.path = Path(os.path.dirname(__file__)).resolve()
        self.model_path = str(self.path)+"/mnist/mnist_cnn_v4.pt"
        #self.p1_damage_obs = ((127, 414, 24, 30), (149, 414, 24, 30), (171, 414, 24, 30))
        #self.p2_damage_obs = ((309, 414, 24, 30), (331, 414, 24, 30), (353, 414, 24, 30))
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

    def predict_damage(self, damage_obs_list):
        damage = ""
        self.eval()
        for damage_obs in damage_obs_list:
            processed_obs = self.precess_obs(damage_obs)
            output = self(processed_obs) # output is 0~10
            pred = output.argmax(dim=1, keepdim=True)[0][0].item()
            if pred != 10: # 10 is no number
                damage += str(pred)
        damage = int(damage) if len(damage) > 0 else 999
        return damage

    def process_obs(self, obs):
        obs = cv2.resize(obs, (32, 32))
        obs = self.extract_black(obs)
        obs = [obs]
        obs = torch.tensor(obs) / 255
        obs = obs.unsqueeze(1)
        return obs

    def extract_black(self, img):
        lower = np.array([0, 0, 0]) 
        upper = np.array([50, 50, 50])
        img = cv2.inRange(img, lower, upper)
        return img

class NetV5(nn.Module):
    def __init__(self):
        super(NetV5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 11)
        self.path = Path(os.path.dirname(__file__)).resolve()
        self.model_path = str(self.path)+"/mnist/mnist_cnn_v5.pt"
        #self.p1_damage_obs = ((127, 414, 24, 30), (149, 414, 24, 30), (171, 414, 24, 30))
        #self.p2_damage_obs = ((309, 414, 24, 30), (331, 414, 24, 30), (353, 414, 24, 30))
        #self.load()

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

    def _preprocess(self, obs):
        # obs: (B, W, H, C) -> (B, 32, 32, 3)
        #print("preprocess", obs.size())
        result_obs = []
        for img in obs:
            #img = img.numpy()
            #print(img.shape)
            img = cv2.resize(img, (32, 32))
            img = self._extract_black(img)
            result_obs.append(img)
        
        result_obs = np.array(result_obs, dtype=np.double)
        result_obs = torch.tensor(result_obs) / 255
        result_obs = result_obs.unsqueeze(1)
        #print("test", result_obs.size())
        # resule_obs: (B, C, W, H) -> (B, 1, 32, 32)
        return result_obs

    def load(self):
        self.load_state_dict(torch.load(self.model_path))

    def predict(self, obs):
        # obs: (B, W, H, C) -> (B, 32, 32, 3)
        obs = self._preprocess(obs)
        output = self(obs.float()) # output is 0~10
        #print("debug2", obs.size(), output.size())
        pred = output.argmax(dim=1, keepdim=True)[0][0].item()
        return pred
        
    def predict_damage(self, damage_obs_list):
        damage = ""
        self.eval()
        for i, damage_obs in enumerate(damage_obs_list):
            pred = self.predict([damage_obs])
            if i == 0 and pred >= 3:
                pred = 10 # no more than > 300 %
            if pred != 10: # 10 is no number
                damage += str(pred)
        damage = int(damage) if len(damage) > 0 else 999
        return damage

    def _extract_black(self, img):
        lower = np.array([0, 0, 0]) 
        upper = np.array([50, 50, 50])
        img = cv2.inRange(img, lower, upper)
        return img