import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7200, 128)
        self.fc2 = nn.Linear(128, 11)
        self.model_path = "./mnist_cnn.pt"
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
            output = self(data) # output is 0~10
            if output != 10: # 10 is no number
                damage += str(output)
        damage = int(damage)
        return damage