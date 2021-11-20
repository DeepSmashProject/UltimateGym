from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
import glob

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

####### For Training ########     
def extract_black(img):
    lower = np.array([0, 0, 0]) 
    upper = np.array([50, 50, 50])
    img = cv2.inRange(img, lower, upper)
    return img

def mean_test_target_split(X, y):
    # 要素数を等しくする
    u, counts = np.unique(y, return_counts=True)
    print(u, counts, min(counts))
    train_result_X = []
    train_result_y = []
    test_result_X = []
    test_result_y = []
    eval_result_X = []
    eval_result_y = []
    count = [0]*len(u)
    test_split_num = int(min(counts) * 0.7)
    eval_split_num = int(min(counts) * 0.9)
    for x, t in zip(X, y):
        count[t] += 1
        if count[t] <=  test_split_num:
            train_result_X.append(x)
            train_result_y.append(t)
        elif test_split_num < count[t] and count[t] <= eval_split_num:
            test_result_X.append(x)
            test_result_y.append(t)
        elif  eval_split_num < count[t] and count[t] <= min(counts):
            eval_result_X.append(x)
            eval_result_y.append(t)
        
    return torch.stack(train_result_X), torch.stack(train_result_y), torch.stack(test_result_X), torch.stack(test_result_y), torch.stack(eval_result_X), torch.stack(eval_result_y)

def get_data():
    data_path = Path(os.path.dirname(__file__)).resolve()
    files = glob.glob(str(data_path) + "/data/*.png")
    #print(files)
    X = []
    y = []
    for file in files:
        label = 10 if file[-5] == "x" else int(file[-5])
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        #print(img.shape)
        img = cv2.resize(img, (32, 32))
        img = extract_black(img)

        # Original Image
        img = Image.fromarray(img)
        X.append(img)
        y.append(label)

        # Croped Image
        transform = transforms.FiveCrop(int(img.width*0.90))
        imgs = transform(img)
        for img in imgs:
            img = np.array(img)
            img = cv2.resize(img, (32, 32))
            X.append(img)
            y.append(label)

    X = torch.tensor(X) / 255
    y = torch.tensor(y, dtype=torch.long)
    X = X.unsqueeze(1)
    print(X.size(), y.size(), X.dtype, y.dtype)
    return X, y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 11)

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

def train(model, device, data, target, optimizer, epoch):
    model.train()
    for i in range(int(len(data)/10), len(data), int(len(data)/10)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}]\tLoss: {:.6f}'.format(
            epoch, len(data), loss.item()))

def test(model, device, data, target):
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    test_loss = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    #print(pred, target)

    print('\nTest set: Average loss: {:.4f}, NUM: {} / {}, Accuracy: {}\n'.format(
        test_loss, correct, len(pred), correct /len(pred)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    X, y = get_data()
    train_X, train_y, test_X, test_y, eval_X, eval_y =mean_test_target_split(X, y)

    u, counts = np.unique(train_y, return_counts=True)
    print("train size", u, counts)
    u, counts = np.unique(test_y, return_counts=True)
    print("test size", u, counts)
    u, counts = np.unique(eval_y, return_counts=True)
    print("eval size", u, counts)

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, 20):
        #train(args, model, device, train_loader, optimizer, epoch)
        #test(model, device, test_loader)
        train(model, device, train_X, train_y, optimizer, epoch)
        test(model, device, test_X, test_y)
        #scheduler.step()

    # eval
    print("Evaluation")
    #model.load_state_dict(torch.load("./mnist_cnn_v4.pt"))
    test(model, device, eval_X, eval_y)

    if args.save_model:
        torch.save(model.state_dict(), "./mnist_cnn_v4.pt")


if __name__ == '__main__':
    main()