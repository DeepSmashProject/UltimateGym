from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from damage_augumented_data.data import get_data, mean_test_target_split
import numpy as np


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