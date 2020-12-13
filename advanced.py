import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from utils import *


class RayNet2(nn.Module):
    def __init__(self, input, output, hidden_sizes=None):
        super(RayNet2, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [20, 20, 20]
        self.layer1 = nn.Linear(input, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[1], output)
        self.layer3 = nn.Linear()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    IMAGE_PATH = "img/hangang.jpg"
    # step1: take a single color image
    original_img = cv2.imread(IMAGE_PATH)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # step 2: convert this color image to black and white
    original_img_gray = make_gray(original_img)
    # step 3: use the left half of the image as training data (both rgb and the black and white)
    train_rgb, _ = cut_rgb_img_half(original_img)
    # step 4: set aside the right half of the image as test data (black and white only)
    train_gray, test_gray = cut_img_in_half(original_img_gray)

    # Steps for NN training:
    # 1) for each pixel in the left half of the image:
    # 1a. split the data into x_train, y_train as (gray scale of current pixel, rgb of current pixel) for 80%
    # 1b. set aside 10% of the left half of the image data as x_val, y_val
    dataset = get_colors(train_rgb)
    gray_dataset = get_grays(train_gray)

    net = RayNet2(input=1, output=1)
    optim = torch.optim.SGD(params=net.parameters(), lr=0.03)
    criterion = nn.MSELoss()
    for x, y in zip(gray_dataset, dataset):
        pred = net(x)
        loss = criterion(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print("loss: {}".format(loss))

