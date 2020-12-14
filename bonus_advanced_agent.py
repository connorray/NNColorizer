import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from utils import *


def data_loader(rgb_img, gray_img, w, h, input_size, output_size):
    # get the 3x3 patches and get the data in the correct shape and what not
    X, Y = np.zeros((w*h, input_size), int), np.zeros((w*h, output_size), int)
    ind = 0
    for i in range(w):
        for j in range(h):
            X[ind] = get_3_patch(gray_img, i, j).flatten()
            Y[ind] = rgb_img[i, j, 0:3]
            ind+=1
    return X, Y


def test_data_loader(gray_img, w, h, input_size):
    # get the 3x3 patches and get the data in the correct shape and what not
    X = np.zeros((w*h, input_size), int)
    ind = 0
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            X[ind] = get_3_patch(gray_img, i, j).flatten()
            ind+=1
    return X


class RayNet2(nn.Module):
    def __init__(self, input, output, hidden_sizes=None, device="cpu"):
        super(RayNet2, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 64]
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input, hidden_sizes[0], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == '__main__':
    IMAGE_PATH = "img/small_train_img.jpg"
    # step1: take a single color image
    original_img = cv2.imread(IMAGE_PATH)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # step 2: convert this color image to black and white
    original_img_gray = make_gray(original_img)
    # step 3: use the left half of the image as training data (both rgb and the black and white)
    train_rgb, _ = cut_rgb_img_half(original_img)
    # step 4: set aside the right half of the image as test data (black and white only)
    train_gray, test_gray = cut_img_in_half(original_img_gray)

    w, h = train_rgb.shape[0], train_rgb.shape[1]
    c = train_rgb.shape[2]
    input_size, output_size = 9, 3
    hidden = [32, 64]
    lr = 0.005

    # normalize data
    train_rgb = np.true_divide(train_rgb, 255)
    train_gray = np.true_divide(train_gray, 255)
    test_gray = np.true_divide(test_gray, 255)

    net = RayNet2(input=input_size, output=output_size, hidden_sizes=hidden)
    criterion = nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optim = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9)
    net.to(device)
    print("Running on: ", device)

    X, Y = data_loader(train_rgb, train_gray, w, h, input_size, output_size)
    epochs = 300
    episodes = 1000
    for e in range(epochs):
        avg_loss = 0.0
        for t in range(episodes):
            rand_pixel = np.random.randint(0, (X.shape[0]) - 1)  # would be expensive to shuffle
            # print(torch.Tensor(X[rand_pixel]))
            pred = net(torch.Tensor(X[rand_pixel]).to(device))
            # print(torch.Tensor(Y[rand_pixel]))
            # exit()
            loss = criterion(pred, torch.Tensor(Y[rand_pixel]).to(device))
            avg_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
        avg_loss /= episodes
        print("epoch {} | avg loss: {}".format(e, avg_loss))

    X_test = test_data_loader(test_gray, w, h, input_size)
    reconstructed_image = []
    for i in range(w * h):
        pred = net(torch.Tensor(X_test[i]).to(device))
        pred = pred.cpu().detach().numpy()
        pred = pred*255
        reconstructed_image.append(pred)
    reconstructed_image = np.array(reconstructed_image)
    reconstructed_image = reconstructed_image.reshape((w, h, c))
    np.clip(reconstructed_image, 0, 255, out=reconstructed_image)
    reconstructed_image = reconstructed_image.astype('uint8')
    plot_img_color(reconstructed_image)

