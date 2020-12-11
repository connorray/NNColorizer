from utils import *
import cv2
import random
from PIL import Image
import os


def relu(z):
    return max(z, 0)


def sigmoid(z):
    return 1/(np.exp(-z)+1)


input_size = 9
h_out_size = 100
o_out_size = 3


class model:
    def __init__(self):
        self.alpha = 0.01
        self.weights_o = np.random.rand(h_out_size+1, o_out_size,)*32
        self.weights_h = 0.02*np.random.rand(input_size+1, h_out_size)-0.01

    def forward(self, x, y, training=False):
        # Hidden  layer
        x_pad = np.pad(x, (1, 0), "constant", constant_values=1)
        in_h = np.dot(x_pad, self.weights_h)
        out_h = sigmoid(in_h)
        print(out_h.shape)
        # Output layer
        out_h_pad = np.pad(out_h, (1, 0), "constant", constant_values=1)
        in_o = np.dot(out_h_pad, self.weights_o)
        out_o = in_o

        if training:
            error = (out_o - y)
            loss = (np.sum(error**2))
            # print(f"output:{out_o} y:{y} loss: {loss}")

            # Update the output layer's weights
            mod_errk = error
            gradk = 2*self.alpha*np.matmul(out_h_pad.reshape(h_out_size+1,1),
                                              mod_errk.reshape(1,3))
            new_weights_o = self.weights_o - gradk

            #Update the hidden layer's weights (10,5)
            g_prime_inj = out_h_pad*(1-out_h_pad)
            print(g_prime_inj.shape)
            mod_errj = np.dot(self.weights_o, mod_errk)
            print(mod_errj.shape)
            mod_errj *=  g_prime_inj
            print(mod_errj.shape)
            exit()
            gradj = 2*self.alpha*np.matmul(x_pad.reshape(10,1),
                                              mod_errj[1:].reshape(1,h_out_size))
            self.weights_h - gradj

            self.weights_o = new_weights_o
        else:
            loss = None
        return out_o, loss


class colorizer:
    def __init__(self):
        self.model = model()
        self.import_data("./img/training/")

    def import_data(self, path):
        print("Importing training files:")
        X = np.empty((0, 9), int)
        Y = np.empty((0, 3), int)
        for filename in os.listdir(path):
            print(f"Importing {filename}...")
            im = Image.open(path+filename)
            im.load()
            rgb_data = np.asarray(im, dtype="int32")
            # print(rgb_data.shape)
            # exit()
            gray = colorizer.rgb_to_grayscale(rgb_data)
            # print(gray.shape)
            # exit()
            padded_gray = np.pad(gray, (1, 1), "constant", constant_values=0)
            # print(padded_gray.shape)
            # exit()
            i_size, j_size = np.shape(gray)
            im_X = np.empty((gray.size, 9), int)
            im_Y = np.empty((gray.size, 3), int)
            # print(im_X.shape)
            # print(im_Y.shape)
            # exit()
            index = 0
            # print(padded_gray[0:3,0:3])
            # print(rgb_data[0, 0, 0:3])
            # exit()
            # get 3x3 patches
            for i in range(i_size):
                for j in range(j_size):
                    im_X[index] = (padded_gray[i:i+3, j:j+3].flatten())
                    im_Y[index] = (rgb_data[i, j, 0:3])
                    index += 1
            X = np.concatenate((X, im_X))
            Y = np.concatenate((Y, im_Y))
            print(f"Imported {filename}: {gray.size} samples imported")
        self.X = np.array(X)
        self.Y = np.array(Y)

    @staticmethod
    def rgb_to_grayscale(data):
        grayscale = [0.21, 0.72, 0.07]
        gray = data[:, :, 0:3].dot(grayscale)
        gray = np.floor(gray)
        # gray = .3 * r + 0.55 * g + 0.15 * b
        grayimage = Image.fromarray(gray).convert("RGB")
        grayimage.save("gray.png")
        return gray


if __name__ == '__main__':
    test = colorizer()
    # print(test.X.shape)
    # print(test.Y.shape)
    # exit()
    trials = 1000
    epochs = 1000
    loss_history = []
    epoch_history = []

    for e in range(epochs):
        epoch_avg = 0
        for _ in range(trials):
            i = random.randint(0,341280-1)
            # print(f"trial#{_} using sample#{i}")
            output, loss = test.model.forward(test.X[i], test.Y[i], training=True)
            loss_history.append(loss)
            epoch_avg += loss/trials
        print(f"Avg loss for epoch#{e}: {epoch_avg}")
        epoch_history.append(epoch_avg)
    # print(test.model.forward(test.X[0], test.Y[0], training=True))

    y_pred = []
    for i in range(test.X.shape[0]):
            output= test.model.forward(test.X[i], test.Y[i], training=False)
            output = output[0]
            output = np.floor(output)
            y_pred.append(output)
            if i % 1000 is 1000:
                print(i)
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(474,720,3)
    np.clip(y_pred, 0, 255, out=y_pred)
    y_pred = y_pred.astype('uint8')
    result = Image.fromarray(y_pred).convert("RGB")
    # result.show()
    result.save("result.png")

    # fig = plt.plot(epoch_history[:])
    # plt.show()
    print("finished")
