import cv2
from utils import *

np.random.seed(1)


# Let's try these:
# 1) linear model
def linear_regression():
    # train data
    X, Y = [], []
    w,h,c = train_rgb.shape[0], train_rgb.shape[1], train_rgb.shape[2]
    for i in range(w):
        for j in range(h):
            X.append(train_gray[i][j])
            Y.append(train_rgb[i][j])
    X, Y = np.array(X), np.array(Y)
    print(X.shape, Y.shape)

    b = 0
    np.random.seed(1)
    w1 = (np.random.random(1) * 0.1)[0]

    epoch = 1000
    episodes = 1000
    losses = []
    epochs = []
    lr = 0.01
    for e in range(epoch):
        avg_loss = 0.0
        for t in range(episodes):
            rand_pixel = np.random.randint(0, (w*h)-1)  # would be expensive to shuffle
            pred = w1*X[rand_pixel]+b
            print(pred, Y[rand_pixel])
            loss = ((1/(w*h)) * np.sum(pred-Y[rand_pixel])**2)
            dz1 = (1/(w*h)) * np.sum((pred-Y[rand_pixel]))
            w1 = w1 - (2*lr*dz1*X[rand_pixel])
            b += -lr * dz1
            losses.append(loss)
            avg_loss += loss
        avg_loss /= episodes
        # print("Running Avg Loss at epoch {}: {}".format(avg_loss, e))
        epochs.append(e)


class Activation:
    def __init__(self, type):
        self.type = type

    def activate(self, z):
        if self.type =='sigmoid':
            return self._sigmoid(z)
        elif self.type == 'relu':
            return self._relu(z)

    def prime(self, a):
        if self.type =='sigmoid':
            return self._sigmoid_prime(a)
        elif self.type == 'relu':
            return self._relu_derivative(a)

    def _sigmoid(self,z):
        return 1/(np.exp(-z)+1)

    def _relu(self, z):
        return np.maximum(z, 0)

    def _sigmoid_prime(self, a):
        """
        Return the derivative of the sigmoid activation function
        :param a: activation
        :return:
        """
        return a * (1-a)

    def _relu_derivative(self, a):
        """
        Fastest way to get relu derivative
        source: https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
        :param a: activation
        :return:
        """
        return np.greater(a, 0)


class Criterion:
    def __init__(self):
        self.error = None

    def get_loss(self, out, y):
        self.error = out-y
        return np.sum(self.error**2)

    def get_constraint_loss(self, out, y, gray_out, gray_groundtruth, lamb=0.01):
        return np.sum((out-y)**2) + lamb * (gray_out - gray_groundtruth) ** 2

    def get_loss_prime(self, out, y):
        return 2 * np.dot(out, y)


class RayLayer:
    def __init__(self, weights, activation='sigmoid'):
        self.weights = weights
        self.act_prime = None
        self.og_input = None
        self.activation = Activation(type=activation)

    def forward(self, x):
        x = [x]  # in order to add the 1. as the bias term
        self.og_input = np.concatenate(([1.], x))  # bias term
        h = np.dot(self.og_input, self.weights)
        self.act_prime = self.activation.prime(h)
        return self.activation.activate(h)


class RayNet:
    def __init__(self, input_size, output_size, hidden_size, lr=0.03):
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        # source: https://stackoverflow.com/questions/22071987/generate-random-array-of-floats-between-a-range
        weight2 = np.random.uniform(low=-1/np.sqrt(input_size+1), high=1/np.sqrt(input_size+1), size=(input_size + 1, hidden_size))
        # weight1 = np.random.randn(hidden_size + 1, output_size) * (1 / np.sqrt(output_size + 1))
        weight1 = np.random.uniform(low=-1/np.sqrt(hidden_size+1), high=1/np.sqrt(hidden_size+1), size=(hidden_size + 1, output_size))
        self.layer1 = RayLayer(weights=weight2, activation='sigmoid')  # in -> h1
        self.layer2 = RayLayer(weights=weight1)  # h2 -> out
        self.criterion = Criterion()

    def forward(self, x, y, training=False):
        x = self.layer1.forward(x)
        out_h = np.concatenate(([1.], x))
        pred = np.dot(out_h, self.layer2.weights)
        if training:
            loss = self.criterion.get_loss(pred, y)
            loss_prime = self.criterion.get_loss_prime(out_h.reshape(self.hidden_size+1,1),
                                                       self.criterion.error.reshape(1,3))
            self.layer2.weights = self.layer2.weights - (self.lr * loss_prime)

            g_prime = self.layer1.activation.prime(out_h)
            back = np.dot(self.layer2.weights, self.criterion.error)
            back *= g_prime
            loss_prime = self.criterion.get_loss_prime(
                self.layer1.og_input.reshape(self.layer1.weights.shape[0], 1),
                back[1:].reshape(1, self.layer1.weights.shape[1]))
            # self.layer1.weights = self.layer1.weights - (self.lr * loss_prime)

            return pred, loss
        return pred, None


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

    dims = train_rgb.shape
    w, h, c = dims[0], dims[1], dims[2]
    X, Y = [], []
    X_test = []
    for i in range(w):
        for j in range(h):
            X.append(train_gray[i][j])
            Y.append(train_rgb[i][j])
            X_test.append(test_gray[i][j])
    X, Y, X_test = np.array(X), np.array(Y), np.array(X_test)

    input = 1
    output = 3
    alpha=0.005
    # hidden=32  # pretty good
    # hidden = 64  # bad
    # hidden = 100  # better than 32
    hidden = 128  # better than 100
    # hidden = 512  # doesn't learn

    net = RayNet(input, output, hidden, lr=alpha)
    epoch = 1000
    episodes = 1000
    losses = []
    epochs = []

    for e in range(epoch):
        avg_loss = 0.0
        for t in range(episodes):
            rand_pixel = np.random.randint(0, (w*h)-1)  # would be expensive to shuffle
            pred, loss = net.forward(X[rand_pixel], Y[rand_pixel], training=True)
            losses.append(loss)
            avg_loss += loss
        avg_loss /= episodes
        avg_loss *= (1/(w*h))
        print("Running Avg Loss at epoch {}: {}".format(avg_loss, e))
        epochs.append(e)

    y_pred = []
    for i in range(X_test.shape[0]):
        out = net.forward(X_test[i], _, training=False)[0]
        out = np.floor(out)
        y_pred.append(out)
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape((w,h,c))
    np.clip(y_pred, 0, 255, out=y_pred)
    y_pred = y_pred.astype('uint8')
    print(y_pred)
    plot_img_color(y_pred)
