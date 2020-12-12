from utils import *
import cv2


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

    def get_loss_prime(self, out, y):
        return 2 * np.dot(out, y)


class RayLayer:
    def __init__(self, weights, activation='sigmoid'):
        self.weights = weights
        self.act_prime = None
        self.og_input = None
        self.activation = Activation(type=activation)

    def forward(self, x):
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
        weight3 = np.random.randn(input_size+1, hidden_size[0])*0.02-0.1 #* (1 / np.sqrt(hidden_size[0] + 1))
        # weight2 = np.random.randn(hidden_size[0] + 1, hidden_size[1])*0.02-0.01 # * (1 / np.sqrt(hidden_size[1] + 1))
        weight1 = np.random.randn(hidden_size[0] + 1, output_size) * (1 / np.sqrt(output_size + 1))
        layer1 = RayLayer(weights=weight3)  # in -> h1
        # layer2 = RayLayer(weights=weight2)  # h1 -> h2
        self.layer3 = RayLayer(weights=weight1)  # h2 -> out
        # self.sequential = [layer1, layer2]  # layer3 last
        self.sequential = [layer1]
        self.criterion = Criterion()

    def forward(self, x, y, training=False):
        for layer in self.sequential:
            x = layer.forward(x)
        x_pad = np.concatenate(([1.], x))
        pred = np.dot(x_pad, self.layer3.weights)

        if training:
            loss = self.criterion.get_loss(pred, y)
            loss_prime = self.criterion.get_loss_prime(x_pad.reshape(self.hidden_size[0]+1,1), self.criterion.error.reshape(1,3))
            self.layer3.weights = self.layer3.weights - (self.lr * loss_prime)

            g_prime = self.sequential[0].activation.prime(x_pad)#.act_prime
            back = np.dot(self.layer3.weights, self.criterion.error)
            back *= g_prime
            loss_prime = self.criterion.get_loss_prime(
                self.sequential[0].og_input.reshape(self.sequential[0].weights.shape[0], 1),
                back[1:].reshape(1, self.sequential[0].weights.shape[1]))
            self.sequential[0].weights = self.sequential[0].weights - (self.lr * loss_prime)

            # g_prime = self.sequential[0].activation.prime(self.sequential[1].og_input) #.act_prime
            # back = np.dot(self.sequential[1].weights, back[1:])
            # back *= g_prime
            # loss_prime = self.criterion.get_loss_prime(self.sequential[0].og_input.reshape(self.sequential[0].weights.shape[0], 1),
            #                                            back[1:].reshape(1, self.sequential[0].weights.shape[1]))
            # self.sequential[0].weights = self.sequential[0].weights - (self.lr * loss_prime)

            # for layer in (self.sequential):
            #     g_prime = layer.activation.prime(x_pad)
            #     print(g_prime.shape)
            #     print(layer.weights.shape)
            #     print(self.criterion.error.shape)
            #     back = np.dot(layer.weights, self.criterion.error)
            #     back *= g_prime
            #     loss_prime = self.criterion.get_loss_prime(layer.og_input.reshape(layer.weights.shape[0],1), back[1:].reshape(1, layer.weights.shape[1]))
            #     layer.weights = layer.weights - (self.lr * loss_prime)
            return pred, loss
        return pred, None


def data_loader(rgb_img, gray_img, w, h, input_size, output_size):
    # get the 3x3 patches and get the data in the correct shape and what not
    X, Y = np.zeros((w*h, input_size), int), np.zeros((w*h, output_size), int)
    ind = 0
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
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

    input = 9
    output = 3
    # hidden = [32,64]
    alpha=0.01
    hidden=[64]
    X, Y = data_loader(train_rgb, train_gray, w, h, input, output)
    # print(X)
    # print(Y)
    # exit()

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
        print("Running Avg Loss at epoch {}: {}".format(avg_loss, e))
        if e % 100 == 0:
            # print([net.sequential[i].weights for i in range(2)])
            print(net.sequential[0].weights)
            print(net.layer3.weights)
        epochs.append(e)

    y_pred = []
    X_test = test_data_loader(test_gray, w, h, input)
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
