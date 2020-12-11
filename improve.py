from utils import *
import cv2


def relu(z):
    return max(z, 0)


def sigmoid(z):
    return 1/(np.exp(-z)+1)


def get_loss(out, y):
    return (out-y), np.sum((out-y)**2)


class RayLayer:
    def __init__(self, fan_in, fan_out):
        # LeCun, Y., Bottou, L., Orr, G. B., and Muller, K. (1998a). Efficient backprop. In Neural Networks, Tricks of the Trade.
        # scaling the random weights by the inverse of the sqrt of the fan-in
        self.weights = np.random.randn(fan_in, fan_out) * (1 / np.sqrt(fan_in + 1))
        self.biases = np.random.randn(1, fan_out) * (1 / np.sqrt(fan_in + 1))  # how to init this?
        self.weights = np.concatenate((self.weights, self.biases))  # let's just treat biases as weights


    def forward(self, x):
        return np.dot(x, self.weights)


class RayNet:
    def __init__(self, input_size, output_size, hidden_size, lr=0.03):
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer1 = RayLayer(input_size, hidden_size)
        self.layer2 = RayLayer(hidden_size, output_size)
        self.weights = [self.layer2.weights, self.layer1.weights]
        # self.weight1 = np.random.randn(hidden_size+1, output_size) * 32
        # self.weight2 = np.random.randn(input_size+1, hidden_size) * 0.02 - 0.01
        # self.weights = [self.weight1, self.weight2]

    def forward(self, x, y, training=False):
        # hidden layer
        x = np.pad(x, (1,0), "constant", constant_values=1)
        # z1 = self.layer1.forward(x)
        z1 = np.dot(x, self.weights[1])
        a1 = sigmoid(z1)
        a1 = np.pad(a1, (1,0), "constant", constant_values=1)
        # output layer
        # z2 = self.layer2.forward(a1)
        z2 = np.dot(a1, self.weights[0])
        a2 = z2

        if training:
            # have to do in reverse
            # output_layer weights
            error, loss = get_loss(a2, y)
            w_new1 = self.weights[0] - self.loss_deriv(a1, error, self.hidden_size+1, self.output_size)

            # hidden layer weights
            temp = np.dot(self.weights[0], error)
            w_new2 = self.weights[1] - self.loss_deriv(x, temp[1:], self.input_size+1, self.hidden_size)

            self.weights[0] = w_new1
            self.weights[1] = w_new2
            return a2, loss
        else:
            return a2, None


    def loss_deriv(self, out, err, input, output):
        return 2*self.lr*(out.reshape(input,1)@err.reshape(1,output))


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

    dims = train_rgb.shape
    w, h, c = dims[0], dims[1], dims[2]

    input = 9
    output = 3
    hidden = 100
    X, Y = data_loader(train_rgb, train_gray, w, h, input, output)
    # print(X.shape)
    # print(Y.shape)
    # exit()

    net = RayNet(input, output, hidden, lr=0.01)
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
    plot_img_color(y_pred)
