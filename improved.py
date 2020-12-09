from utils import *
import cv2


def relu(z):
    return max(z, 0)


def sigmoid(z):
    return 1/(np.exp(-z)+1)


def loss(y, y_prime):
    n = y_prime.shape[1]
    return (1/(2*n)) * np.sum((y - y_prime)**2)

def loss_prime(y, y_prime):
    return y - y_prime


class RayLayer:
    def __init__(self, num_inputs, num_neurons):
        # LeCun, Y., Bottou, L., Orr, G. B., and Muller, K. (1998a). Efficient backprop. In Neural Networks, Tricks of the Trade.
        # scaling the random weights by the inverse of the sqrt of the fan-in
        self.weights = np.random.randn(num_inputs, num_inputs) * (1/np.sqrt(num_inputs+1))
        self.biases = np.zeros((1, num_neurons))

    def forward(self, x):
        self.output = np.dot(x, self.weights) + self.biases


class RayNet:
    def __init__(self, input_size, output_size, hidden_size, lr=0.03):
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer1 = RayLayer(input_size, hidden_size)
        self.layer2 = RayLayer(hidden_size, output_size)
        self.weights = [self.layer1.weights, self.layer2.weights]
        self.biases = [self.layer1.biases, self.layer2.biases]

    def forward(self, x):
        pre_activations = {}
        z1 = self.layer1.forward(x)
        pre_activations['z1'] = z1
        a1 = relu(z1)
        pre_activations['a1'] = a1
        z2 = self.layer2.forward(x)
        pre_activations['z2'] = z2
        a2 = sigmoid(z2)
        pre_activations['a2'] = a2
        return a2, pre_activations

    def backward(self, x, y, y_prime, pre_activations):
        a1 = pre_activations['a1']
        a2 = pre_activations['a2']
        dz2 = a2 - y
        self.weights[0] = self.weights[0] - self.lr
        self.weights[1] = self.weights[1] - self.lr



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
