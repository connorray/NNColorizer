from utils import *
import cv2


def relu(z):
    return max(z, 0)


def sigmoid(z):
    return 1/(np.exp(-z)+1)


class RayNet:
    def __init__(self, input_size, output_size, hidden_size, lr=0.03):
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.weights = []
        # +1 for the bias term in these weight initializations
        self.weights[0] = self.init_weights(self.input_size+1, self.hidden_size)
        self.weights[1] = self.init_weights(self.hidden_size+1, self.output_size)

    @staticmethod
    def init_weights(input, out):
        # LeCun, Y., Bottou, L., Orr, G. B., and Muller, K. (1998a). Efficient backprop. In Neural Networks, Tricks of the Trade.
        # scaling the random weights by the inverse of the sqrt of the fan-in
        return np.random.randn(input+1, out) * (1/np.sqrt(input+1))

    def forward(self, x):
        pre_activations = []
        activations = [x]
        for weight in self.weights:
            z = np.dot(weight, x)
            x = sigmoid(z)
            pre_activations.append(z)
            activations.append(x)
        return x, pre_activations, activations

    def backward(self, x, pre_activations, activations):
        error = (x - )

    def loss(self, y, y_prime):
        n = y_prime.shape[1]
        return (1/(2*n)) * np.sum((y - y_prime)**2)


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
