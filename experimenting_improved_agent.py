import cv2
import sklearn.model_selection as sk
from utils import *
from improve import data_loader, Activation

np.random.seed(22)


# supplementary to improve.py
class Criterion:
    def __init__(self):
        self.error = None

    def get_constraint_loss(self, out, y, gray_out, gray_groundtruth, w, lamb=0.01, lamb2=0.001):
        """
        includes regularization but it didn't really help
        :param out: the predicted rgb value
        :param y: the label rgb value
        :param gray_out: the gray scale of out
        :param gray_groundtruth: the gray scale of y
        :param w: the weight of the layer
        :param lamb: param for level of constraining
        :param lamb2: for regularization
        :return:
        """
        # MSE loss with a neat trick for constraining values for the conversion from gray to rgb
        self.error = np.subtract(out, y)
        return (np.square(self.error) + lamb * (gray_out - gray_groundtruth) ** 2 + (lamb2 * np.linalg.norm(w))).mean()

    def get_loss_prime(self, x):
        x = x.reshape(x.shape[0],1)
        err = self.error.reshape(1,3)
        return 2 * np.dot(x, err)


class RayLayer:
    def __init__(self, weights, activation='sigmoid'):
        self.weights = weights
        self.act_prime = None
        self.og_input = None
        self.activation = Activation(type=activation)

    def forward(self, x):
        if x.shape == ():  # only for the first input gray pixel
            x = [x]  # in order to add the 1. as the bias term
        self.og_input = np.concatenate(([1.], x))  # bias term
        z = np.dot(self.og_input, self.weights)
        self.act_prime = self.activation.prime(z)
        return self.activation.activate(z)


class RayNet:
    def __init__(self, input_size, output_size, hidden_size, lr=0.03):
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        # source: https://stackoverflow.com/questions/22071987/generate-random-array-of-floats-between-a-range
        weight2 = np.random.uniform(low=-1/np.sqrt(input_size+1), high=1/np.sqrt(input_size+1), size=(input_size + 1, hidden_size))
        weight1 = np.random.uniform(low=-1/np.sqrt(hidden_size+1), high=1/np.sqrt(hidden_size+1), size=(hidden_size + 1, output_size))
        self.layer1 = RayLayer(weights=weight2, activation='sigmoid')  # in -> h
        self.layer2 = RayLayer(weights=weight1, activation='linear')  # h -> out
        self.criterion = Criterion()
        self.pre_activations = {}
        self.activation_z = None

    def forward(self, x):
        a1 = self.layer1.forward(x)  # feed into the first layer in --> h
        pre_activation_x = self.layer1.og_input
        self.pre_activations['a1'] = (pre_activation_x)
        pred = self.layer2.forward(a1)  # linear activation here for final layer for the regression problem
        pre_activation_out_h = self.layer2.og_input
        self.pre_activations['a2'] = (pre_activation_out_h)
        self.activation_z = pred
        return pred

    def backward(self, pred ,y):
        gray_out = np.dot(pred, [0.21, 0.72, 0.07])
        gray_ground_truth = np.dot(y, [0.21, 0.72, 0.07])
        loss = self.criterion.get_constraint_loss(pred, y, gray_out=gray_out, gray_groundtruth=gray_ground_truth, w=self.layer2.weights)
        grad_w2 = self.criterion.get_loss_prime(self.pre_activations['a2'])
        self.layer2.weights = self.layer2.weights - (self.lr * grad_w2)
        return loss

    def train(self, epochs, episodes, train_set, labels, validation_set, validation_labels):
        """
        Same as the one in improve.py but with validation set
        """
        losses = []
        validation_losses = []
        epoch_history = []
        for e in range(epochs):
            avg_loss = 0.0
            validation_loss = 0.0
            for t in range(episodes):
                rand_pixel = np.random.randint(0, (train_set.shape[0]) - 1)  # would be expensive to shuffle
                pred = self.forward(train_set[rand_pixel])
                loss = self.backward(pred, labels[rand_pixel])

                rand_pixel_val = np.random.randint(0, (validation_set.shape[0]) - 1)
                val_pred = self.forward(validation_set[rand_pixel_val])
                gray_val_pred = np.dot(val_pred, [0.21, 0.72, 0.07])
                gray_val_ground_truth = np.dot(validation_labels[rand_pixel_val], [0.21, 0.72, 0.07])
                validation_loss += self.criterion.get_constraint_loss(val_pred, validation_labels[rand_pixel_val], gray_val_pred, gray_val_ground_truth, self.layer2.weights)

                avg_loss += loss
            avg_loss /= episodes
            validation_loss /= episodes
            print("Running Avg Loss at epoch {}: {}".format(e, avg_loss))
            losses.append(avg_loss)
            validation_losses.append(validation_loss)
            epoch_history.append(e)
        return losses, validation_losses, epoch_history

    def evaluate(self, w, h, c, test_set, plot=False):
        reconstructed_image = []
        for i in range(w * h):
            pred = self.forward(test_set[i])
            reconstructed_image.append(pred)
        reconstructed_image = np.array(reconstructed_image)
        reconstructed_image = reconstructed_image.reshape((w, h, c))
        np.clip(reconstructed_image, 0, 255, out=reconstructed_image)
        reconstructed_image = reconstructed_image.astype('uint8')
        if plot:
            plot_img_color(reconstructed_image)
        return reconstructed_image


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
    X, Y, X_test = data_loader(train_gray, train_rgb, test_gray, w, h)

    # validation set for experiments
    X, X_val, Y, Y_val = sk.train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=22)
    # print(X.shape, Y.shape, X_val.shape, Y_val.shape)
    # exit()

    # hyper parameters
    INPUT_DIM = 1
    OUTPUT_DIM = 3
    LR=0.005
    # LR = 0.01  # ok
    EPOCH = 1000
    EPISODES = 10000
    # hidden=32  # pretty good
    # hidden = 64  # bad
    # hidden = 100  # better than 32
    HIDDEN_SIZE = 128  # better than 100
    # hidden = 256  # doesn't learn
    # hidden = 512  # doesn't learn

    model = RayNet(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE, lr=LR)
    losses, validation_losses, epochs = model.train(EPOCH, EPISODES, X, Y, X_val, Y_val)

    plt.xlabel('Epochs (10000 episodes per)')
    plt.ylabel('Loss of the validation and training data')
    plt.plot(epochs, losses, label="train")
    plt.plot(epochs, validation_losses, label="validation")
    plt.legend()
    plt.show()

    plot_img_color(train_rgb)
    test_rgb = model.evaluate(w, h, c, X_test, plot=True)
