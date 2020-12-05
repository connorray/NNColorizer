import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter

IMAGE_PATH = "img/small_train_img.jpg"


def make_gray(rgb_matrix):
    """
    Function to turn a colored image matrix into a matrix of the gray scaled values.
    :param rgb_matrix: the matrix of the rgb values for the image
    :return: matrix of gray scaled rgb values
    """
    gray_arr = np.copy(rgb_matrix)
    return np.dot(gray_arr, [0.21, 0.72, 0.07])


def cut_img_in_half(img):
    # https://stackoverflow.com/questions/45384968/how-to-cut-an-image-vertically-into-two-equal-sized-images
    # section = (gray_image_matrix[:len(gray_image_matrix)//2])  # this is the top half, need to go other way
    height, width = img.shape
    half = width // 2
    s1 = img[:, :half]
    s2 = img[:, half:]
    return s1, s2  # return the first half and the second half


def cut_rgb_img_half(img):
    height, width, depth = img.shape
    half = width // 2
    s1 = img[:, :half]
    s2 = img[:, half:]
    return s1, s2  # return the first half and the second half


def plot_img_color(rgb_matrix):
    # https://stackoverflow.com/questions/22777660/display-an-rgb-matrix-image-in-python
    plt.imshow(rgb_matrix)
    plt.show()


def plot_img_gray(gray_image_matrix):
    plt.imshow(gray_image_matrix, cmap=plt.get_cmap('gray'))
    plt.show()


def plot_scatter_rgb(rgb_matrix):
    # https://stackoverflow.com/questions/39885178/how-can-i-see-the-rgb-channels-of-a-given-image-with-python
    # https://stackoverflow.com/questions/19181485/splitting-image-using-opencv-in-python
    fig = plt.figure()
    r, g, b = rgb_matrix[:, :, 0], rgb_matrix[:, :, 1], rgb_matrix[:, :, 2]
    ax = Axes3D(fig)
    ax.scatter(r, g, b, s=0.1, c="blue")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    fig.set_size_inches(14, 8)
    plt.show()


def get_unique_colors(colors):
    """
    Don't know if i will need this, but it returns the unique colors in an image.
    :param colors: rgb matrix
    :return: np.array object
    """
    rgbs = []
    for i in range(colors.shape[0]):
        for j in range(colors.shape[1]):
            rgbs.append(colors[i][j])
    rgbs = np.array(rgbs)
    return np.unique(rgbs, axis=0)


def kmeans(k, image, dataset, max_ita=10, plot=False):
    """
    Run K-Means to get the k best representative colors
    :param k: the number of clusters that we want to consider
    :param image: the matrix of colors in the original image, each data point is 3-dimensional
    :return: the k colors that we want to use to represent the original image's colors, c1,c2,...,ck
    """
    np.random.seed(1)  # reproduce
    centroids = []
    for _ in range(k):
        centroids.append(dataset[np.random.randint(0, dataset.shape[0])])  # choose centers from dataset itself
    centroids = np.array(centroids)
    for run in range(max_ita):
        clusters = {}
        for i in range(k):
            clusters[i] = []  # initialize the clusters data structure
        for data in dataset:
            distance = []
            for j in range(k):
                distance.append(np.linalg.norm(data - centroids[j]))  # Euclidian distance between data point and center
            clusters[np.argmin(distance)].append(data)  # choose the arg 0,k-1 where the datapoint was closest to center
        for i in range(k):
            centroids[i] = np.average(clusters[i], axis=0)  # new center averaging the points in the cluster

    labels = np.zeros(len(dataset), dtype=int)  # label which cluster each datapoint in the dataset belongs
    for i in range(dataset.shape[0]):
        distances = [np.linalg.norm(dataset[i] - centroids[centroid]) for centroid in range(len(centroids))]
        labels[i] = np.argmin(distances)
    rep_img = representative_image(image, centroids, labels)
    if plot:
        plot_img_color(rep_img)
    return rep_img


def representative_image(image, centers, labels):
    """
    Make the image that is colored based on the representative colors chosen by kmeans
    :param image: the original image
    :param centers: the centroids computed by kmeans
    :param labels: the cluster that each data point in the image should belong to
    :return:
    """
    original_shape = image.shape
    image = image.reshape((-1, 3))
    rep_img = np.zeros(image.shape, dtype=int)
    for i in range(len(image)):
        cluster = labels[i]
        rep_img[i] = centers[cluster]  # 0-4 here for k=5
    rep_img.shape = original_shape
    return rep_img


def get_colors(data):
    """
    Grab all the present colors in an image in a list form
    :param data: the image
    :return: list of (r,g,b) values
    """
    dataset = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dataset.append(data[i][j])
    dataset = np.array(dataset)
    return dataset


def get_3_patch(data, i, j):
    """
    Helper function to return a 3x3 patch of given data at a given index.
    :param data: a given gray scale image matrix
    :return: a 3x3 numpy array
    """
    patch = np.zeros([3, 3], dtype=float)
    len, width = data.shape[0], data.shape[1]
    if i == 0 and j == 0:
        # this means that we are at the border and don't want to consider some indices in the patch
        patch[1][1] = data[i][j]
        patch[2][1] = data[i + 1][j]
        patch[1][2] = data[i][j + 1]
        patch[2][2] = data[i + 1][j + 1]
        return patch
    if i == 0 and j == width - 1:
        patch[1][0] = data[i][j - 1]
        patch[2][0] = data[i + 1][j - 1]
        patch[1][1] = data[i][j]
        patch[2][1] = data[i + 1][j]
        return patch
    if i == len - 1 and j == width - 1:
        patch[0][0] = data[i - 1][j - 1]
        patch[1][0] = data[i][j - 1]
        patch[0][1] = data[i - 1][j]
        patch[1][1] = data[i][j]
        return patch
    if i == 0:
        patch[1][0] = data[i][j - 1]
        patch[2][0] = data[i + 1][j - 1]
        patch[1][1] = data[i][j]
        patch[2][1] = data[i + 1][j]
        patch[1][2] = data[i][j + 1]
        patch[2][2] = data[i + 1][j + 1]
        return patch
    if i == len - 1:
        patch[0][0] = data[i - 1][j - 1]
        patch[1][0] = data[i][j - 1]
        patch[0][1] = data[i - 1][j]
        patch[1][1] = data[i][j]
        patch[0][2] = data[i - 1][j + 1]
        patch[1][2] = data[i][j + 1]
        return patch
    if j == 0:
        patch[0][1] = data[i - 1][j]
        patch[1][1] = data[i][j]
        patch[2][1] = data[i + 1][j]
        patch[0][2] = data[i - 1][j + 1]
        patch[1][2] = data[i][j + 1]
        patch[2][2] = data[i + 1][j + 1]
        return patch
    if j == width - 1:
        # this means that we are at the border and don't want to consider some indices in the patch
        patch[0][0] = data[i - 1][j - 1]
        patch[1][0] = data[i][j - 1]
        patch[2][0] = data[i + 1][j - 1]
        patch[0][1] = data[i - 1][j]
        patch[1][1] = data[i][j]
        patch[2][1] = data[i + 1][j]
        return patch
    patch[0][0] = data[i - 1][j - 1]
    patch[1][0] = data[i][j - 1]
    patch[2][0] = data[i + 1][j - 1]
    patch[0][1] = data[i - 1][j]
    patch[1][1] = data[i][j]
    patch[2][1] = data[i + 1][j]
    patch[0][2] = data[i - 1][j + 1]
    patch[1][2] = data[i][j + 1]
    patch[2][2] = data[i + 1][j + 1]
    return patch


def get_rep_color_from_patch(train_rgb, i, j):
    return tuple(train_rgb[i][j])


def basic_agent_logic(test_data, train_data_gray, train_data_rgb):
    """
    Get every 3x3 patch in data
    :param test_data: input grayscale image
    :return:
    """
    rep_test_img = np.zeros(train_data_rgb.shape, dtype=int)  # to return
    for i in range(0, (test_data.shape[0])):
        for j in range(0, (test_data.shape[1])):
            # we grab a 3x3 patch of the gray scale data each iteration
            test_patch = get_3_patch(test_data, i, j)
            # get the 6 most similar patches from the train data
            top_6_dict = get_6_similar_patches(test_patch, train_data_gray)
            # use these similar patches to get a representative color for the test data
            rep_colors = []
            for (index), value in top_6_dict.items():
                # similar_patch = get_3_patch(train_data_gray, index[0], index[1])  # one of the most similar patches
                # from these most similar patches, extract the representative color based on the kmeans train rgb image
                # we want the middle pixel of the index that the dictionary contains
                colored_of_patch = get_rep_color_from_patch(train_data_rgb, index[0], index[1])
                rep_colors.append(colored_of_patch)
            # get the most common color if there is a most common one, otherwise use the value in the dict to
            # see which patch of the six are the most similar to the test data patch and use that middle pixel
            colors = {}
            for color in rep_colors:
                if color in colors:
                    colors[color] += 1
                else:
                    colors[color] = 1
            (most_common_rep_color) = max(colors.keys(), key=lambda k: colors[k])  # get the highest count of color
            # search if this most common color's count appear more than once
            tie = False
            for key, count in colors.items():
                if key == most_common_rep_color:
                    continue
                if count == colors[most_common_rep_color]:
                    tie = True
                    break
            if tie is False:
                rep_test_img[i, j] = list(most_common_rep_color)
            else:
                # choose min value from the dictionary of top 6 patches as the representative patch for the test data
                (most_similar_train_patch) = min(top_6_dict.keys(), key=lambda k:top_6_dict[k])
                rep_test_img[i,j] = get_rep_color_from_patch(train_data_rgb, most_similar_train_patch[0], most_similar_train_patch[1])
            print(rep_test_img)
    return rep_test_img


def get_6_similar_patches(patch, train_data_gray):
    """
    For the given patch, search the gray train data for the 6 most similar patches. The metric for similarity will
    be the distance between the two patch matrices.

    :param patch: given patch from the test data
    :param train_data_gray: the left half of the original image in gray scale
    :return: a dictionary which maps the indices of the minimum distances (most similar) patches in the train data
                to the given patch we are comparing to:
                                                            of form: {(i, j) : float}
    """
    distances = {}  # have a dictionary mapping the index of the patch to the similarity of the train patch at
    # the index to the given patch that we are comparing to
    # store all the distances for each patch of the train data compared to the input patch from the test data
    for i in range(0, (train_data_gray.shape[0])):
        for j in range(0, (train_data_gray.shape[1])):
            train_patch = get_3_patch(train_data_gray, i, j)
            distance = get_cumulative_distance(patch, train_patch)
            distances[(i, j)] = distance
    # need the top 6 most similar patches from the train patches compared to the original patch from the test data
    K = 6
    # Source: https://www.geeksforgeeks.org/python-smallest-k-values-in-dictionary/
    res = dict(sorted(distances.items(), key=itemgetter(1))[:K])
    return res


def get_cumulative_distance(m1, m2):
    """
    Get the cumulative Euclidian distance between two matrices
    :param m1: input matrix1
    :param m2: input matrix2
    :return: float
    """
    total_distance = 0.0
    for i in range(len(m1)):
        for j in range(len(m2)):
            distance = np.linalg.norm(m1[i][j] - m2[i][j])
            total_distance += distance
    return total_distance


def kmeans_elbow_helper(k, dataset):
    """
    Run K-Means to get the k best representative colors
    :param k: the number of clusters that we want to consider
    :param image: the matrix of colors in the original image, each data point is 3-dimensional
    :return: the k colors that we want to use to represent the original image's colors, c1,c2,...,ck
    """
    np.random.seed(1)  # reproduce
    centroids = []
    for _ in range(k):
        centroids.append(dataset[np.random.randint(0, dataset.shape[0])])  # choose centers from dataset itself
    distances = []
    for run in range(1):
        centroids = np.array(centroids)
        clusters = {}
        for i in range(k):
            clusters[i] = []  # initialize the clusters data structure
        for data in dataset:
            distance = []
            for j in range(k):
                distance.append(np.linalg.norm(data - centroids[j]))  # Euclidian distance between data point and center
            clusters[np.argmin(distance)].append(data)  # choose the arg 0,k-1 where the datapoint was closest to center
            distances.append(np.min(distance))
        for i in range(k):
            centroids[i] = np.average(clusters[i], axis=0)  # new center averaging the points in the cluster
    return np.average(distances, axis=0)


def plot_elbow_kmeans(dataset):
    # Source: https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    # the above source is how I smoothed the elbow plot
    ks = [k for k in range(1, 100)]
    avg_distances_to_nearest_center = []
    for k in ks:
        avg_distances_to_nearest_center.append(kmeans_elbow_helper(k, dataset))
    # plt.xticks(np.arange(1,29, step=1))
    plt.ylabel("Avg Distance to Nearest Center")
    plt.xlabel("K")
    plt.plot(ks, avg_distances_to_nearest_center)
    plt.show()


if __name__ == '__main__':
    K = 59
    original_input_image = mpimg.imread(IMAGE_PATH)
    rgb_matrix = np.array(original_input_image)  # these are all of the RGB vectors needed
    gray_image_matrix = make_gray(rgb_matrix)  # these are all the gray scale vectors needed
    left_half_gray, right_half_gray = cut_img_in_half(gray_image_matrix)
    left_half_rgb, _ = cut_rgb_img_half(rgb_matrix)
    train_data_rgb = left_half_rgb
    train_data_gray = left_half_gray
    test_data = right_half_gray  # test data is just the gray correspondence of the image

    dataset = get_colors(train_data_rgb)
    # just plotting this to see if we have a reasonable reconstruction learned
    # dataset_entire_img = get_colors(rgb_matrix)
    # representative_entire_image = kmeans(K, rgb_matrix, dataset_entire_img, plot=True)
    train_rgb_5_colors = kmeans(K, train_data_rgb, dataset, plot=False)
    rep_test_img = basic_agent_logic(test_data, train_data_gray, train_rgb_5_colors)
    plot_img_color(rep_test_img)
    # plot_elbow_kmeans(dataset)
