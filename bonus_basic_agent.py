from utils import *  # all implementation logic is in utils
import cv2


if __name__ == '__main__':
    IMAGE_PATH = "img/small_train_img.jpg"
    K = 59
    original_img = cv2.imread(IMAGE_PATH)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img_gray = make_gray(original_img)
    train_rgb, _ = cut_rgb_img_half(original_img)
    train_gray, test_gray = cut_img_in_half(original_img_gray)
    colors = get_colors(train_rgb)
    rep_train_rgb = kmeans(K, train_rgb, colors, plot=False)
    K_N = int(np.floor(np.sqrt(train_rgb.shape[0] * train_rgb.shape[1])))
    # https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb#:~:text=The%20optimal%20K%20value%20usually,be%20aware%20of%20the%20outliers.
    rep_test_img = basic_agent_logic(test_gray, train_gray, rep_train_rgb, K=K_N)
    plot_img_color(rep_test_img)
    # plot_elbow_kmeans(dataset, range_k=500)