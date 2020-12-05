import cv2

IMAGE_PATH = "img/small_train_img.jpg"
K = 5
from utils import *


if __name__ == '__main__':
    # step 1: take a single color image
    '''source: https://stackoverflow.com/questions/50963283/python-opencv-imshow-doesnt-need-convert-from-bgr-to-rgb'''
    original_img = cv2.imread(IMAGE_PATH)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # step 2: convert this color image to black and white
    original_img_gray = make_gray(original_img)
    # step 3: use the left half of the image as training data (both rgb and the black and white)
    train_rgb, _ = cut_rgb_img_half(original_img)
    # step 4: set aside the right half of the image as test data (black and white only)
    train_gray, test_gray = cut_img_in_half(original_img_gray)
    # step 5: try to recolor the test data based on the training data
    # step 5aa: set aside an array of the colors in the train data
    colors = get_colors(train_rgb)  # maybe we don't need this
    # step 5a: run k means on the rgb training data to get the 5 representative colors
    rep_train_rgb = kmeans(K, train_rgb, colors, plot=False)
    # step5b: set aside the recoloring of the rgb training data by replacing each pixel's true color with the colors
        # from step 5a.
    print(rep_train_rgb.shape)
    # step 5c: have a function to grab each 3x3 patch in the test data
        # for each patch in the test data, find the 6 most similar patches in the black and white training data
    rep_test_rgb = basic_agent_logic(test_gray, train_gray, rep_train_rgb)
    # step 5d: for each patch selected from 5c, take the representative color (r,g,b) of the middle pixel in the
        # recolored training data from 5b.

    # step 5e: amongst the representative colors for the given patch we are checking in step 5d (from the 6 patches),
        # take the majority representative color to be the color of the middle pixel in the patch for the test data

    # step 5f: if there is no majority representative color in 5e, break ties by choosing the representative color
        # from the patch in the train data's 6 patches most similar to the test data patch that we are checking

    # step 6: repeat steps 5c-5f until  we have recolored the middle pixel of each patch in the test data

    # step 7: plot the original image, the recolored left half, the recolored right half
    plot_img_color(original_img)
    plot_img_color(rep_train_rgb)
    plot_img_color(rep_test_rgb)