from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Source: https://www.idtools.com.au/segmentation-k-means-python/
def km_clust(array, n_clusters):
    # Create a line array, the lazy way
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = cluster.KMeans(n_clusters=n_clusters, n_init=1)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_
    return (values, labels)


# Read the data as greyscale
img = mpimg.imread('./img/hangang.jpg')
# Group similar grey levels using 8 clusters
values, labels = km_clust(img, n_clusters=5)
# print(values)  # k values -- centers
# print(labels.shape)  #
# exit()
# Create the segmented array from labels and values
img_segm = np.choose(labels, values).astype('uint8')
# Reshape the array as the original image
img_segm.shape = img.shape
# Get the values of min and max intensity in the original image
vmin = img.min()
vmax = img.max()
fig = plt.figure(1)
# Plot the original image
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img, vmin=vmin, vmax=vmax)
ax1.set_title('Original image')
# Plot the simplified color image
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(img_segm, vmin=vmin, vmax=vmax)
ax2.set_title('Simplified levels')
# Get rid of the tick labels
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
plt.show()