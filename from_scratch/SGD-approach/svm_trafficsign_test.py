import sys
sys.path.append('from_scratch')

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from imutils import paths
import cv2
from hog_example import hog_scratch, hog_skimage

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plot
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

TEST_PATH  = r'from_scratch\test'

"""
- input feature SVM: numpy.ndarray shape [N, features] free range (normalize?) float 64 example [-71.64189796 -73.98173469 -69.47391837 ... -33.86195918 -42.39957143] 
- input label SVM: numpy.ndarray shape [N, label] int 32. Example [6 9 9 ... 4 9 3]
- Training data shape:  (49000, 3072)
- Validation data shape:  (1000, 3072)
- Test data shape:  (1000, 3072)
"""
# initialize the data matrix and labels

from cs231n.classifiers.linear_svm import svm_loss_naive
import time

from cs231n.classifiers import LinearSVM

print("[INFO] extracting testing features...")
data = []
imgs = []
labels = []
classes = ['cam dung va do', 'cam nguoc chieu', 'cam re']
num_classes = len(classes)

for imagePath in paths.list_images(TEST_PATH):

    # load the image, convert it to grayscale
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    gray = cv2.resize(gray, (100, 100))

    imgs.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) )
       
    # extract Histogram of Oriented Gradients from the logo
    # H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
    H = hog_scratch(gray, 
            cell_size=4, 
            block_size=2, 
            bins=9)
    # update the data and labels
    data.append(H)


X_test= np.stack(data, axis=0)
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

svm = LinearSVM()
svm.load_weights(row = X_test.shape[1], col = num_classes)

y_test_pred = svm.predict(X_test)

samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_test_pred == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(imgs[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()