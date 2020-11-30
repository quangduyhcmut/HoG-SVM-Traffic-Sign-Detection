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

TRAIN_PATH = r'from_scratch\train'
TEST_PATH  = r'from_scratch\test'

"""
- input feature SVM: numpy.ndarray shape [N, features] free range (normalize?) float 64 example [-71.64189796 -73.98173469 -69.47391837 ... -33.86195918 -42.39957143] 
- input label SVM: numpy.ndarray shape [N, label] int 32. Example [6 9 9 ... 4 9 3]
- Training data shape:  (49000, 3072)
- Validation data shape:  (1000, 3072)
- Test data shape:  (1000, 3072)
"""
# initialize the data matrix and labels
print("[INFO] extracting features...")
data = []
imgs = []
labels = []
classes = ['cam dung va do', 'cam nguoc chieu', 'cam re']
num_classes = len(classes)

for imagePath in paths.list_images(TRAIN_PATH):
    # extract the make of the car

    make = imagePath.split("\\")[-2]
    make = classes.index(make)

    # load the image, convert it to grayscale, and detect edges
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    gray = cv2.resize(gray, (100, 100))
    
    imgs.append(gray)
       
    # extract Histogram of Oriented Gradients from the logo
    # H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
    H = hog_scratch(gray, 
            cell_size=4, 
            block_size=2, 
            bins=9)
    # update the data and labels
    data.append(H)
    labels.append(make)

X_train = np.stack(data, axis=0)
y_train = np.stack(labels, axis=0)

# for imagePath in paths.list_images(TEST_PATH):
#     # extract the make of the car

#     make = imagePath.split("\\")[-2]
#     make = classes.index(make)

#     # load the image, convert it to grayscale, and detect edges
#     image = cv2.imread(imagePath)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#     gray = cv2.resize(gray, (100, 100))
       
#     # extract Histogram of Oriented Gradients from the logo
#     # H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
#     H = hog_scratch(gray, 
#             cell_size=4, 
#             block_size=2, 
#             bins=9)
#     # update the data and labels
#     data.append(H)
#     labels.append(make)

# X_test = np.stack(data, axis=0)
# y_test = np.stack(labels, axis=0)

print ('Training data shape: ', X_train.shape)
print ('Training labels shape: ', y_train.shape)
# print ('Test data shape: ', X_test.shape)
# print ('Test labels shape: ', y_test.shape)

samples_per_class = 5
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(imgs[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

"""
# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', y_val.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
"""
# As a sanity check, print out the shapes of the data
print ('Training data shape: ', X_train.shape)
# print ('Validation data shape: ', X_val.shape)
# print ('Test data shape: ', X_test.shape)
# print ('dev data shape: ', X_dev.shape)


# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
# X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
# X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
# X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
print (X_train.shape)
# print (X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

from cs231n.classifiers.linear_svm import svm_loss_naive
import time

# # generate a random SVM weight matrix of small numbers
# W = np.random.randn(3073, 10) * 0.0001 

# loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.00001)
# print ('loss: %f' % (loss, ))

# # Once you've implemented the gradient, recompute it with the code below
# # and gradient check it with the function we provided for you

# # Compute the loss and its gradient at W.
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)

# # Numerically compute the gradient along several randomly chosen dimensions, and
# # compare them with your analytically computed gradient. The numbers should match
# # almost exactly along all dimensions.
# from cs231n.gradient_check import grad_check_sparse
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
# grad_numerical = grad_check_sparse(f, W, grad)

# # do the gradient check once again with regularization turned on
# # you didn't forget the regularization gradient did you?
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 1e2)
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 1e2)[0]
# grad_numerical = grad_check_sparse(f, W, grad)

# # Next implement the function svm_loss_vectorized; for now only compute the loss;
# # we will implement the gradient in a moment.
# tic = time.time()
# loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.00001)
# toc = time.time()
# print ('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

# from cs231n.classifiers.linear_svm import svm_loss_vectorized
# tic = time.time()
# loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.00001)
# toc = time.time()
# print ('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# # The losses should match but your vectorized implementation should be much faster.
# print ('difference: %f' % (loss_naive - loss_vectorized))

# # Complete the implementation of svm_loss_vectorized, and compute the gradient
# # of the loss function in a vectorized way.

# # The naive implementation and the vectorized implementation should match, but
# # the vectorized version should still be much faster.
# tic = time.time()
# _, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.00001)
# toc = time.time()
# print ('Naive loss and gradient: computed in %fs' % (toc - tic))

# tic = time.time()
# _, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.00001)
# toc = time.time()
# print ('Vectorized loss and gradient: computed in %fs' % (toc - tic))

# # The loss is a single number, so it is easy to compare the values computed
# # by the two implementations. The gradient on the other hand is a matrix, so
# # we use the Frobenius norm to compare them.
# difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
# print ('difference: %f' % difference)

# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.

# print(type(X_train))
# print(type(y_train))
# print(X_train.dtype)
# print(y_train.dtype)
# print(X_train[0])
# print(y_train)

from cs231n.classifiers import LinearSVM
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=5e-7, reg=5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print ('That took %fs' % (toc - tic))

svm.save_weights()

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(X_train)
print ('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
# y_val_pred = svm.predict(X_val)
# print ('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))

# print("[INFO] extracting testing features...")
# data = []
# imgs = []
# labels = []
# classes = ['cam dung va do', 'cam nguoc chieu', 'cam re']
# num_classes = len(classes)

# for imagePath in paths.list_images(TEST_PATH):

#     # load the image, convert it to grayscale
#     image = cv2.imread(imagePath)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
#     gray = cv2.resize(gray, (100, 100))

#     imgs.append(gray)
       
#     # extract Histogram of Oriented Gradients from the logo
#     # H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
#     H = hog_scratch(gray, 
#             cell_size=4, 
#             block_size=2, 
#             bins=9)
#     # update the data and labels
#     data.append(H)


# X_test= np.stack(data, axis=0)
# X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
# y_test_pred = svm.predict(X_test)

# samples_per_class = 5
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_test_pred == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(imgs[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()