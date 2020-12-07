# SVM SGD hyperparameter search (1 negative)
# SVM SGD for multiclass classification from scratch training with 8 class (1 negative)
# TODO: add RBF kernel's params searching
import sys, os
sys.path.append(r'from_scratch')

import random
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import cv2
from hogExtractor import hogDescriptorScratch, hogDescriptorSkimage

from calSVMLoss import svm_loss_naive, svm_loss_vectorized
from linearClassifier import LinearSVM
import time
from sklearn import metrics

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plot
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

TRAIN_PATH = r'from_scratch\train_3'
TEST_PATH  = r'from_scratch\test_3'
trainPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\train'
testPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\test'

# initialize the data matrix and labels

print("[INFO] extracting features...")
data_train = []
labels_train = []

for imagePath in paths.list_images(trainPath):

	make = imagePath.split("\\")[-2]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the logo
	hogFeature = hogDescriptorScratch(gray, 
											orientations=9, 
											cell_size=(8,8), 
											cells_per_block=(2,2),
											block_norm='L2',
											visualize=False)
	# update the data and labels
	# print(H.shape)
	data_train.append(hogFeature)
	labels_train.append(int(make))

data_train = np.stack(data_train, axis=0)
labels_train = np.stack(labels_train, axis=0)
# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
data_train = np.hstack([data_train, np.ones((data_train.shape[0], 1))])

print("[INFO] extracting features...")
data_val = []
labels_val = []

for imagePath in paths.list_images(testPath):

	make = imagePath.split("\\")[-2]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the logo
	hogFeature = hogDescriptorScratch(gray, 
											orientations=9, 
											cell_size=(8,8), 
											cells_per_block=(2,2),
											block_norm='L2',
											visualize=False)
	# update the data and labels
	data_val.append(hogFeature)
	labels_val.append(int(make))

data_val = np.stack(data_val, axis=0)
labels_val = np.stack(labels_val, axis=0)

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
data_val = np.hstack([data_val, np.ones((data_val.shape[0], 1))])

lr = [1e-2]
r = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

for learning_rate in lr:
    for reg in r:
        svm = LinearSVM()
        # learning_rate = 0.00001
        # reg = 0.0001
        tic = time.time()
        loss_hist = svm.train(data_train, labels_train, learning_rate=learning_rate, reg=reg, num_iters=10000, verbose=False)
        toc = time.time()
        # print ('That took %fs' % (toc - tic))

        # svm.save_weights(path = 'model/SGD-SVM-scratch-8-class.sav')

        # A useful debugging strategy is to plot the loss as a function of
        # iteration number:
        plt.plot(loss_hist, label = 'lr = {} reg = {}'.format(learning_rate, reg))
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        # plt.show()
        plt.legend()
        plt.savefig(r'from_scratch\figures\lr-{}-reg-{}.png'.format(learning_rate, reg))

        # Write the LinearSVM.predict function and evaluate the performance on both the
        # training and validation set
        y_train_pred = svm.predict(data_train)
        print ('training acc lr = {} reg = {}'.format(learning_rate, reg) ,np.mean(labels_train == y_train_pred), )

        # predict
        predLabel = svm.predict(data_val)
        # print(predLabel)
        # print(labels)

        # print(metrics.confusion_matrix(predLabel, labels_val))
        print('valid acc lr = {} reg = {}'.format(learning_rate, reg), metrics.accuracy_score(labels_val, predLabel))