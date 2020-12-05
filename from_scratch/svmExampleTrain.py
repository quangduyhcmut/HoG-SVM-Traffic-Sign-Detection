# SVM SGD for multiclass classification from scratch training with 8 class (1 negative)
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

for imagePath in paths.list_images(trainPath):

	make = imagePath.split("\\")[-2]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the logo
	# hogFeature = hogDescriptorSkimage(gray, 
	# 										orientations=9, 
	# 										pixels_per_cell=(8, 8),
	# 										cells_per_block=(2, 2), 
	# 										transform_sqrt=True, 
	# 										visualize=False,
	# 										block_norm='L2')
	hogFeature = hogDescriptorScratch(gray, 
											orientations=9, 
											cell_size=(8,8), 
											cells_per_block=(2,2),
											block_norm='L2',
											visualize=False)
	# update the data and labels
	# print(H.shape)
	data.append(hogFeature)
	# data.append(hogFeatureScratch)
	labels.append(int(make))

data = np.stack(data, axis=0)
labels = np.stack(labels, axis=0)

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
data = np.hstack([data, np.ones((data.shape[0], 1))])
# print (data.shape)
# print(labels.shape)
# print(labels)

svm = LinearSVM()

tic = time.time()
loss_hist = svm.train(data, labels, learning_rate=0.00001, reg=0.0001, num_iters=25000, verbose=False)
toc = time.time()
print ('That took %fs' % (toc - tic))

svm.save_weights(path = 'model/SGD-SVM-scratch-8-class.sav')

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(data)
print ('training accuracy: %f' % (np.mean(labels == y_train_pred), ))


print("[INFO] extracting features...")
data = []
imgs = []
labels = []

for imagePath in paths.list_images(testPath):

	make = imagePath.split("\\")[-2]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the logo
	# hogFeature = hogDescriptorSkimage(gray, 
	# 										orientations=9, 
	# 										pixels_per_cell=(8, 8),
	# 										cells_per_block=(2, 2), 
	# 										transform_sqrt=True, 
	# 										visualize=False,
	# 										block_norm='L2')
	hogFeature = hogDescriptorScratch(gray, 
											orientations=9, 
											cell_size=(8,8), 
											cells_per_block=(2,2),
											block_norm='L2',
											visualize=False)
	# update the data and labels
	# print(H.shape)
	data.append(hogFeature)
	# data.append(hogFeatureScratch)
	labels.append(int(make))

data = np.stack(data, axis=0)
labels = np.stack(labels, axis=0)

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
data = np.hstack([data, np.ones((data.shape[0], 1))])

load_svm = LinearSVM()
load_svm.load_weights(row = data.shape[1], col = len(os.listdir(testPath)), path = r'model\svm-scratch.sav')
predLabel = load_svm.predict(data)
# print(predLabel)
# print(labels)

print(metrics.confusion_matrix(predLabel, labels))
print(metrics.accuracy_score(labels,predLabel))