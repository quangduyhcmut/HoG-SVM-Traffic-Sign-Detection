# SVM SGD for multiclass classification from sklearn training with 8 class (1 negative)
import os, sys, shutil
sys.path.append('from_scratch')
import numpy as np
import cv2
from hogExtractor import hogDescriptorScratch, hogDescriptorSkimage

import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from imutils import paths
import argparse
import time
import pickle

# construct the argument parse and parse command line arguments
trainPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\train'
testPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\test'
# trainPath = r'from_scratch\train_3'
# testPath = r'from_scratch\test_3'
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=False, default=trainPath, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=False, default=testPath, help="Path to the test dataset")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
print("[INFO] extracting features...")
data = []
labels = []

for imagePath in paths.list_images(args["training"]):

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
	labels.append(make)

data = np.stack(data, axis=0)
labels = np.stack(labels, axis=0)
print(data.shape)
print(labels.shape)

clf = SGDClassifier(learning_rate='optimal', loss='hinge', penalty='l2', alpha=1e-5, max_iter=5000, verbose=False, n_jobs=8, tol=1e-3)
a = clf.fit(data, labels)
print(clf.score(data, labels))

# "train" the nearest neighbors classifier

print("[INFO] saving model...")
modelPath = 'model/SGD-SVM-Sklearn-8-class.sav'
pickle.dump(clf, open(modelPath, 'wb'))

print("[INFO] evaluating...")

data = []
labels= []
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
	# load the test image, convert it to grayscale, and resize it to
	# the canonical size
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
	make = imagePath.split("\\")[-2]
 
	# extract Histogram of Oriented Gradients from the test image and
	# predict the make of the car
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
	
	data.append(hogFeature)
	labels.append(make)
 
data = np.stack(data, axis=0)
labels = np.stack(labels, axis=0) 

print('Test accuracy on Skimage HoG extractor', clf.score(data, labels))
# print('Test accuracy on Scratch HoG extractor', model2.score(data2, labels))

print(metrics.confusion_matrix(clf.predict(data), labels))
