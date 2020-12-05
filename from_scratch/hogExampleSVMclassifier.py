import os, sys, shutil
sys.path.append('from_scratch')
import numpy as np
import cv2
from from_scratch.hogExtractor import hogDescriptorScratch, hogDescriptorSkimage

import sklearn
from sklearn import svm
from sklearn import metrics
from imutils import paths
import argparse
import time
import pickle

# construct the argument parse and parse command line arguments
trainPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\train'
testPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\test'
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=False, default=trainPath, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=False, default=testPath, help="Path to the test dataset")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
print("[INFO] extracting features...")
data1 = []
data2 = []
labels = []

for imagePath in paths.list_images(args["training"]):

	make = imagePath.split("\\")[-2]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the logo
	hogFeatureSkimage = hogDescriptorSkimage(gray, 
											orientations=9, 
											pixels_per_cell=(8, 8),
											cells_per_block=(2, 2), 
											transform_sqrt=True, 
											visualize=False,
											block_norm='L2')
	hogFeatureScratch = hogDescriptorScratch(gray, 
											orientations=9, 
											cell_size=(8,8), 
											cells_per_block=(2,2),
											block_norm='L2',
											visualize=False)
	# update the data and labels
	# print(H.shape)
	data1.append(hogFeatureSkimage)
	data2.append(hogFeatureScratch)
	labels.append(make)

data1 = np.stack(data1, axis=0)
data2 = np.stack(data2, axis=0)
labels = np.stack(labels, axis=0)
print(data1.shape)
print(data2.shape)
print(labels.shape)

# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
model1 = svm.SVC(C=10.0, kernel='rbf', verbose=False, gamma=0.005)
model2 = svm.SVC(C=10.0, kernel='rbf', verbose=False, gamma=0.005)

model1.fit(data1, labels)
model2.fit(data2, labels)

print("[INFO] saving model...")
modelPath1 = 'model/mini-zalo-data-7class-skimage.sav'
modelPath2 = 'model/mini-zalo-data-7class-scratch.sav'
pickle.dump(model1, open(modelPath1, 'wb'))
pickle.dump(model2, open(modelPath2, 'wb'))

print("[INFO] evaluating...")

data1 = []
data2 = []
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
	hogFeatureSkimage = hogDescriptorSkimage(gray, 
											orientations=9, 
											pixels_per_cell=(8, 8),
											cells_per_block=(2, 2), 
											transform_sqrt=True, 
											visualize=False,
											block_norm='L2')
	hogFeatureScratch = hogDescriptorScratch(gray, 
											orientations=9, 
											cell_size=(8,8), 
											cells_per_block=(2,2),
											block_norm='L2',
											visualize=False)
	
	data1.append(hogFeatureSkimage)
	data2.append(hogFeatureScratch)
	labels.append(make)
 
data1 = np.stack(data1, axis=0)
data2 = np.stack(data2, axis=0)
labels = np.stack(labels, axis=0) 

print('Test accuracy on Skimage HoG extractor', model1.score(data1, labels))
print('Test accuracy on Scratch HoG extractor', model2.score(data2, labels))

print(metrics.confusion_matrix(model1.predict(data1), labels))
print(metrics.confusion_matrix(model2.predict(data2), labels))