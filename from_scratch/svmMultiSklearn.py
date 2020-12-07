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
from sklearn.calibration import CalibratedClassifierCV
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
print("[INFO] extracting training features...")
data_train = []
labels_train = []

for imagePath in paths.list_images(args["training"]):

	make = imagePath.split("\\")[-2]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the logo
	hogFeature = hogDescriptorSkimage(gray, 
											orientations=9, 
											pixels_per_cell=(8, 8),
											cells_per_block=(2, 2), 
											transform_sqrt=True, 
											visualize=False,
											block_norm='L2')
	# hogFeature = hogDescriptorScratch(gray, 
	# 										orientations=9, 
	# 										cell_size=(8,8), 
	# 										cells_per_block=(2,2),
	# 										block_norm='L2',
	# 										visualize=False)
	# update the data and labels
	# print(H.shape)
	data_train.append(hogFeature)
	labels_train.append(make)

data_train = np.stack(data_train, axis=0)
data_ = data_train.copy()
labels_train = np.stack(labels_train, axis=0)
print(data_train.shape)
print(labels_train.shape)

print("[INFO] extracting validation feature...")

data_val = []
labels_val = []
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
	# load the test image, convert it to grayscale, and resize it to
	# the canonical size
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
	make = imagePath.split("\\")[-2]
 
	# extract Histogram of Oriented Gradients from the test image and
	# predict the make of the car
	hogFeature = hogDescriptorSkimage(gray, 
											orientations=9, 
											pixels_per_cell=(8, 8),
											cells_per_block=(2, 2), 
											transform_sqrt=True, 
											visualize=False,
											block_norm='L2')
	# hogFeature = hogDescriptorScratch(gray, 
	# 										orientations=9, 
	# 										cell_size=(8,8), 
	# 										cells_per_block=(2,2),
	# 										block_norm='L2',
	# 										visualize=False)
	
	data_val.append(hogFeature)
	labels_val.append(make)

data_val = np.stack(data_val, axis=0)
labels_val = np.stack(labels_val, axis=0) 

# define classifier

clf = SGDClassifier(learning_rate='optimal', loss='hinge', penalty='l2', alpha=1e-5, max_iter=1000, verbose=False, n_jobs=8, tol=1e-3)
# calibration for probability estimation
clf = CalibratedClassifierCV(clf)
clf.fit(data_train, labels_train)
print(clf.score(data_train, labels_train))

# "train" the nearest neighbors classifier

print("[INFO] saving model...")
modelPath = 'model/SGD-SVM-Sklearn-8-class.sav'
pickle.dump(clf, open(modelPath, 'wb'))  

print('Test accuracy on Skimage HoG extractor', clf.score(data_val, labels_val))
# print('Test accuracy on Scratch HoG extractor', model2.score(data2, labels))

print(metrics.confusion_matrix(clf.predict(data_val), labels_val))

# print(clf.predict_proba(data_val))
prob = clf.predict_proba(data_val)
for sc in prob:
    print('score list', sc)
    print('score', max(sc))
    print('predicted class', sc.copy().tolist().index(max(sc)))

# score = clf.decision_function(data)
# score = clf.predict_proba(data)
# for sc in score:
#     print('score list', sc)
#     print('max score', max(sc))
#     print('predicted class', sc.copy().tolist().index(max(sc)))

# training_score = clf.decision_function(data_)
# print(training_score.shape)
# average_score_list = [0,0,0,0,0,0,0,0]
# number_of_sample = [0,0,0,0,0,0,0,0]
# for sample_score in training_score:
#     label = sample_score.copy().tolist().index(max(sample_score))
#     average_score_list[label] += max(sample_score)
#     number_of_sample[label] += 1
# print(number_of_sample)

# for index, score in enumerate(average_score_list):
#     average_score_list[index] = score/number_of_sample[index]
    
# print(average_score_list)
# print('average score for each class: ', average_score_list)