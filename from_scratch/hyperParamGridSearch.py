import os, sys, shutil
sys.path.append('from_scratch')
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import cv2
from hogExtractor import hogDescriptorScratch, hogDescriptorSkimage
import numpy as np
from imutils import paths
import argparse
import time

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


param_grid = {'C': [1.0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              'kernel': ['rbf', 'linear', 'poly' ]}
SVC = svm.SVC(kernel='rbf', class_weight='balanced')
clf = GridSearchCV(SVC, param_grid, cv=4)
clf = clf.fit(data1, labels)
# print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

"""
Best estimator found by grid search:
SVC(C=10.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""
