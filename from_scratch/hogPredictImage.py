# predict using SVM SGD from scratch model
import os, sys, shutil
sys.path.append('from_scratch')
import numpy as np
import cv2
from hogExtractor import hogDescriptorScratch, hogDescriptorSkimage

import sklearn
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from imutils import paths
import argparse
import time
import pickle
from linearClassifier import LinearSVM
# TODO: training with NO-SIGN image (negative image) large amount

defaultTestImage = r'from_scratch\images\test_3\3\1341.png'
testPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\test'
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--imgPath", required=False, default=defaultTestImage, help="Path to test image")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
print("[INFO] extracting features...")

imagePath = args["imgPath"]
 
	# load the image, convert it to grayscale, and detect edges
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (100, 100))
 
# extract Histogram of Oriented Gradients from the logo
# hogFeature = hogDescriptorSkimage(gray, 
# 								orientations=9, 
# 								pixels_per_cell=(8, 8),
# 								cells_per_block=(2, 2), 
# 								transform_sqrt=True, 
# 								visualize=False,
# 								block_norm='L2')
hogFeature = hogDescriptorScratch(gray, 
								    orientations=9, 
									cell_size=(8,8), 
									cells_per_block=(2,2),
									block_norm='L2',
									visualize=False)

data = np.stack(hogFeature, axis=0)
data = np.expand_dims(data, axis=0)
print(data.shape)
# data = np.hstack([data, np.ones((1, 1))])


print("[INFO] loading classifier...")
# svm = LinearSVM()

# svm = svm.SVC(C=10.0, kernel='rbf', verbose=False, gamma=0.005)
svm = SGDClassifier(learning_rate='optimal', loss='modified_huber', penalty='l2', alpha=1e-5, max_iter=5000, verbose=False, n_jobs=8, tol=1e-3)

modelPath1 = 'model/SGD-SVM-Sklearn-8-class.sav'
# print(data.shape[0])
# svm.load_weights(row = data.shape[1], col = 8, path = modelPath1)

# modelPath2 = 'model/mini-zalo-data-7class-scratch.sav'
svm = pickle.load(open(modelPath1, 'rb'))
# svm = CalibratedClassifierCV(svm)
start = time.time()

label = svm.predict(data.reshape((1,-1)))
score = svm.predict_proba(data.reshape((1,-1)))
# score = svm.decision_function(data.reshape((1,-1)))
# print(score)
# svm.fit(data, label)
# print(svm.predict_proba(data.reshape((1,-1))))[:,1]
print("[INFO] predicting time: ", time.time()-start)

print(label)
print('confident score: ', max(score[0]))

"""
mini zalo data
0: cam nguoc chieu
1: cam dung va do
2: cam re
3: gioi han toc do
4: cam khac
5: nguy hiem
6: hieu lenh
"""