# predict using SVM SGD from scratch model
import os, sys, shutil
sys.path.append('compare/utils-scratch')
import numpy as np
import cv2

import sklearn
from skimage.feature import hog
from imutils import paths
import argparse
import time
import pickle
from svm_loss import svm_loss_naive, svm_loss_vectorized
from linearClassifier import LinearSVM

default_test_img = r'images\test-3-classes\3\858.png'

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--imgPath", required=False, default=default_test_img, help="Path to test image")
args = vars(ap.parse_args())
 
print("[INFO] extracting features...")
imagePath = args["imgPath"]
 
# load the image, convert it to grayscale, and detect edges
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (100, 100))
 
# extract Histogram of Oriented Gradients from the logo
hogFeature = hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(4, 4), transform_sqrt=True, visualize=False,block_norm='L2')

data = np.stack(hogFeature, axis=0)
data = np.expand_dims(data, axis=0)
data = np.hstack([data, np.ones((data.shape[0], 1))])
# print(data.shape)

print("[INFO] loading classifier...")

model_path = 'models\compare_scratch_model.sav'
svm = LinearSVM()
svm.load_weights(row = data.shape[1], col = 8, path = model_path)

start = time.time()

label = svm.predict(data.reshape((1,-1)))

print("[INFO] predicting time: ", time.time()-start)
labels_list = ['cam nguoc chieu', 'cam dung va do', 'cam re', 'gioi han toc do', 'cam khac', 'nguy hiem', 'hieu lenh', 'negative']
print('[INFO] Predicting result: ', labels_list[int(label)])


"""
mini zalo data
0: cam nguoc chieu
1: cam dung va do
2: cam re
3: gioi han toc do
4: cam khac
5: nguy hiem
6: hieu lenh
7: negative
"""