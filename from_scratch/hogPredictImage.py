import os, sys, shutil
sys.path.append('from_scratch')
import numpy as np
import cv2
from hogExtractor import hogDescriptorScratch, hogDescriptorSkimage

import sklearn
from sklearn import svm
from imutils import paths
import argparse
import time
import pickle

# TODO: training with NO-SIGN image (negative image) large amount
 
defaultTestImage = r'from_scratch\test_3\2\98.png'
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
hogFeature = hogDescriptorSkimage(gray, 
								orientations=9, 
								pixels_per_cell=(8, 8),
								cells_per_block=(2, 2), 
								transform_sqrt=True, 
								visualize=False,
								block_norm='L2')
# hogFeature = hogDescriptorScratch(gray, 
# 								    orientations=9, 
# 									cell_size=(8,8), 
# 									cells_per_block=(2,2),
# 									block_norm='L2',
# 									visualize=False)

data = np.stack(hogFeature, axis=0)

# "train" the nearest neighbors classifier
print("[INFO] loading classifier...")
# model = svm.SVC(C=10.0, kernel='rbf', verbose=False, gamma=0.005)
modelPath1 = 'model/mini-zalo-data-7class-skimage.sav'
# modelPath2 = 'model/mini-zalo-data-7class-scratch.sav'
model = pickle.load(open(modelPath1, 'rb'))
start = time.time()
label = model.predict(data.reshape((1,-1)))
print("[INFO] predicting time: ", time.time()-start)
print(label)

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