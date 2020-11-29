import os, sys, shutil
import numpy as np
import cv2
from hog_example import hog_scratch, hog_skimage

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
 
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
print("[INFO] extracting features...")
data = []
labels = []

for imagePath in paths.list_images(args["training"]):
	# extract the make of the car
	# print(imagePath.split("\\"))
	make = imagePath.split("\\")[-2]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# edged = imutils.auto_canny(gray)
 
	# # find contours in the edge map, keeping only the largest one which
	# # is presmumed to be the car logo
	# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)
	# c = max(cnts, key=cv2.contourArea)
 
	# # extract the logo of the car and resize it to a canonical width
	# # and height
	# (x, y, w, h) = cv2.boundingRect(c)
	# logo = gray[y:y + h, x:x + w]
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the logo
	H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
	# H = hog_scratch(gray, 
    #         cell_size=4, 
    #         block_size=2, 
    #         bins=9)
	# update the data and labels
	# print(H.shape)
	data.append(H)
	labels.append(make)

data = np.stack(data, axis=0)
labels = np.stack(labels, axis=0)
print(data.shape)
print(labels.shape)


# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
print("[INFO] evaluating...")

for (i, imagePath) in enumerate(paths.list_images(args["test"])):
	# load the test image, convert it to grayscale, and resize it to
	# the canonical size
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the test image and
	# predict the make of the car
	H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True,visualize=False)
	# H = hog_scratch(gray, 
    #         cell_size=4, 
    #         block_size=2, 
    #         bins=9)
	pred = model.predict(H.reshape(1, -1))[0]
 
	# # visualize the HOG image
	# hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	# hogImage = hogImage.astype("uint8")
	# cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
 
	# draw the prediction on the test image and display it
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .5,
		(0, 255, 0), 1)
	print(pred.title())
	cv2.imshow("Test Image #{}".format(i + 1), image)
	cv2.waitKey(0)