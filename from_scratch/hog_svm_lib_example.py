import os, sys, shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics

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

""" # EXAMPLE CODE
digits = datasets.load_digits()

for i in range(0,4):
    plt.subplot(2, 4,i + 1)
    plt.axis('off')
    imside = int(np.sqrt(digits.data[i].shape[0]))
    im1 = np.reshape(digits.data[i],(imside,imside))
    plt.imshow(im1, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: {}'.format(digits.target[i]))
plt.show()

n_samples = len(digits.images)
print(n_samples)
data_images = digits.images.reshape((n_samples, -1))
data_labels = digits.target
print(data_labels.shape)
print(data_images.shape)
print(type(data_images))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_images,digits.target)

classifier = svm.SVC(gamma = 0.001)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
"""

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

	make = imagePath.split("\\")[-2]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the logo
	H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
	# H = hog_scratch(gray, 
    #         cell_size=4, 
    #         block_size=2, 
    #         bins=9)
	# update the data and labels
	data.append(H)
	labels.append(make)

data = np.stack(data, axis=0)
labels = np.stack(labels, axis=0)

classifier = svm.SVC(gamma = 0.001)
classifier.fit(data, labels)


for (i, imagePath) in enumerate(paths.list_images(args["test"])):
	# load the test image, convert it to grayscale, and resize it to
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (100, 100))
 
	# extract Histogram of Oriented Gradients from the test image and
	H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True,visualize=False)
	# H = hog_scratch(gray, 
    #         cell_size=4, 
    #         block_size=2, 
    #         bins=9)
	pred = classifier.predict(H.reshape(1, -1))[0]
 
	# draw the prediction on the test image and display it
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .5,
		(0, 255, 0), 1)
	print(pred.title())
	cv2.imshow("Test Image #{}".format(i + 1), image)
	cv2.waitKey(0)