import sys
sys.path.append('./blog_svm_2classes/code/')

from binary_classification import SVM
from kernel import Kernel
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import argh
import time
from imutils import paths

from hog_example import hog_scratch
import argparse
import cv2


def binary_classifier(samples, labels):
    num_samples = samples.shape[0]
    num_features = samples.shape[1]
    
    start = time.time()
    clf = SVM(Kernel.rbf(0.1), 0.1)
    clf.fit(samples, labels)
    print('Training time: ', time.time() - start)

    # start = time.time()
    # labels = clf.predict(samples)
    # # print(labels)
    # print('Predicting time: ', time.time() - start)

def get_hog_feature():
    return 

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    # argh.dispatch_command(binary_classifier)
    
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--training", required=False, default='./train_binary', help="Path to the logos training dataset")
    ap.add_argument("-t", "--test", required=False, default='./test_binary',  help="Path to the test dataset")
    args = vars(ap.parse_args())
    
    # initialize the data matrix and labels
    print("[INFO] extracting features...")
    data = []
    labels = []

    for imagePath in paths.list_images(args["training"]):
        # print(imagePath.split("\\"))
        make = imagePath.split("\\")[-2]
    
        # load the image, convert it to grayscale, and detect edges
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        gray = cv2.resize(gray, (100, 100))
    
        # extract Histogram of Oriented Gradients from the logo
        # H = hog_hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
        H = hog_scratch(gray, 
                cell_size=4, 
                block_size=2, 
                bins=9)
        # update the data and labels
        # print(H.shape)
        data.append(H)
        if make == 'cam nguoc chieu':
            labels.append(1.0)
        else:
            labels.append(-1.0)
        
        # labels.append(make)

    data = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0).astype(np.float64)
    print(data.shape)
    print(data.dtype)
    print(labels.shape)
    print(labels)


    # "train" the nearest neighbors classifier
    print("[INFO] training classifier...")
    start = time.time()
    clf = SVM(Kernel.rbf(0.1), 0.1)
    clf.fit(data, labels)
    print('Training time: ', time.time() - start)
    print("[INFO] evaluating...")
    
    data_test = []
    # TODO: them vectorization 
    for (i, imagePath) in enumerate(paths.list_images(args["test"])):
        # load the test image, convert it to grayscale, and resize it to
        # the canonical size
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
    
        # extract Histogram of Oriented Gradients from the test image and
        # predict the make of the car
        # H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True,visualize=False)
        H = hog_scratch(gray, 
                cell_size=4, 
                block_size=2, 
                bins=9)
        # print(H)
        data_test.append(H)
    
    data_test = np.stack(data_test, axis=0)
    pred = clf.predict(data_test)
    print(pred)
    if pred == 1.0:
        label = 'cam nguoc chieu'
    else: label = 'cam re'
    # print(label)
    
        # # visualize the HOG image
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")
        # cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
    
        # draw the prediction on the test image and display it
        
        # cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .5,
        #     (0, 255, 0), 1)
        # print(pred.title())
        # cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .5,
        #     (0, 255, 0), 1)
        # print(label)
        # cv2.imshow("Test Image #{}".format(i + 1), image)
        # cv2.waitKey(0)
        