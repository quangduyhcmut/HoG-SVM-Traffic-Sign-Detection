import sys, os
sys.path.append("compare/utils-scratch/")

import random
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import cv2
from skimage.feature import hog

from svm_loss import svm_loss_naive, svm_loss_vectorized
from linearClassifier import LinearSVM
import time
from sklearn import metrics
from tqdm import tqdm
import argparse

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plot
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def extract_feature(path):
	print("[INFO] extracting training features from {}...".format(path))
	data = []
	labels = []
	filenames = []
	index = 0
	for imagePath in tqdm(paths.list_images(path)):
		index +=1
		make = imagePath.split("\\")[-2]
	
		# load the image, convert it to grayscale, and detect edges
		image = cv2.imread(imagePath)
		try:			
			gray = cv2.resize(image, (96, 96))
			gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
			# extract Histogram of Oriented Gradients from the logo
			hogFeature = hog(gray,orientations=9,pixels_per_cell=(8, 8),cells_per_block=(4, 4),transform_sqrt=True,visualize=False,block_norm='L2')
			data.append(hogFeature)
			labels.append(int(make))
			filenames.append(imagePath)
		except:
			print(imagePath)
		# break
	data = np.stack(data, axis=0)
	print(data.shape)
	data = np.hstack([data, np.ones((data.shape[0], 1))])
	print(data.shape)
	# only has to worry about optimizing a single weight matrix W => bias trick
	labels = np.stack(labels, axis=0)
	print(labels.shape)
	print("[INFO] Feature shape: {}".format(data.shape))
	return data, labels, filenames

trainPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage2_classifier\train'
valPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage2_classifier\valid'
modelPath = 'models/compare_scratch_model.sav'

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=False, default=trainPath, help="Path to the training dataset")
ap.add_argument("-v", "--validation", required=False, default=valPath, help="Path to the validation dataset")
args = vars(ap.parse_args())

# initialize the data matrix and labels
start = time.time()
data_train, labels_train, _ = extract_feature(path=args["training"])
data_val, labels_val, filenames_val = extract_feature(path = args["validation"])
print("[INFO] Finish extracting HoG features. Total time: {}".format(time.time()-start))

svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(data_train, labels_train, learning_rate=1e-2, reg=0.001, num_iters=15000, verbose=False)
toc = time.time()
print ('[INFO] That took %fs' % (toc - tic))

svm.save_weights(path = modelPath)

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.grid('on')
plt.savefig("figures/training_log.png")
# plt.show()

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(data_train)
print ('[INFO] Training accuracy: %f' % (np.mean(labels_train == y_train_pred), ))

load_svm = LinearSVM()
load_svm.load_weights(row = data_val.shape[1], col = len(os.listdir(valPath)), path = r'models/compare_scratch_model.sav')
predLabel = load_svm.predict(data_val)

print('[INFO] Confusion matrix... \n', metrics.confusion_matrix(predLabel, labels_val))
print('[INFO] Validation accuracy: {}'.format(metrics.accuracy_score(labels_val,predLabel)))