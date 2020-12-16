# SVM SGD for multiclass classification from sklearn training with 8 class (1 negative)
import os, sys, shutil
sys.path.append('from_scratch')
import numpy as np
import cv2
from hogExtractor import hogDescriptorScratch, hogDescriptorSkimage

import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from imutils import paths
import argparse
import time
import pickle
import random

# TODO: save feature vectors

def extract_feature(path, method = 'Sklearn'):
	print("[INFO] extracting training features from {}...".format(path))
	data = []
	labels = []
	index = 0
	for imagePath in paths.list_images(path):
		index +=1
		make = imagePath.split("\\")[-2]
	
		# load the image, convert it to grayscale, and detect edges
		image = cv2.imread(imagePath)
		try:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray = cv2.resize(gray, (100, 100))
			if method == 'Sklearn':
			# extract Histogram of Oriented Gradients from the logo
				hogFeature = hogDescriptorSkimage(gray, 
														orientations=9, 
														pixels_per_cell=(4, 4),
														cells_per_block=(2, 2), 
														transform_sqrt=True, 
														visualize=False,
														block_norm='L2')
			else:	
				hogFeature = hogDescriptorScratch(gray, 
														orientations=9, 
														cell_size=(8,8), 
														cells_per_block=(2,2),
														block_norm='L2',
														visualize=False)

			data.append(hogFeature)
			labels.append(make)
		except:
			print(imagePath)
		
		# if index == 10:
		# 	break

	data = np.stack(data, axis=0)
	data_ = data.copy()
	labels = np.stack(labels, axis=0)
	print("[INFO] Feature shape: {}".format(data.shape))
	return data, labels
 

# construct the argument parse and parse command line arguments
trainPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\big-zalo-data\train'
testPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\big-zalo-data\valid'
# trainPath = r'from_scratch\train_3'
# testPath = r'from_scratch\test_3'
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=False, default=trainPath, help="Path to the trafficsign training dataset")
ap.add_argument("-t", "--test", required=False, default=testPath, help="Path to the test dataset")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
start = time.time()
data_train, labels_train = extract_feature(path=args["training"])
data_val, labels_val = extract_feature(path = args["test"])

# save feature
# from numpy import asarray
# from numpy import save
# trainFeature = 'model/trainFeature.npy'
# trainLabel = 'model/trainLabel.npy'
# valFeature = 'model/valFeature.npy'
# valLabel = 'model/valLabel.npy'
# ftTrain = asarray(data_train)
# lbTrain = asarray(labels_train)
# ftVal = asarray(data_val)
# lbVal = asarray(labels_val)
# save(trainFeature, ftTrain, allow_pickle=True)
# save(trainLabel, lbTrain, allow_pickle=True)
# save(valFeature, ftVal, allow_pickle=True)
# save(valLabel, lbVal, allow_pickle=True)

print("[INFO] Finish extracting HoG features. Total time: {}".format(time.time()-start))
# define classifier

start = time.time()
print("[INFO] training...")
clf = SVC(C=10.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

# calibration for probability estimation
clf.fit(data_train, labels_train)
print(clf.score(data_train, labels_train))
print("[INFO] Finish training SVM model. Total time: {}".format(time.time()-start))
# "train" the nearest neighbors classifier

print("[INFO] saving model...")
modelPath = 'model/SGD-SVM-Sklearn-8-class.sav'
pickle.dump(clf, open(modelPath, 'wb'))  

print('[INFO] Validation accuracy on Skimage HoG extractor', clf.score(data_val, labels_val))
# print('Test accuracy on Scratch HoG extractor', model2.score(data2, labels))

print(metrics.confusion_matrix(clf.predict(data_val), labels_val))

# print(clf.predict_proba(data_val))
prob = clf.predict_proba(data_val)
# print(prob)
# random.shuffle(prob)
# for sc in prob:
#     print('score list', sc)
#     print('score', max(sc))
#     print('predicted class', sc.copy().tolist().index(max(sc)))
    

# score = clf.decision_function(data)
# score = clf.predict_proba(data)
# for sc in score:
#     print('score list', sc)
#     print('max score', max(sc))
#     print('predicted class', sc.copy().tolist().index(max(sc)))

training_score = prob
print(training_score.shape)
average_score_list = [0,0,0,0,0,0,0,0]
number_of_sample = [0,0,0,0,0,0,0,0]
for sample_score in training_score:
    label = sample_score.copy().tolist().index(max(sample_score))
    average_score_list[label] += max(sample_score)
    number_of_sample[label] += 1
print(number_of_sample)

for index, score in enumerate(average_score_list):
    average_score_list[index] = score/number_of_sample[index]
    
print(average_score_list)
print('average score for each class: {}'.format(average_score_list))

# probability estimation reference: https://mmuratarat.github.io/2019-10-12/probabilistic-output-of-svm#:~:text=SVMs%20don't%20output%20probabilities,the%20output%20to%20class%20probabilities.&text=One%20standard%20way%20to%20obtain,in%20many%20decent%20SVM%20implementations.