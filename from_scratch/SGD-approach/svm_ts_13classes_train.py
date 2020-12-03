## LINEAR
## TODO: 
## - Add kernel function
## - Create mini data for fast training and testing
## - cross validation and grid search for hyper parameter
#########################################################
sys.path.append(r'from_scratch')

import random
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import cv2
from hog_example import hog_scratch, hog_skimage
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers import LinearSVM
import time

plt.rcParams['figure.figsize'] = (20.0, 16.0) # set default size of plot
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# DATA_PATH = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_TGM\cropped'
DATA_PATH = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\Bang-chia-data-zalo'

"""
- input feature SVM: numpy.ndarray shape [N, features] free range (normalize?) float 64 example [-71.64189796 -73.98173469 -69.47391837 ... -33.86195918 -42.39957143] 
- input label SVM: numpy.ndarray shape [N, label] int 32. Example [6 9 9 ... 4 9 3]
- Training data shape:  (49000, 3072)
- Validation data shape:  (1000, 3072)
- Test data shape:  (1000, 3072)
"""
# initialize the data matrix and labels
print("[INFO] extracting features...")

data = []
imgs = []
labels = []

# classes = ['cam nguoc chieu', 'cam o to, mo to, xe tai', 'cam re', 'gioi han toc do', 'cam dung va do xe', 'cam do xe', 'cam khac', 'nguy hiem: tre em', 'nguy hiem khac', 'huong phai di vong sang phai', 'vong xuyen', 'hieu lenh khac']
# classes_num = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
classes = ['0','1', '2', '3', '4', '5', '6']
classes_num = ['0','1', '2', '3', '4', '5', '6']
num_classes = len(classes)

for index, imagePath in enumerate(paths.list_images(DATA_PATH)):
    # extract the make of the car

    make = imagePath.split("\\")[-2]
    make = classes_num.index(make)

    # load the image, convert it to grayscale, and detect edges
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    gray = cv2.resize(gray, (100, 100))
    
    imgs.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
       
    # extract Histogram of Oriented Gradients from the logo
    # H = hog_skimage(gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
    H = hog_scratch(gray, 
            cell_size=4, 
            block_size=2, 
            bins=9)
    # update the data and labels
    data.append(H)
    labels.append(make)
    # if index == 20:
    #     break
    

data = np.stack(data, axis=0)
labels = np.stack(labels, axis=0)
# print(labels)
# quit()
print ('Training all data shape: ', data.shape)
print ('Training all labels shape: ', labels.shape)

# samples_per_class = 10
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(labels == y)
#     # print(idxs)
#     # print(y)
#     # print(labels, y)
#     # print(y,labels == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(imgs[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()

# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training = int(data.shape[0] * 0.8)
num_validation = data.shape[0] - num_training

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = data[mask]
y_val = labels[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = data[mask]
y_train = labels[mask]

print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', y_val.shape)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))

# As a sanity check, print out the shapes of the data
print ('Training data shape: ', X_train.shape)
print ('Validation data shape: ', X_val.shape)

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

print (X_train.shape, X_val.shape)
# quit()
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=5e-7, reg=5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print ('That took %fs' % (toc - tic))

svm.save_weights()

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(X_train)
print ('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = svm.predict(X_val)
print ('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))