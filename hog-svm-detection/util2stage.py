import os, sys, shutil, imutils, cv2, sklearn, time
import numpy as np
from sklearn import svm
from skimage.feature import hog
from nms import lnms

# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.linear_model import SGDClassifier
# from imutils import paths
# import argparse
# import time
# import pickle
# from linearClassifier import LinearSVM
# extract Histogram of Oriented Gradients from the logo
def slide_window(img, mask,
                x_start_stop=[None, None], y_start_stop=[None, None], 
                xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Slide window over image and return all resulting bounding boxes."""
    # If x and/or y start/stop positions not defined, set to image size
    if not x_start_stop[0]:
        x_start_stop[0] = 0
    if not x_start_stop[1]:
        x_start_stop[1] = img.shape[1]
    if not y_start_stop[0]:
        y_start_stop[0] = 0
    if not y_start_stop[1]:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    w = x_start_stop[1] - x_start_stop[0]
    h = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    pps_x = int((1.0 - xy_overlap[0]) * xy_window[0])
    pps_y = int((1.0 - xy_overlap[1]) * xy_window[1])
    # Compute the number of windows in x/y
    n_x = int((w - xy_window[0])/pps_x + 1)
    n_y = int((h - xy_window[1])/pps_y + 1)
    # Initialize a list to append window positions to
    window_list = []
    for i in range(n_y):
        y_pos = i * pps_y + y_start_stop[0]
        for j in range(n_x):
            x_pos = j * pps_x + x_start_stop[0]
            # print(np.count_nonzero(mask[y_pos:y_pos+xy_window[1], x_pos:x_pos+xy_window[0]]))
            if (np.count_nonzero(mask[y_pos:y_pos+xy_window[1], x_pos:x_pos+xy_window[0]])>(xy_window[0]*xy_window[1]//10)):
                # print(np.count_nonzero(mask[y_pos:y_pos+xy_window[1], x_pos:x_pos+xy_window[0]]))
                # print(y_pos,y_pos+xy_window[1], x_pos, x_pos+xy_window[0])
                bbox = ((x_pos,y_pos), (x_pos+xy_window[0],y_pos+xy_window[1]))
                window_list.append(bbox)
    return window_list

def get_multiscale_windows(img, mask, size):
    """Return bounding boxes of windows of different scales slid over img
    for likely vehicle positions."""
    cv2.waitKey()
    window_list = list()
    method = "full"
    if method == "right":
        # for the included video this is fine to improve speed
        # but for other videos, method == "full" should be used
        window_list += slide_window(img, mask, 
                xy_overlap = (0.75, 0.75),
                # x_start_stop = [620, 620 + 6*96],
                # y_start_stop = [385, 385 + 2*96],
                xy_window = (32, 32))
        window_list += slide_window(img, mask, 
                xy_overlap = (0.75, 0.75),
                # x_start_stop = [620, None],
                # y_start_stop = [385, 385 + 2*128],
                xy_window = (48, 48))
    elif method == "full":
        # window_list += slide_window(img, mask,
        #         xy_overlap = (0.75, 0.75),
        #         # x_start_stop = [620 - 6*96, 620 + 6*96],
        #         # y_start_stop = [385, 385 + 2*96],
        #         xy_window = (64, 64))
        window_list += slide_window(img, mask,
                xy_overlap = (0.75, 0.75),
                x_start_stop = [0, size[1]],
                y_start_stop = [0, size[0]],
                xy_window = (45, 45))
        window_list += slide_window(img, mask,
                xy_overlap = (0.75, 0.75),
                x_start_stop = [0, size[1]],
                y_start_stop = [0, size[0]],
                xy_window = (60, 60))
        window_list += slide_window(img, mask,
                xy_overlap = (0.75, 0.75),
                x_start_stop = [0, size[1]],
                y_start_stop = [0, size[0]],
                xy_window = (75, 75))
    else:
        raise ValueError(method)
    return window_list

def extract_window(img, bbox, stage):
    """Extract patch from window and rescale to size used by classifier."""
    row_begin = bbox[0][1]
    row_end = bbox[1][1]
    col_begin = bbox[0][0]
    col_end = bbox[1][0]
    patch = img[row_begin:row_end, col_begin:col_end]
    window = cv2.resize(patch, (96,96))
    return hog(window, 
                orientations=9, 
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), 
                transform_sqrt=True, 
                visualize=False,
                block_norm='L2')
    # return window

def resize(image, size):
    height, width, _ = image.shape
    if height > width:
        scale = size / height
        resized_height = size
        resized_width = int(width * scale)
        flag = (scale, 1)
    else:
        scale = size / width
        resized_height = int(height * scale)
        resized_width = size
        flag = (1, scale)
    image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    new_image = np.zeros((size, size, 3), dtype = np.uint8)
    new_image[0:resized_height, 0:resized_width] = image
    return new_image, scale, (resized_height, resized_width)

def detect(img, window_list, pipeline1, pipeline2):
    """Classify all windows within img.
       *Return list of box: [[x1, y1, x2, y2, label, score], [...]]"""
    windows = []
    for bbox in window_list:
        window = extract_window(img, bbox, stage=1)
        windows.append(window)
    windows = np.stack(windows)
    sign_or_notsign = pipeline1.predict(windows.reshape((len(windows),-1)))
    sign_index = np.where(sign_or_notsign != '0')
    sign_feature = np.squeeze(np.take(windows, sign_index, axis=0))
    
    if len(sign_feature)==0:
        return []
    else:
        sign_prob = pipeline2.predict_proba(sign_feature)
        # thresholded = np.where(sign_prob>[0.85, 0.87, 0.82, 0.84, 0.80, 0.92, 0.80, 0.5],[0,1,2,3,4,5,6,7],[7,7,7,7,7,7,7,7])
        thresholded = np.where(sign_prob>[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],[0,1,2,3,4,5,6,7],[7,7,7,7,7,7,7,7])
        # thresholded = np.where(sign_prob>0.5,[0,1,2,3,4,5,6,7],[7,7,7,7,7,7,7,7])
        r = list(range(len(thresholded))) 
        c = thresholded.min(axis=1) 
        detected_windows = [[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1],label,prob] for label, bbox, prob in zip(np.amin(thresholded, axis=1), np.squeeze(np.take(window_list, sign_index, axis = 0)), sign_prob[r,c]) if label!=7]
        return detected_windows

def predictimage(image, pipeline1, pipeline2):
	label =['cam nguoc chieu', 'cam dung va do', 'cam re', 'gioi han toc do', 'cam khac', 'nguy hiem', 'hieu lenh', 'Not sign']
	new_image, scale, newsize = resize(image, 1000)
	gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([255, 255, 255]))
	window_list = np.array((get_multiscale_windows(new_image, mask, newsize)))
	st = time.time()
	detected_windows = detect(gray, window_list, pipeline1, pipeline2)
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	fontScale              = 2.5
	fontColor              = (255,255,255)
	lineType               = 5
	detected_windows_nms = lnms(detected_windows, 0.3)
	for box in detected_windows_nms:
		cv2.rectangle(image, tuple((int(box[0]/scale), int(box[1]/scale))), tuple((int(box[2]/scale), int(box[3]/scale))),(0, 255, 0), 5)
		cv2.putText(image, label[box[4]] + ' ' + str(round(box[5],2)), (int(box[0]/scale), int(box[3]/scale)), font, fontScale,fontColor,lineType)
	# for box in detected_windows_nms:
	# 	cv2.rectangle(new_image, tuple((box[0], box[1])), tuple((box[2],box[3])),(0, 255, 0), 2)
	# 	cv2.putText(new_image, label(box[4]) + ' ' + str(round(box[5],2)), (box[0], box[3]), font, fontScale,fontColor,lineType)
	return image