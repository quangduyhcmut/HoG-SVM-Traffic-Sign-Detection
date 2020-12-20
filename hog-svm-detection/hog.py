import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import time

if __name__ == '__main__':
  
	im = cv2.imread('./images/test-3-classes/2/11.png')
	print(im.shape)
	im = cv2.resize(im, (96,96))
	# im_ = im.copy()/255
	# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	start = time.time()
	hogFeatureSkimage, hogImageSkimage = hog(im, 
															orientations=9, 
															pixels_per_cell=(8, 8),
															cells_per_block=(4, 4), 
															transform_sqrt=True, 
															visualize=True,
															block_norm='L2')
	print("Skimage HoG: ", time.time()- start)
	print("Feature vector: ", hogFeatureSkimage)
	print("Feature vector shape: ", hogFeatureSkimage.shape)
	
	# hogImageSkimage = np.stack([hogImageSkimage, hogImageSkimage, hogImageSkimage], axis=-1)
	
	# vis = np.concatenate((im, hogImageSkimage), axis = 1)
	# cv2.imwrite('output-images/hog-img.png', vis*255)
	cv2.imshow("HOG Image", hogImageSkimage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

