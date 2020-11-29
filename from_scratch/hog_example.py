import os
import cv2
import numpy as np
from numpy import linalg as LA
import time
from skimage import exposure
from skimage import feature

IMG = '38.png'
# no input image normalizing 
# TODO: square norm or variance norm => DONE


def hog_scratch(img_gray, cell_size=4, block_size=2, bins=9):
    # https://minhng.info/tutorials/histograms-of-oriented-gradients.html
    img = img_gray
    h, w = img.shape # 100, 100
    
    # sqrt norm
    img = np.sqrt(img).astype(np.float32)
    
    # gradient
    x_kernel = np.array([[-1, 0, 1]])
    y_kernel = np.array([[-1], [0], [1]])
    dx = cv2.filter2D(img, cv2.CV_32F, x_kernel)
    dy = cv2.filter2D(img, cv2.CV_32F, y_kernel)
    
    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx+0.00001)) # radian
    orientation = np.degrees(orientation) # -90 -> 90
    orientation += 90 # 0 -> 180
    
    num_cell_x = w // cell_size # 100
    num_cell_y = h // cell_size # 100
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins]) # 25 x 25 x 9
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
            mag = magnitude[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag) # 1-D vector, 9 elements
            hist_tensor[cy, cx, :] = hist
        pass
    pass
    
    # normalization L2 norm
    redundant_cell = block_size-1
    feature_tensor = np.zeros([num_cell_y-redundant_cell, num_cell_x-redundant_cell, block_size*block_size*bins])
    for bx in range(num_cell_x-redundant_cell): # 7
        for by in range(num_cell_y-redundant_cell): # 15
            by_from = by
            by_to = by+block_size
            bx_from = bx
            bx_to = bx+block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten() # to 1-D array (vector)
            feature_tensor[by, bx, :] = v / (LA.norm(v, 2) + 0.0000001)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any(): # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v
    
    return feature_tensor.flatten().reshape(-1,) # 3780 features

def hog_skimage(img, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), transform_sqrt=True, visualize=False):
    return feature.hog(img, orientations = orientations, pixels_per_cell = pixels_per_cell,cells_per_block = cells_per_block, transform_sqrt = transform_sqrt, visualize = visualize)

def main(img_path):
    # gray image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # print(img.shape)
    img = cv2.resize(src=img, dsize=(100, 100))
    start = time.time()
    
    f = hog_scratch(img, 
            cell_size=4, 
            block_size=2, 
            bins=9)
    
    print('Processing time from scratch: ', time.time() - start)
    # print('Extracted feature vector of %s. Shape:' % img_path)
    print('Feature size:', f.shape)
    print('Features (HOG):', f)
    
    start_time = time.time()
    
    H = feature.hog(img, 
                    orientations=9, 
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2), 
                    transform_sqrt=True, 
                    visualize=False)
    print('Processing time from skimage: ', time.time()-start)
    # print(hogImage.shape)
    print(H)
    print(H.shape)

if __name__ == "__main__":

    main(IMG)

    
