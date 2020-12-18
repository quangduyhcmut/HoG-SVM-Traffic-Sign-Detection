import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import PIL.Image as Image

def _hog_normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')

    return out

def hogDescriptorScratch(im, cell_size=(8,8), orientations = 9, block_norm = None, cells_per_block=(4,4), visualize = True, visualize_grad=True):
	# square root normalization and extract image shape
	image = np.sqrt(im).astype(np.float32)
	sx, sy = image.shape # image size
	orientations = orientations # number of gradient bins
	cx, cy = cell_size # pixels per cell
	b_row, b_col = cells_per_block	# number of cells in each block
	n_cellsx = int(np.floor(sx / cx))  # number of cells in x
	n_cellsy = int(np.floor(sy / cy))  # number of cells in y
	# compute gradient on image
	gx = np.zeros(image.shape)
	gy = np.zeros(image.shape)
	gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
	gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
	# visualize gradient image
	if visualize_grad:
		fig, a = plt.subplots(1,2)    
		a[0].imshow(gx, cmap='gray')
		a[1].imshow(gy, cmap='gray')
		plt.show()
	# compute magnitute and orientation (phase) of gradient image
	grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
	grad_ori =  np.rad2deg(np.arctan2(gy, gx + 1e-15)) % 180
			
	# compute histogram of orientations with magnitute-based weights
	orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations)) 
	for dx in range(n_cellsx):
		for dy in range(n_cellsy):
			ori = grad_ori[dy*cy:dy*cy+cy, dx*cx:dx*cx+cx]
			mag = grad_mag[dy*cy:dy*cy+cy, dx*cx:dx*cx+cx]
			# https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
			hist, _ = np.histogram(ori, bins=orientations, range=(0, 180), weights=mag) # 1-D vector, 9 elements
			orientation_histogram[dy, dx, :] = hist
   
   
		pass
	pass

	# compute block normalization (L2)
	if block_norm is not None:
		n_blocks_row = (n_cellsy - b_row) + 1
		n_blocks_col = (n_cellsx - b_col) + 1
		normalized_blocks = np.zeros((n_blocks_row, n_blocks_col,b_row, b_col, orientations))

		for r in range(n_blocks_row):
			for c in range(n_blocks_col):
				block = orientation_histogram[r:r + b_row, c:c + b_col, :]
				normalized_blocks[r, c, :] = _hog_normalize_block(block, method=block_norm)
		
		# visualize HoG feature
		if visualize:
			hog_image = None
			from skimage.draw import draw
			radius = min(cy, cx) // 2 - 1
			orientations_arr = np.arange(orientations)
			orientation_bin_midpoints = (np.pi * (orientations_arr + .5) / orientations)
			dr_arr = radius * np.sin(orientation_bin_midpoints)
			dc_arr = radius * np.cos(orientation_bin_midpoints)
			hog_image = np.zeros((sy, sx), dtype=float)
			for r in range(n_cellsy):
				for c in range(n_cellsx):
					for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
						centre = tuple([r * cy + cy // 2,
										c * cx + cx // 2])
						rr, cc = draw.line(int(centre[0] - dc),
										int(centre[1] + dr),
										int(centre[0] + dc),
										int(centre[1] - dr))
						hog_image[rr, cc] += orientation_histogram[r, c, o]
			return normalized_blocks.ravel(), hog_image
		else:
			return normalized_blocks.ravel()
	else:
		return orientation_histogram.ravel()

if __name__ == '__main__':
  
	im = cv2.imread(r'images\test-3-classes\3\42.png')
	im = cv2.resize(im, (96,96))
	im_ = im.copy()
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	import time
 
	start = time.time()
	hogFeatureScratch, hogImageScratch = hogDescriptorScratch(im, 
																orientations=9, 
																cell_size=(8,8), 
																cells_per_block=(4,4),
																block_norm='L2',
																visualize=True)
	print("Scratch HoG: ", time.time()- start)
	print("Feature vector: ", hogFeatureScratch)
	print("Feature vector shape: ", hogFeatureScratch.shape)
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
	# visualize experiment result
	fig, a = plt.subplots(1,3)    
	a[0].imshow(cv2.cvtColor(im_, cv2.COLOR_BGR2RGB))
	a[0].set_title('Original traffic sign')
	a[1].imshow(hogImageSkimage, cmap='gray')
	a[1].set_title('HOG feature from Skimage')
	a[2].imshow(hogImageScratch, cmap='gray')
	a[2].set_title('HOG feature from scratch')
	plt.tight_layout()
	plt.show()
	hogImageScratch = np.stack([hogImageScratch,hogImageScratch,hogImageScratch],axis=-1)
	print(hogImageScratch.shape)
	
	cv2.imwrite('output-images/ori.png', im_)
	cv2.imwrite('output-images/hog_scratch.png', hogImageScratch)
	pass

