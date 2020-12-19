import cv2
import os
from skimage.feature import hog

img = cv2.imread(r'images\test-3-classes\3\944.png')

ft, im = hog(img, visualize=True)

cv2.imshow('a',im/255)
cv2.waitKey(0)
cv2.destroyAllWindows()