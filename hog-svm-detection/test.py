from util2stage import *
import cv2
import numpy as np
img = cv2.imread(r'.\img_test\eq.png')
new_img, scale, newsize = resize(img, 1000)
mask = get_mask(new_img)
res = cv2.bitwise_or(new_img, new_img, mask = mask)
cv2.imshow('red', res)
cv2.waitKey()
# cv2.imshow('blue', blue_mask)
# cv2.waitKey()
cv2.destroyAllWindows()
