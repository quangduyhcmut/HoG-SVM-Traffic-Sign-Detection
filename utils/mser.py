import cv2
import time
import os
import numpy as np
for item in os.listdir('img_test'):
    if os.path.isdir(os.path.join('img_test',item)):
        continue
    im = cv2.imread('img_test/{}'.format(item))
    im = cv2.resize(im, (1000,1000))
    mser = cv2.MSER_create(5, 60, 14400, 0.25, .2, 200, 1.01, 0.003, 5)
    start = time.time()
    msers, bboxes = mser.detectRegions(im)
    print(len(msers))
    print(len(bboxes))
    print(time.time()-start)
    mask = np.zeros((im.shape[0], im.shape[1]),dtype = np.uint8)
    for box in bboxes:
        mask[box[0]-box[2]//2:box[0]+box[2]//2,box[1]-box[3]//2:box[1]+box[3]//2] = 1
    
        # cv2.rectangle(im, (box[0]-box[2]//2,box[1]-box[3]//2), (box[0]+box[2]//2,box[1]+box[3]//2), (0,255,0), 1)
    im = cv2.bitwise_and(im,im, mask = mask)
    cv2.imshow('a',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()