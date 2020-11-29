from skimage import exposure
from skimage import feature
import cv2

img_path = '38.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(src=img, dsize=(100, 100))

(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(4, 4),
	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
	visualize=True)

print(H)
# cv2.imshow("HOG Image", hogImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
# hogImage = hogImage.astype("uint8")



# cv2.imshow("HOG Image", hogImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

