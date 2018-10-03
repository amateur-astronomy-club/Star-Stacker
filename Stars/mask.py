import cv2
import numpy as np

img = cv2.imread('img1.jpeg',0)
resized_image = cv2.resize(img, (100, 50))
print (np.shape(resized_image))
cv2.imshow('img',resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
