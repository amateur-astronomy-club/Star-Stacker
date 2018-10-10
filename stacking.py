#newfile

import numpy as np
from scipy import ndimage
import skimage
import cv2
from glob import glob
import math

images = glob('Stars/rotate_out*.jpeg')

image_list = [cv2.imread(i) for i in images]
coordinates_array = []
offset_array = []
output_img = np.zeros((np.shape(image_list[0])[0], np.shape(image_list[0])[1], 3))


pady = 10
padx = 10


count = 0
for image in image_list:
	#print (np.shape(image))
	output_img += image

cv2.imwrite("Stars/final.jpeg", output_img.clip(max=255))

