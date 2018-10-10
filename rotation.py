#newfile

import numpy as np
from scipy import ndimage
import skimage
import cv2
from glob import glob
import math

images = glob('Stars/out*.jpeg')

print(images[4][-12::])
image_list = [cv2.imread(i) for i in images]
coordinates_array = []
offset_array = []
output_img = [[[]]]

pady = 10
padx = 10


count = 0
for image in image_list:
	# image = adjust_gamma(image,2)
	# image = cv2.resize(image,(1200,600))
	# blue = image[:,:,0]
	# green = image[:,:,1]
	# red = image[:,:,2]

	# red = np.pad(red, ((10,10), (10,10)), 'mean')
	# blue = np.pad(blue, ((10,10), (10,10)), 'mean')
	# green = np.pad(green, ((10,10), (10,10)), 'mean')

	# image = np.dstack((blue,green,red))

	img = np.sum(image, axis = 2)
	com = ndimage.measurements.center_of_mass(img)
	# com[1] = math.ceil(com[1])
	# com[0] = math.ceil(com[0])
	coordinates_array.append((math.ceil(com[1]),math.ceil(com[0])))

for c in range(len(coordinates_array)-1):
	print (coordinates_array[c+1][0]-coordinates_array[0][0], coordinates_array[c+1][1]-coordinates_array[0][1])
	offset_array.append([coordinates_array[c+1][0]-coordinates_array[0][0], coordinates_array[c+1][1]-coordinates_array[0][1], 0])

for i in range(1, len(image_list)):
	print (offset_array[i-1])
	output_img = ndimage.interpolation.shift(image, offset_array[i-1], order=3, mode='constant', cval=0.0, prefilter=True)
	cv2.imwrite("Stars/rotate_out"+str(i)+".jpeg", output_img)
	# img = np.sum(output_img, axis = 2)
	# com = ndimage.measurements.center_of_mass(output_img)
	# print (com)





	#print (math.atan(com[0]/com[1])*180/math.pi)

