import cv2
import numpy as np
import scipy.sparse as sp
from glob import glob
from math import fabs, sqrt


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


images = glob('Stars/img*.jpeg')

image_list = [cv2.imread(i) for i in images]

count = 1
for image in image_list:
	image = adjust_gamma(image,2)
	# image = cv2.resize(image,(1200,600))
	padx = 51 - (np.shape(image)[1]%51)
	pady = 51 - (np.shape(image)[0]%51)

	blue = image[:,:,0]
	green = image[:,:,1]
	red = image[:,:,2]

	red = np.pad(red, ((0,pady), (0,padx)), 'mean')
	blue = np.pad(blue, ((0,pady), (0,padx)), 'mean')
	green = np.pad(green, ((0,pady), (0,padx)), 'mean')

	image = np.dstack((blue,green,red))

	light_model = np.zeros(np.shape(image)) #the datatype HAS to be uint8 to work with opencv

	mask = np.zeros((np.shape(image)[0], np.shape(image)[1]))
	for i in range(0, np.shape(image)[0], 51): #upto 3009
		for j in range(0, np.shape(image)[1], 51): #upto 3978
			inp = image[i:i+51,j:j+51,:]
			img = np.sum(inp, axis = 2)

			avg = np.mean(img)
			var = np.var(img)

			out = img - avg
			window_mask = (np.less(np.absolute(out), 2*np.ones((51,51))*sqrt(var)) + 0) * 255
			mask[i:i+51, j:j+51] = window_mask

			###### LIGHT POLLUTION ESTIMATION ######
			X = np.asarray([[m,l,1] for m in range(i,i+51) for l in range(j,j+51)])
			W = sp.diags(window_mask.flatten())

			for k in range(3):
				y = inp[:,:,k].flatten()
				beta = (np.linalg.inv(X.T @ W @ X)) @ (X.T @ W @ y)
				light_model[i:i+51,j:j+51,k] = np.reshape(X@beta,(np.shape(img)[0], np.shape(img)[1]))


	# light_model = cv2.resize(light_model,(600,1200))
	# print(light_model)
	ideal = image-light_model
	ideal = (ideal.clip(min=0)).astype('uint8')
	# print(image[0:10,0:10,:])
	# print(light_model[0:10,0:10,:])
	# print(ideal[0:10,0:10,:])
	cv2.imwrite("Stars/out"+str(count)+".jpeg", ideal)
	count += 1



images = glob('Stars/out*.jpeg')

image_list = [cv2.imread(i) for i in images]

count = 1
for image in image_list:
	image = adjust_gamma(image,2)
	# image = cv2.resize(image,(1200,600))
	padx = 51 - (np.shape(image)[1]%51)
	pady = 51 - (np.shape(image)[0]%51)

	blue = image[:,:,0]
	green = image[:,:,1]
	red = image[:,:,2]

	red = np.pad(red, ((0,pady), (0,padx)), 'mean')
	blue = np.pad(blue, ((0,pady), (0,padx)), 'mean')
	green = np.pad(green, ((0,pady), (0,padx)), 'mean')

	image = np.dstack((blue,green,red))

	light_model = np.zeros(np.shape(image)) #the datatype HAS to be uint8 to work with opencv

	mask = np.zeros((np.shape(image)[0], np.shape(image)[1]))
	for i in range(0, np.shape(image)[0], 51): #upto 3009
		for j in range(0, np.shape(image)[1], 51): #upto 3978
			inp = image[i:i+51,j:j+51,:]
			img = np.sum(inp, axis = 2)

			avg = np.mean(img)
			var = np.var(img)

			out = img - avg
			window_mask = (np.less(np.absolute(out), 2*np.ones((51,51))*sqrt(var)) + 0) * 255
			mask[i:i+51, j:j+51] = window_mask

			###### LIGHT POLLUTION ESTIMATION ######
			X = np.asarray([[m,l,1] for m in range(i,i+51) for l in range(j,j+51)])
			W = sp.diags(window_mask.flatten())

			for k in range(3):
				y = inp[:,:,k].flatten()
				beta = (np.linalg.inv(X.T @ W @ X)) @ (X.T @ W @ y)
				light_model[i:i+51,j:j+51,k] = np.reshape(X@beta,(np.shape(img)[0], np.shape(img)[1]))


	# light_model = cv2.resize(light_model,(600,1200))
	# print(light_model)
	ideal = image-light_model
	ideal = (ideal.clip(min=0)).astype('uint8')
	# print(image[0:10,0:10,:])
	# print(light_model[0:10,0:10,:])
	# print(ideal[0:10,0:10,:])
	cv2.imwrite("Stars/out"+str(count)+".jpeg", ideal)
	count += 1
