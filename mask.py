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


images = glob('Stars/input.jpg')

image_list = [cv2.imread(i) for i in images]

for image in image_list:
    image = adjust_gamma(image,2)
    # image = cv2.resize(image,(1200,600))
    print(np.max(image))
    padx = 51 - (np.shape(image)[1]%51)
    pady = 51 - (np.shape(image)[0]%51)

    blue = image[:,:,0]
    green = image[:,:,1]
    red = image[:,:,2]

    red = np.pad(red, ((0,pady), (0,padx)), 'mean')
    blue = np.pad(blue, ((0,pady), (0,padx)), 'mean')
    green = np.pad(green, ((0,pady), (0,padx)), 'mean')

    image = np.dstack((blue,green,red))

    light_model = np.zeros(np.shape(image))

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

			y_blue = inp[:,:,0]
			y_green = inp[:,:,1]
			y_red = inp[:,:,2]

			beta_blue = (np.linalg.inv(X.T @ W @ X)) @ (X.T @ W @ y_blue)
			beta_green = (np.linalg.inv(X.T @ W @ X)) @ (X.T @ W @ y_green)
			beta_red = (np.linalg.inv(X.T @ W @ X)) @ (X.T @ W @ y_red)

			light_model_blue[i:i+51,j:j+51] = np.reshape(X@beta_blue,(np.shape(img)[0], np.shape(img)[1]))
			light_model_geen[i:i+51,j:j+51] = np.reshape(X@beta_geen,(np.shape(img)[0], np.shape(img)[1]))
			light_model_red[i:i+51,j:j+51] = np.reshape(X@beta_red,(np.shape(img)[0], np.shape(img)[1]))

            for k in range(3):
                y = inp[:,:,k].flatten()
                beta = (np.linalg.inv(X.T @ W @ X)) @ (X.T @ W @ y)
                light_model[i:i+51,j:j+51,k] = np.reshape(X@beta,(np.shape(img)[0], np.shape(img)[1]))

light_model = cv2.resize(light_model,(600,1200))
cv2.imshow("Removed light pollution", light_model)
cv2.waitKey(0)
cv2.destroyAllWindows()
