import cv2
import numpy as np
from glob import glob
from math import fabs, sqrt

images = glob('Stars/input.jpg')
np.set_printoptions(threshold=np.nan)

image_list = [cv2.imread(i) for i in images]

for image in image_list:
    img = np.sum(image, axis = 2)

    padx = 51 - (np.shape(img)[1]%51)
    pady = 51 - (np.shape(img)[0]%51)

    img = np.pad(img, ((0,pady), (0,padx)), 'mean')

    count = 0
    mask = np.zeros((np.shape(img)[0], np.shape(img)[1]))
    for i in range(0, np.shape(img)[0], 51): #upto 3009
        for j in range(0, np.shape(img)[1], 51): #upto 3978
            inp = img[i:i+51, j:j+51]

            avg = np.mean(inp)
            var = np.var(inp)

            out = inp - avg
            window_mask = np.less(np.absolute(out), 2*np.ones((51,51))*sqrt(var)) + 0

            mask[i:i+51, j:j+51] = window_mask


X = np.asarray([[i,j,1] for i in range(np.shape(img)[1]) for j in range(np.shape(img)[0])])
W = np.diagflat(mask)
y = image.flatten()

beta = (X.T @ W @ X).I @ (X.T @ W @ y)
