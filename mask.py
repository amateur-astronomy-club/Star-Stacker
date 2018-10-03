import cv2
import numpy as np
from glob import glob
from math import fabs, sqrt

images = glob('Stars/*.jpeg')
np.set_printoptions(threshold=np.nan)

image_list = [cv2.imread(i) for i in images]

for image in image_list:
    img = np.sum(image, axis = 2)
    padx = 51 - (np.shape(img)[1]%51)
    pady = 51 - (np.shape(img)[0]%51)

    img = np.pad(img, ((0,pady), (0,padx)), 'median')

    count = 0
    for i in range(0, np.shape(img)[0], 51): #upto 3009
        for j in range(0, np.shape(img)[1], 51): #upto 3978
            inp = img[i:i+51, j:j+51]


            avg = np.mean(inp)
            var = np.var(inp)

            out = inp - avg

            mask = np.less(np.absolute(out), 2*np.ones((51,51))*sqrt(var)) + 0
            # print (np.shape(mask))
