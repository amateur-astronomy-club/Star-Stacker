import cv2
import numpy as np
from glob import glob
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from scipy import optimize

images = glob('Stars/out*.jpeg')

image_list = [cv2.imread(i) for i in images]

np.set_printoptions(threshold=np.nan)


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p




for image in image_list:
    image_gray = rgb2gray(image)

    blobs = blob_dog(image_gray, max_sigma=30, threshold=.1).astype('uint8')
    blobs[:, 2] = blobs[:, 2] * sqrt(2)

    # print(blobs)
    # print(np.shape(blobs))
    # print (blobs[0][0])
    for i in range(np.shape(blobs)[0]):
        x = blobs[i][1]
        y = blobs[i][0]

        if x < 32:
            x_lower = 0
            x_upper = 64
        elif x > np.shape(image)[1] - 32:
            x_lower = np.shape(image)[1] - 64
            x_upper = np.shape(image)[1]
        else:
            x_lower =  x - 32
            x_upper = x + 32


        if y < 32:
            y_lower = 0
            y_upper = 64
        elif y > np.shape(image)[0] - 32:
            y_lower = np.shape(image)[0] - 64
            y_upper = np.shape(image)[0]
        else:
            y_lower =  y - 32
            y_upper = y + 32
        inp = image[y_lower:y_upper,x_lower:x_upper,:]
        # print(inp)

        img = np.sum(inp, axis = 2)
        # print(img)

        avg = np.mean(img)
        var = np.var(img)

        out = img - avg

        window = (np.less(np.absolute(out), 2*np.ones((64,64))*sqrt(var)) + 0).astype('uint8')
        # print (window)
        for i in range(64):
            for j in range(64):
                image[y_lower+j,x_lower+i,:] *= window[j,i]

    cv2.imshow('SKYYYYY',cv2.resize(image,(900,600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
