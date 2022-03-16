#!/usr/bin/env python

import cv2
import numpy as np

class ImageRegistration(object):
    """An object that runs image registration algorithms on image data."""

    def __init__(self):
        # Define your image topic
        self.initialized = False
        self.OPENCV_VISUALIZE_MATCHES = True
        self.OPENCV_SHOW_HOMOGRAPHY_BOUNDS_ON_REFERENCE = False
        self.OPENCV_SHOW_HOMOGRAPHY_MAXDIM = 1600
        self.SAVE_WARPEDIMAGE = False
        self.MIN_FEATURES_FOR_ALIGNMENT = 10
        self.image_ref = None

    def toString(self):
        return "Null"
    
    @staticmethod
    def convertImageColorSpace(image):
        (height, width, channels) = image.shape[:3];
        if (channels >= 3):
            image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_bw = image[:, :, 1]
        return image_bw
    
    @staticmethod
    def resizeImage(image, maxdim_limit=1600):
        (height, width) = image.shape[:2]
        maxdim = max(height, width)
        resized = None
        scalef = 1.0
        if (maxdim > maxdim_limit):
            scalef = float(maxdim_limit) / maxdim;
            width = int(width * scalef)
            height = int(height * scalef)
            dim = (width, height)
            print('Resized Dimensions : ', dim)
            # resize image
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        else:
            resized = image.copy()
        return (resized, scalef)

    @staticmethod
    def fuseImage(moving_image, fixed_image, H_mov2fix):
        if (moving_image is None or fixed_image is None):
            return
        if (False):
            img = fixed_image.copy()
            blue = img[:, :, 0].copy()
            #cv2.fillPoly(blue, [np.int_(pts_transformed_homogenous[:, :2])], color=228, lineType=8, shift=0)
            img[:, :, 0] = blue
        else:
            fixed_bw = ImageRegistration.convertImageColorSpace(fixed_image)
            moving_bw = ImageRegistration.convertImageColorSpace(moving_image)
            moving_bw_xformed = cv2.warpPerspective(moving_bw, H_mov2fix, (fixed_bw.shape[1], fixed_bw.shape[0]))
            blue = np.zeros(fixed_bw.shape[:2], dtype=np.uint8)
            green = fixed_bw
            red = moving_bw_xformed
            # print("size (r,g,b)=(%s,%s,%s)" % (red.shape[:2], green.shape[:2], blue.shape[:2]))
            #cv2.fillPoly(blue, [np.int_(pts_transformed_homogenous[:, :2])], color=128, lineType=8, shift=0)
            img = np.dstack((blue, green, red)).astype(np.uint8)
        return img

