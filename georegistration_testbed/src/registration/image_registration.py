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
