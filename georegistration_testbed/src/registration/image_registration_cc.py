#!/usr/bin/env python

import cv2
from image_registration import *
import numpy as np
import scipy.optimize

class ImageRegistrationWithCrossCorrelation(ImageRegistration):
    """An object that runs image registration algorithms on image data."""

    def __init__(self):
        self.DEBUG = False
        # Define your image topi
        ImageRegistration.__init__(self)
        #Declare maximum iteration number for mutual information optimization
        self.maximum_iteration_number = 100
        self.GAUSSIAN_FILTER_SIZE = 15
        self.MOVING_IMAGE_SIZE = 100
        self.cross_correlation_multiscale = False
        self.Method = cv2.TM_SQDIFF # Squared Difference
        #self.Method = cv2.TM_SQDIFF_NORMED # Normalized Squared Difference 
        #self.Method = cv2.TM_CCORR # Cross Correlation
        #self.Method = cv2.TM_CCORR_NORMED # Normalized Cross Correlation
        #self.Method = cv2.TM_CCOEFF # Cross Coefficient
        #self.Method = cv2.TM_CCOEFF_NORMED # Normalized Cross Coefficient

    def toString(self):
        if (self.Method == cv2.TM_SQDIFF):
            method = "SQDIFF"
        if (self.Method == cv2.TM_SQDIFF_NORMED):
            method = "SQDIFF_NORMED"
        if (self.Method == cv2.TM_CCORR):
            method = "CCORR"
        if (self.Method == cv2.TM_CCORR_NORMED):
            method = "CCORR_NORMED"
        if (self.Method == cv2.TM_CCOEFF):
            method = "CCOEFF"
        if (self.Method == cv2.TM_CCOEFF_NORMED):
            method = "CCOEFF_NORMED"

        if (self.cross_correlation_multiscale == True):
            return "alg = CC max_iter = %s multiscaling_ON method = %s" % (self.maximum_iteration_number,method)
        else:
            return "alg = CC max_iter = %s multiscaling_OFF method = %s" % (self.maximum_iteration_number,method)

    @staticmethod
    def templateMatching(img,template,method):
        """ 
        Utilizes OpenCVs function for template matching and finding the min/max value
        https://docs.opencv.org/3.4/df/dfb/group__imgproc__object.html
        cv2.matchTemplate(image, templ, method[, result[, mask]]	) -> result
        https://docs.opencv.org/3.4/d2/de8/group__core__array.html
        cv2.minMaxLoc(	src[, mask]	) -> minVal, maxVal, minLoc, maxLoc
        """
        c = cv2.matchTemplate(img,template,method)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(c)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            return minVal
        else:
            return -1*maxVal

    @staticmethod
    def applyTransformation(imgIn, imgRefDims, transform):
        """
        Input = Image
        Output = Transformed Image
        """
        #self.iteration_number = self.iteration_number + 1
        #print('Image_Registration::Applying Transformation for iteration = ' + str(self.iteration_number))
        
        #theta_rad = transform[0]
        #shape_imgOut = imgRef.T.shape
        transformComplete = np.append(transform, [1.0], axis=0)
        projTransform = np.reshape(transformComplete, (3, 3))
        #translation_matrix = np.float32([[cos(theta_rad), -transform[1] * sin(theta_rad), transform[3]],
        #                                [transform[2] * sin(theta_rad), cos(theta_rad), transform[4]]])
        #imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags = cv2.INTER_CUBIC)
        imgOut = cv2.warpPerspective(imgIn, projTransform, (imgRefDims[1], imgRefDims[0]), flags=cv2.INTER_NEAREST)
        #imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags = cv2.INTER_LINEAR)
        return imgOut

    def calculateCrossCorrelation(self, image_moving_bw, image_ref, transformAsVec):
        """
        1. Image Transformation function call
        2. Cross Correlation function call
        3. Optimization
        """
        #print(str(len(image_moving_bw.ravel())))        
        warpedImage = ImageRegistrationWithCrossCorrelation.applyTransformation(image_moving_bw, image_ref.shape, transformAsVec)
        #print(str(len(transform.ravel())))
        #print(str(len(image_ref.ravel())))
        mutualInformation = ImageRegistrationWithCrossCorrelation.templateMatching(image_ref,warpedImage,self.Method)
        
        return mutualInformation

    def blur_and_subSample(self, imgIn, gaussian_window_size):
        if ((gaussian_window_size % 2) == 0):
            gaussian_window_size = gaussian_window_size - 1
        img_blurred = cv2.GaussianBlur(imgIn, (gaussian_window_size, gaussian_window_size), 0.3*gaussian_window_size)
        imgOut = cv2.pyrDown(img_blurred)        
        return imgOut

    def registerImagePair(self, image_moving, image_ref, initialTransform):
        """
        fminsearch matlab equivalent
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
        """
        self.DEBUG = True
        #image_ref_blurred = cv2.GaussianBlur(image_ref, (15, 15), 0)
        #scalef = float(150)/max(image_ref_blurred.shape[:2])
        #image_ref_scaled = cv2.resize(image_ref_blurred, (0, 0), fx=scalef, fy=scalef)        
        #image_ref_bw = ImageRegistration.convertImageColorSpace(image_ref)
        #image_ref_bw = image_ref
        if (len(image_ref.shape)>2):
            image_ref_bw = ImageRegistration.convertImageColorSpace(image_ref)
        else:
            image_ref_bw = image_ref
        if (len(image_moving.shape)>2):
            image_moving_bw = ImageRegistration.convertImageColorSpace(image_moving)
        else:
            image_moving_bw = image_moving

        if (self.DEBUG == True):
            print("initialTransform = %s" % str(initialTransform))
        transformAsVec = np.reshape(initialTransform, (1, 9))
        transformAsVec = transformAsVec[0, :8]

        if (self.cross_correlation_multiscale == True):
            image_ref_pyramid = image_ref_bw.copy()
            pyramid_level = int(np.ceil(np.log2(min(image_ref_bw.shape)) - np.log2(self.MOVING_IMAGE_SIZE)))
            if (self.DEBUG == True):
                print("PYRAMID LEVEL = %s" % str(pyramid_level))
                print("min_image_ref_bw_shape = %s" % str(min(image_ref_bw.shape)))
            for ii in range(pyramid_level):
                image_ref_pyramid = ImageRegistrationWithCrossCorrelation.blur_and_subSample(self, image_ref_pyramid, self.GAUSSIAN_FILTER_SIZE)
            #scalef = 0.5**pyramid_level
            image_ref_scaled = image_ref_pyramid
            #transformAsVec = transformAsVec * scalef
        else:
            scalef = float(self.MOVING_IMAGE_SIZE)/max(image_ref.shape[:2])
            image_ref_scaled = cv2.resize(image_ref_bw, (0, 0), fx=scalef, fy=scalef)
        #image_ref_scaled = cv2.GaussianBlur(image_ref_scaled, (15, 15), 0)
        #print("size %s" % str(image_ref_scaled.shape))
        
        noiseVec = np.random.normal(0, 1, (1, 8))
        noisy_transformAsVec = transformAsVec + 0.05*np.multiply(transformAsVec,noiseVec)
        #print("transformAsVec = %s" % str(transformAsVec))

        minimizationErrorFunction = lambda x: self.calculateCrossCorrelation(image_moving_bw, image_ref_scaled, x)
        
        xopt = scipy.optimize.fmin(func=minimizationErrorFunction, x0 = noisy_transformAsVec, args = (), xtol = 0.001, ftol = None, 
                                   maxiter = self.maximum_iteration_number, maxfun = 50, full_output = False, disp = False, retall = False,
                                   callback = None, initial_simplex = None)
        #if (self.cross_correlation_multiscale == True):
         #   xopt = xopt / scalef

        estTransformAsVec = np.append(xopt, [1.0], axis=0)
        Hest = np.reshape(estTransformAsVec, (3, 3))
        if (self.DEBUG == True):
            print("finalTransform = %s" % str(Hest))        
        mask = None
        self.DEBUG = False
        return (Hest, mask)
    
# example run via console
#
# ./image_registration.py ../../im22.jpg ../../im2.jpg
#
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print 'usage: %s image_moving image_reference' % sys.argv[0]
        sys.exit(1)

    image_moving_path = sys.argv[1]
    image_reference_path = sys.argv[2]

    #image_moving = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #image_reference = cv2.imread(img2_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    image_moving = cv2.imread(image_moving_path)
    image_reference = cv2.imread(image_reference_path)
    
    image_registration = ImageRegistrationWithCrossCorrelation()
    image_registration.SAVE_WARPEDIMAGE = True
    image_registration.OPENCV_SHOW_HOMOGRAPHY_BOUNDS_ON_REFERENCE = True
    #image_registration.OPENCV_SHOW_HOMOGRAPHY_MAXDIM = 512
    image_registration.registerImagePair(image_moving, image_ref=image_reference)
    image_registration.registerImagePairMultiScale(image_moving, image_ref=image_reference)
