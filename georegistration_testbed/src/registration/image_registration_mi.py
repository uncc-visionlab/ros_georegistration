#!/usr/bin/env python

import cv2
from image_registration import *
import numpy as np
import scipy.optimize

class ImageRegistrationWithMutualInformation(ImageRegistration):
    """An object that runs image registration algorithms on image data."""

    def __init__(self):
        self.DEBUG = False
        # Define your image topi
        ImageRegistration.__init__(self)
        self.numIntensityBins = 48
        #Declare maximum iteration number for mutual information optimization
        self.maximum_iteration_number = 100
        
    def toString(self):
        return "alg = MI max_iter = %s num_bins = %s" % (self.maximum_iteration_number, self.numIntensityBins)

    @staticmethod
    def histogramOutput(image_moving_bw, image_ref_bw, n_bins):
        """
        calculates histogram for equal sized moving image and reference image
        https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
        numpy.histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None)
        """
        histogram_out, x_edges, y_edges = np.histogram2d(image_moving_bw.ravel(), image_ref_bw.ravel(), bins=n_bins, range=None)
        #range=[[0,n_bins-1],[0,n_bins-1]])
        return histogram_out

    @staticmethod
    def mutualInformation(hgram):
        """ Mutual information for joint histogram
        Code taken from https://matthew-brett.github.io/teaching/mutual_information.html
        """
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1, keepdims=True) # marginal for x over y
        py = np.sum(pxy, axis=0, keepdims=True) # marginal for y over x
        #px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        px_py = px * py
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        vector1 = pxy[nzs]
        vector2 = px_py[nzs]
        vector2 = np.log2(pxy[nzs] / px_py[nzs])
        mutual_information = np.sum(vector1 * vector2)
        return mutual_information

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
        imgOut = cv2.warpPerspective(imgIn, projTransform, (imgRefDims[0], imgRefDims[1]), flags=cv2.INTER_NEAREST)
        #imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags = cv2.INTER_LINEAR)
        return imgOut

    def calculateMutualInformation(self, image_moving_bw, image_ref, transformAsVec):
        """
        1. Image Transformation function call
        2. Histogram function call
        3. Mutual Information function call
        4. Optimization
        """
        #print(str(len(image_moving_bw.ravel())))        
        warpedImage = ImageRegistrationWithMutualInformation.applyTransformation(image_moving_bw, image_ref.shape, transformAsVec)
        #print(str(len(transform.ravel())))
        #print(str(len(image_ref.ravel())))
        histogram2D_output = ImageRegistrationWithMutualInformation.histogramOutput(warpedImage, image_ref, n_bins=self.numIntensityBins)
        mutualInformation = ImageRegistrationWithMutualInformation.mutualInformation(histogram2D_output)
        
        return mutualInformation

    def registerImagePair(self, image_moving_bw, image_ref, initialTransform, add_noise=False):
        """
        fminsearch matlab equivalent
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
        """
        self.DEBUG = True
        #image_ref_blurred = cv2.GaussianBlur(image_ref, (15, 15), 0)
        #scalef = float(150)/max(image_ref_blurred.shape[:2])
        #image_ref_scaled = cv2.resize(image_ref_blurred, (0, 0), fx=scalef, fy=scalef)        
        #image_ref_bw = ImageRegistration.convertImageColorSpace(image_ref)
        image_ref_bw = image_ref
        scalef = float(100)/max(image_ref.shape[:2])
        image_ref_scaled = cv2.resize(image_ref_bw, (0, 0), fx=scalef, fy=scalef)
        #image_ref_scaled = cv2.GaussianBlur(image_ref_scaled, (15, 15), 0)
        #print("size %s" % str(image_ref_scaled.shape))
        if (self.DEBUG == True):
            print("initialTransform = %s" % str(initialTransform))
        transformAsVec = np.reshape(initialTransform, (1, 9))
        transformAsVec = transformAsVec[0, :8]

        if (add_noise == True):
            noiseVec = np.random.normal(0, 1, (1, 8))
            noisy_transformAsVec = transformAsVec + 0.025*np.multiply(transformAsVec,noiseVec)

        #print("transformAsVec = %s" % str(transformAsVec))

        minimizationErrorFunction = lambda x: -self.calculateMutualInformation(image_moving_bw, image_ref_scaled, x)
        
        xopt = scipy.optimize.fmin(func=minimizationErrorFunction, x0 = noisy_transformAsVec, args = (), xtol = 0.001, ftol = None, 
                                   maxiter = self.maximum_iteration_number, maxfun = 50, full_output = False, disp = False, retall = False,
                                   callback = None, initial_simplex = None)
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
    
    image_registration = ImageRegistrationWithMutualInformation()
    image_registration.SAVE_WARPEDIMAGE = True
    image_registration.OPENCV_SHOW_HOMOGRAPHY_BOUNDS_ON_REFERENCE = True
    #image_registration.OPENCV_SHOW_HOMOGRAPHY_MAXDIM = 512
    image_registration.registerImagePair(image_moving, image_ref=image_reference)
