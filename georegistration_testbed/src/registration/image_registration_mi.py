#!/usr/bin/env python

import cv2
from image_registration import *
from numericalldiff import *
import numpy as np
import scipy.optimize


class ImageRegistrationWithMutualInformation(ImageRegistration):
    """An object that runs image registration algorithms on image data."""

    def __init__(self):
        self.DEBUG = False
        # Define your image topi
        ImageRegistration.__init__(self)
        self.numIntensityBins = 48
        # Declare maximum iteration number for mutual information optimization
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
        histogram_out, x_edges, y_edges = np.histogram2d(image_moving_bw.ravel(), image_ref_bw.ravel(), bins=n_bins,
                                                         range=None)
        # delete the bin with value P(X=0,Y=y) from the histogram these are all counts of pixels inf the reference
        # image having locations that correspond to empty (off-image) pixels of the moving image
        np.delete(histogram_out, 0, 1)
        return histogram_out

    @staticmethod
    def mutualInformation(hgram):
        """ Mutual information for joint histogram
        Code taken from https://matthew-brett.github.io/teaching/mutual_information.html
        """
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1, keepdims=True)  # marginal for x over y
        py = np.sum(pxy, axis=0, keepdims=True)  # marginal for y over x
        # px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        px_py = px * py
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
        vector1 = pxy[nzs]
        # vector2 = px_py[nzs]
        vector2 = np.log2(pxy[nzs] / px_py[nzs])
        mutual_information = np.sum(vector1 * vector2)
        return mutual_information

    @staticmethod
    def applyTransformation(imgIn, imgRefDims, projTransform):
        """
        Input = Image
        Output = Transformed Image
        """
        # self.iteration_number = self.iteration_number + 1
        # print('Image_Registration::Applying Transformation for iteration = ' + str(self.iteration_number))

        # theta_rad = transform[0]
        # shape_imgOut = imgRef.T.shape
        # transformComplete = np.append(transform, [1.0], axis=0)
        # projTransform = np.reshape(transformComplete, (3, 3))
        # translation_matrix = np.float32([[cos(theta_rad), -transform[1] * sin(theta_rad), transform[3]],
        #                                [transform[2] * sin(theta_rad), cos(theta_rad), transform[4]]])
        # imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags = cv2.INTER_CUBIC)
        image_out = cv2.warpPerspective(imgIn, projTransform, (imgRefDims[1], imgRefDims[0]),
                                        flags=cv2.INTER_NEAREST, borderValue=0)
        # imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags = cv2.INTER_LINEAR)
        window_title = 'perspective transformed image'
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, image_out)
        cv2.resizeWindow(window_title, 600, 600)
        cv2.waitKey(10)
        return image_out

    def calculateMutualInformation(self, image_moving, image_ref, transformAsVec):
        """
        1. Image Transformation function call
        2. Histogram function call
        3. Mutual Information function call
        4. Optimization
        """
        # print(str(len(image_moving_bw.ravel())))
        transformComplete = np.append(transformAsVec, [1.0], axis=0)
        H_mov2fix = np.reshape(transformComplete, (3, 3))
        warpedImage = ImageRegistrationWithMutualInformation.applyTransformation(image_moving, image_ref.shape,
                                                                                 H_mov2fix)
        # print(str(len(transform.ravel())))
        # print(str(len(image_ref.ravel())))
        histogram2D_output = ImageRegistrationWithMutualInformation.histogramOutput(warpedImage, image_ref,
                                                                                    n_bins=self.numIntensityBins)
        mutualInformation = ImageRegistrationWithMutualInformation.mutualInformation(histogram2D_output)
        print('MI = ' + str(mutualInformation))
        fusedImage = ImageRegistration.fuseImage(image_moving, image_ref, H_mov2fix)
        window_title = 'moving-to-fixed image matches'
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, fusedImage)
        cv2.resizeWindow(window_title, 600, 600)
        cv2.waitKey(10)
        return mutualInformation

    def registerImagePair(self, image_moving_bw, image_ref, initialTransform=np.eye(3), add_noise=False):
        """
        fminsearch matlab equivalent
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
        """
        self.DEBUG = True
        # image_ref_blurred = cv2.GaussianBlur(image_ref, (15, 15), 0)
        # scalef = float(150)/max(image_ref_blurred.shape[:2])
        # image_ref_scaled = cv2.resize(image_ref_blurred, (0, 0), fx=scalef, fy=scalef)
        # image_ref_bw = ImageRegistration.convertImageColorSpace(image_ref)
        image_ref_bw = image_ref
        # scalef = float(100)/max(image_ref.shape[:2])
        # image_ref_scaled = cv2.resize(image_ref_bw, (0, 0), fx=scalef, fy=scalef)
        # image_ref_scaled = cv2.GaussianBlur(image_ref_scaled, (15, 15), 0)
        image_ref_scaled = image_ref_bw
        # print("size %s" % str(image_ref_scaled.shape))
        if (self.DEBUG == True):
            print("initialTransform = %s" % str(initialTransform))
        transformAsVec = np.reshape(initialTransform, (1, 9))
        transformAsVec = transformAsVec[0, :8]

        if (add_noise == True):
            noiseVec = np.random.normal(0, 1, (1, 8))
            noisy_transformAsVec = transformAsVec + 0.15 * np.multiply(transformAsVec, noiseVec)
        else:
            noisy_transformAsVec = transformAsVec
        # print("transformAsVec = %s" % str(transformAsVec))

        minimizationErrorFunction = lambda x: -self.calculateMutualInformation(image_moving_bw, image_ref_scaled, x)
        minimizationErrorFunction_df = NumericalDiff(minimizationErrorFunction, 8, 1, 'central')
        jacobianErrorFunction = lambda x: minimizationErrorFunction_df.jacobian(x)
        hessianErrorFunction = lambda x: minimizationErrorFunction_df.hessian(x)
        #minimize_method = 'Nelder-Mead'
        minimize_method = 'Newton-CG'
        #minimize_method = 'CG'
        #minimize_method = 'SLSQP'

        #hessianMethod = '3-point'
        hessianMethod = hessianErrorFunction

        #jacobianMethod = 'None'
        #jacobianMethod = '3-point'
        jacobianMethod = jacobianErrorFunction

        maximum_iterations = 1000

        transform_bounds = [[-10, 10], [-10, 10], [-1000, 1000], [-10, 10], [-10, 10], [-1000, 1000], [-1, 1], [-1, 1]];
        noisy_transformAsVec = scipy.optimize.fmin(func=minimizationErrorFunction, x0=noisy_transformAsVec, args=(), xtol=0.001,
                                          ftol=None,
                                          maxiter=self.maximum_iteration_number, maxfun=100, full_output=False,
                                          disp=False,
                                          retall=False,
                                          callback=None, initial_simplex=None)
        if False:
            x_optimized = scipy.optimize.fmin(func=minimizationErrorFunction, x0=noisy_transformAsVec, args=(),
                                              xtol=0.1,
                                              ftol=0.1,
                                              maxiter=self.maximum_iteration_number, maxfun=500, full_output=False,
                                              disp=False,
                                              retall=False,
                                              callback=None, initial_simplex=None)
        else:
            optimized_result = scipy.optimize.minimize(fun=minimizationErrorFunction,
                                                       x0=noisy_transformAsVec,
                                                       method=minimize_method,
                                                       jac=jacobianMethod,
                                                       hess=hessianMethod, hessp=None, bounds=transform_bounds,
                                                       constraints=(), tol=0.001,
                                                       callback=None, options={'maxiter': maximum_iterations})
            x_optimized = optimized_result.x

        hessian = hessianMethod(x_optimized)
        print('x_optimized = ' + str(x_optimized))
        print('Hessian = ' + str(hessian))
        estTransformAsVec = np.append(x_optimized, [1.0], axis=0)
        Hest = np.reshape(estTransformAsVec, (3, 3))
        fused_image = ImageRegistration.fuseImage(image_moving_bw, image_ref, Hest)
        print('Saving image ' + 'registration_result_image.png')
        cv2.imwrite('registration_result_image_.png', fused_image)
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
        print('usage: %s image_moving image_reference' % sys.argv[0])
        sys.exit(1)

    image_moving_path = sys.argv[1]
    image_reference_path = sys.argv[2]

    # image_moving = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # image_reference = cv2.imread(img2_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    image_moving = cv2.imread(image_moving_path)
    image_reference = cv2.imread(image_reference_path)
    image_moving = cv2.blur(image_moving, (40, 40))
    image_reference = cv2.blur(image_reference, (40, 40))
    scale_factor = 1.0 / 10.0
    image_moving = cv2.resize(image_moving,
                              (
                                  int(image_moving.shape[1] * scale_factor), int(image_moving.shape[0] * scale_factor)))
    image_reference = cv2.resize(image_reference, (
        int(image_reference.shape[1] * scale_factor), int(image_reference.shape[0] * scale_factor)))
    image_moving_bw = ImageRegistration.convertImageColorSpace(image_moving)
    image_reference_bw = ImageRegistration.convertImageColorSpace(image_reference)

    image_registration = ImageRegistrationWithMutualInformation()
    image_registration.SAVE_WARPEDIMAGE = True
    image_registration.OPENCV_SHOW_HOMOGRAPHY_BOUNDS_ON_REFERENCE = True
    # image_registration.OPENCV_SHOW_HOMOGRAPHY_MAXDIM = 512
    initialGuess = np.matrix(
        '1.54878482e+00 -3.19168050e-02 -4.74815444e+02; 2.36119902e-01  1.33933203e+00 -1.32688449e+02; 5.52313139e-04 -2.22734174e-05 1.00000000e+00');
    scale_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    # initialGuess = np.matmul(np.matmul(np.linalg.inv(scale_matrix), initialGuess), scale_matrix)
    initialGuess = np.matmul(np.matmul(scale_matrix, initialGuess), np.linalg.inv(scale_matrix))
    # initialGuess = np.matmul(scale_matrix, initialGuess)
    # image_registration.registerImagePair(image_moving, image_ref=image_reference, initialTransform=initialGuess)
    image_registration.registerImagePair(image_moving_bw, image_ref=image_reference_bw,
                                         initialTransform=initialGuess, add_noise=True)
