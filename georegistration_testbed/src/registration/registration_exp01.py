#!/usr/bin/env python

from image_registration import *
from image_registration_mi import ImageRegistrationWithMutualInformation
import numpy as np
import re


def read_csv(file_name):
    float_regex = '[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    pattern = re.compile(float_regex)
    file = open(file_name, 'r')
    for line in file.readlines():
        str_values = line.rstrip().split(',')  # using rstrip to remove the \n
        #print(str_values)
        row_values = []
        for element in str_values:
            if pattern.match(element):
                row_values.append(float(element))
        if 'ground_truth_values' in locals():
            ground_truth_values = np.vstack([ground_truth_values, [row_values]])
        else:
            ground_truth_values = np.asmatrix(row_values)
    return ground_truth_values


# example run via console
#
# ./image_registration.py ../../im22.jpg ../../im2.jpg
#
if __name__ == '__main__':
    root_folder = 'dataset_SAR_EO'
    fixed_folder = 'fixed'
    moving_folder = 'moving'
    homography_file = 'gt_dataset_homography.csv'
    fixed_filename_prefix = 'registration_fixed_image_excerpt_'
    fixed_filename_suffix = '.png'
    moving_filename_prefix = 'registration_moving_image_'
    moving_filename_suffix = '.png'
    ground_truth_values = read_csv(root_folder + '/' + homography_file)
    for row in range(0,ground_truth_values.shape[0]):
        index_value = int(ground_truth_values[row, 0])
        H_ground_truth = np.reshape(ground_truth_values[row, 1:], (3, 3))
        image_moving_path = root_folder + '/' + moving_folder + '/' + moving_filename_prefix + str(
            index_value) + moving_filename_suffix
        image_reference_path = root_folder + '/' + fixed_folder + '/' + fixed_filename_prefix + str(
            index_value) + fixed_filename_suffix

        # image_moving = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # image_reference = cv2.imread(img2_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        image_moving = cv2.imread(image_moving_path)
        image_reference = cv2.imread(image_reference_path)
        image_moving = cv2.blur(image_moving, (10, 10))
        image_reference = cv2.blur(image_reference, (80, 80))
        scale_factor = 1.0 / 1.0
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
        initialGuess = H_ground_truth
        scale_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
        # initialGuess = np.matmul(np.matmul(np.linalg.inv(scale_matrix), initialGuess), scale_matrix)
        initialGuess = np.matmul(np.matmul(scale_matrix, initialGuess), np.linalg.inv(scale_matrix))
        # initialGuess = np.matmul(scale_matrix, initialGuess)
        # image_registration.registerImagePair(image_moving, image_ref=image_reference, initialTransform=initialGuess)
        transformAsVec = np.squeeze(np.asarray(initialGuess.flatten()))
        transformAsVec = transformAsVec[:6]
        #transformAsVec[2] = transformAsVec[2] + 15
        #transformAsVec[5] = transformAsVec[5] + 15
        num_params = np.size(transformAsVec)
        add_noise = True
        if (add_noise == True):
            noiseVec = np.random.normal(0, 1, num_params)
            transformAsVec = transformAsVec + 0.8 * np.multiply(transformAsVec, noiseVec)
        else:
            transformAsVec = transformAsVec

        image_registration.registerImagePair(image_moving_bw, image_ref=image_reference_bw,
                                             initialTransform=transformAsVec, add_noise=True)
