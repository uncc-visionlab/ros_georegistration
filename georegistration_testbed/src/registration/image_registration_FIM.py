#!/usr/bin/env python
import math
import cv2
# from image_registration import *
from numericalldiff import *
from image_registration_mi_updated import *
from image_registration_node_updated import *
import numpy as np
import scipy.optimize
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.search.random_search import RandomSearch
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.backend.topology import Pyramid
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class ImageRegistrationFIM(ImageRegistration):
    """An object that runs image registration algorithms on image data."""

    def __init__(self):
        self.DEBUG = False
        # Define your image topi
        ImageRegistration.__init__(self)
        self.numIntensityBins = 8
        # Declare maximum iteration number for mutual information optimization
        self.maximum_iteration_number = 50
        self.transformation_perturbation = 0.2
        self.transformation_perturbation_xy = 0.3  # tx, ty (b1, b2)
        self.transformation_perturbation_theta = 0.05  # a1, a2, a3, a4
        self.transformation_perturbation_cxcy = 0  # c1, c2
        # self.Affine_Xform = False
        # self.MI_Image_Registration = True # MI on = True, MI off = False
        self.OPT_DEFAULT = False
        self.OPT_NelderMeid = False
        self.OPT_PSO = False  # Particle Swarm Optimization (PSO)
        self.OPT_PSO_LocalBest = False  # PSO Local best optimization
        self.xy_grid_points = 20
        self.xy_grid_search = False
        self.optimize_x = False
        self.optimize_xy = True
        self.optimize_6DOF = True
        self.optimize_total = False

        self.Affine_Xform = False
        ##### FIM #####
        self.reference_gazebo_model_size_x = 7643
        self.reference_gazebo_model_size_y = 7345
        self.fx = 110
        self.fy = self.fx
        self.cx = 50
        self.cy = self.cx

    def toString(self):
        if (self.Affine_Xform == True):
            return "alg = MI Affine max_iter = %s num_bins = %s " % (
            self.maximum_iteration_number, self.numIntensityBins)
        else:
            return "alg = MI Perspective max_iter = %s num_bins = %s" % (
            self.maximum_iteration_number, self.numIntensityBins)

    def applyTransformation_pso(self, imgIn, imgRefDims, params):
        """
        Input = Image
        Output = Transformed Image
        """
        # print("b1 and b2 (applyTransformation_pso) = %s %s " % (str(self.b1_grid.copy()), str(self.b2_grid.copy())))
        # print("transformAsVec_params = %s" % str(params))
        # transform = self.GroundTruth
        if (self.Affine_Xform == True):
            transform = params
            affineTransform = np.reshape(transform, (2, 3))
            imgOut = cv2.warpAffine(imgIn, affineTransform, (imgRefDims[1], imgRefDims[0]), flags=cv2.INTER_NEAREST)
        else:
            transform = self.GroundTruth
            if (self.optimize_x):
                transform[2] = params[0]
            elif (self.optimize_xy):
                transform[2] = params[0]
                transform[5] = params[1]
            elif (self.optimize_6DOF):
                if (self.xy_grid_search == True):
                    transform[2] = self.b1_grid.copy()
                    transform[5] = self.b2_grid.copy()
                    transform[:2] = params[:2]
                    transform[3:5] = params[3:5]
                else:
                    transform[:6] = params
            elif (self.optimize_total):
                if (self.xy_grid_search == True):
                    transform[2] = self.b1_grid.copy()
                    transform[5] = self.b2_grid.copy()
                    transform[:2] = params[:2]
                    transform[3:5] = params[3:5]
                    transform[6:8] = params[6:8]
                else:
                    transform[:8] = params
            # print("applyTransformation = %s" % str(transform))
            transformComplete = np.append(transform, [1.0], axis=0)
            projTransform = np.reshape(transformComplete, (3, 3))
            imgOut = cv2.warpPerspective(imgIn, projTransform, (imgRefDims[1], imgRefDims[0]), flags=cv2.INTER_NEAREST)
        return imgOut

    # Particle Swarm Optimization
    def calculateMutualInformation_pso(self, image_moving_bw, image_ref, transformAsVec_pso):
        """
        1. Image Transformation function call
        2. Histogram function call
        3. Mutual Information function call
        4. Optimization
        """
        mi_pso = np.zeros(np.shape(transformAsVec_pso)[0])
        for ii in np.arange(np.shape(transformAsVec_pso)[0]):
            transformAsVec = transformAsVec_pso[ii]
            warpedImage = self.applyTransformation_pso(image_moving_bw, image_ref.shape, transformAsVec)
            temp_multiplier = np.ones(image_ref.shape[:2])
            temp_array = np.multiply(warpedImage.copy(), temp_multiplier)
            image_ref_temp = np.multiply(temp_array, image_ref.copy())
            histogram2D_output = ImageRegistrationWithMutualInformation.histogramOutput(warpedImage, image_ref_temp,
                                                                                        n_bins=self.numIntensityBins)
            # histogram2D_output = ImageRegistrationWithMutualInformation.histogramOutput(warpedImage, image_ref, n_bins=self.numIntensityBins)
            mutualInformation = ImageRegistrationWithMutualInformation.mutualInformation(histogram2D_output)
            mi_pso[ii] = mutualInformation
        return mi_pso

    # @staticmethod
    def applyTransformation_fim(self, imgIn, imgRefDims, params):
        """
        Input = Image
        Output = Transformed Image
        """

        if (self.Affine_Xform == True):
            transform = params
            affineTransform = np.reshape(transform, (2, 3))
            imgOut = cv2.warpAffine(imgIn, affineTransform, (imgRefDims[1], imgRefDims[0]), flags=cv2.INTER_NEAREST)
        else:
            transform = params
            transformComplete = np.append(transform, [1.0], axis=0)
            projTransform = np.reshape(transformComplete, (3, 3))
            imgOut = cv2.warpPerspective(imgIn, projTransform, (imgRefDims[1], imgRefDims[0]), flags=cv2.INTER_NEAREST)
        # translation_matrix = np.float32([[cos(theta_rad), -transform[1] * sin(theta_rad), transform[3]],
        #                                [transform[2] * sin(theta_rad), cos(theta_rad), transform[4]]])
        # imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags = cv2.INTER_CUBIC)
        # imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags = cv2.INTER_LINEAR)
        return imgOut

    # Particle Swarm Optimization
    def calculateMutualInformation_fim(self, image_moving_bw, image_ref, transformAsVec):
        """
        1. Image Transformation function call
        2. Histogram function call
        3. Mutual Information function call
        4. Optimization
        """
        self.ROS_VISUALIZE_SENSOR_VIEW_FRUSTUM = False
        warpedImage = self.applyTransformation_fim(image_moving_bw, image_ref.shape, transformAsVec)
        # warpedImage = warpedImage.T
        # print("size (r,g,b)=(%s,%s,%s)" % (warpedImage.shape[:2], image_ref.shape[:2], image_moving_bw.shape[:2]))
        if (self.ROS_VISUALIZE_SENSOR_VIEW_FRUSTUM == True and image_ref is not None):
            if (False):
                img = image_ref.copy()
                blue = img[:, :, 0].copy()
                cv2.fillPoly(blue, [np.int_(pts_transformed_homogenous[:, :2])], color=228, lineType=8, shift=0)
                img[:, :, 0] = blue
            else:
                fixed_bw = image_ref
                moving_bw = image_moving_bw
                moving_bw_xformed = warpedImage
                blue = np.zeros(fixed_bw.shape[:2], dtype=np.uint8)
                green = fixed_bw
                red = warpedImage
                # print("size (r,g,b)=(%s,%s,%s)" % (red.shape[:2], green.shape[:2], blue.shape[:2]))
                # cv2.fillPoly(blue, [np.int_(pts_transformed_homogenous[:, :2])], color=128, lineType=8, shift=0)
                img = np.dstack((blue, green, red)).astype(np.uint8)

            window_title = 'sensor-to-reference image matches'
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.imshow(window_title, img)
            cv2.resizeWindow(window_title, 600, 600)
            cv2.waitKey(5)
        # if (self.ROS_VISUALIZE_SENSOR_VIEW_FRUSTUM == True and image_ref is not None):
        #   window_title = 'warpedImage'
        # cv2.fillConvexPoly(img, pn, color, lineType=8, shift=0)
        #  cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        # cv2.imshow(window_title, warpedImage)
        # cv2.resizeWindow(window_title, 600, 600)
        # cv2.waitKey(5)
        # print("warpedImage = %s" % str(warpedImage))
        temp_multiplier = np.ones(image_ref.shape[:2])
        temp_array = np.multiply(warpedImage.copy(), temp_multiplier)
        image_ref_temp = np.multiply(temp_array, image_ref.copy())
        histogram2D_output = ImageRegistrationWithMutualInformation.histogramOutput(warpedImage, image_ref_temp,
                                                                                    n_bins=self.numIntensityBins)
        mutualInformation = ImageRegistrationWithMutualInformation.mutualInformation(histogram2D_output)
        return mutualInformation

    def registerImagePair(self, image_moving, image_ref, GroundTruth, noisy_perturbation, AFFINE=True, OPT_NM=True,
                          OPT_PSO=False):
        """
        fminsearch matlab equivalent
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
        """
        self.GroundTruth = GroundTruth
        ADD_GAUSSIAN_NOISE = False
        if ADD_GAUSSIAN_NOISE:
            noise_dim = 8
            noiseVec = np.random.normal(0, 1, (1, noise_dim))[0]  # perspective
            noise_a1, noise_a2, noise_b1, noise_a3, noise_a4, noise_b2, noise_c1, noise_c2 = np.multiply(
                self.GroundTruth[:8].copy(), noiseVec.copy())
        else:
            noise_dim = 8
            noise_a1, noise_a2, noise_b1, noise_a3, noise_a4, noise_b2, noise_c1, noise_c2 = np.ones(noise_dim)
        print("GT_registerImagePair = %s" % str(self.GroundTruth))
        perturbation_xy = noisy_perturbation[0]
        perturbation_theta = noisy_perturbation[1]
        perturbation_cxcy = noisy_perturbation[2]
        gt_a1, gt_a2, gt_b1, gt_a3, gt_a4, gt_b2, gt_c1, gt_c2 = self.GroundTruth[:8]

        min_a1, max_a1 = gt_a1 - noise_a1 * perturbation_theta, gt_a1 + noise_a1 * perturbation_theta
        min_a2, max_a2 = gt_a2 - noise_a2 * perturbation_theta, gt_a2 + noise_a2 * perturbation_theta
        min_a3, max_a3 = gt_a3 - noise_a3 * perturbation_theta, gt_a3 + noise_a3 * perturbation_theta
        min_a4, max_a4 = gt_a4 - noise_a4 * perturbation_theta, gt_a4 + noise_a4 * perturbation_theta
        min_b1, max_b1 = gt_b1 - noise_b1 * perturbation_xy, gt_b1 + noise_b1 * perturbation_xy
        min_b2, max_b2 = gt_b2 - noise_b2 * perturbation_xy, gt_b2 + noise_b2 * perturbation_xy
        min_c1, max_c1 = gt_c1 - noise_c1 * perturbation_cxcy, gt_c1 + noise_c1 * perturbation_cxcy
        min_c2, max_c2 = gt_c2 - noise_c2 * perturbation_cxcy, gt_c2 + noise_c2 * perturbation_cxcy
        noisy_transformAsVec = np.array([max_a1, max_a2, max_b1, max_a3, max_a4, max_b2, max_c1, max_c2])
        print("noisy_transformAsVec = %s" % str(noisy_transformAsVec))
        if (OPT_NM == True and OPT_PSO == False):
            self.OPT_NelderMeid = True
            self.OPT_PSO = False
        elif (OPT_NM == False and OPT_PSO == True):
            self.OPT_NelderMeid = False
            self.OPT_PSO = True
        elif (OPT_NM == True and OPT_PSO == True):
            self.OPT_NelderMeid = False
            self.OPT_PSO = True  # Pick PSO #Mixed optimizer (future work)
        else:
            self.OPT_DEFAULT = True
        if (AFFINE == True):
            self.Affine_Xform = True
        else:
            self.Affine_Xform = False
            self.optimize_xy = True
        self.DEBUG = True

        # image_ref_blurred = cv2.GaussianBlur(image_ref, (15, 15), 0)
        # scalef = float(150)/max(image_ref_blurred.shape[:2])
        # image_ref_scaled = cv2.resize(image_ref_blurred, (0, 0), fx=scalef, fy=scalef)
        # image_ref_bw = ImageRegistration.convertImageColorSpace(image_ref)
        if (len(image_ref.shape) > 2):
            image_ref_bw = ImageRegistration.convertImageColorSpace(image_ref)
        else:
            image_ref_bw = image_ref
        if (len(image_moving.shape) > 2):
            image_moving_bw = ImageRegistration.convertImageColorSpace(image_moving)
        else:
            image_moving_bw = image_moving

        image_ref_scaled = image_ref_bw
        # scalef = float(100)/max(image_ref.shape[:2])
        # image_ref_scaled = cv2.resize(image_ref_bw, (0, 0), fx=scalef, fy=scalef)
        # image_ref_scaled = cv2.GaussianBlur(image_ref_scaled, (15, 15), 0)
        # print("size %s" % str(image_ref_scaled.shape))
        # if (self.DEBUG == True):
        #   print("initialTransform = %s" % str(initialTransform))

        minimizationErrorFunction = lambda x: -self.calculateMutualInformation(image_moving_bw, image_ref_scaled, x)
        minimizationErrorFunction_pso = lambda x: -self.calculateMutualInformation_pso(image_moving_bw,
                                                                                       image_ref_scaled, x)

        if (self.OPT_DEFAULT == True):
            xopt_default = scipy.optimize.fmin(func=minimizationErrorFunction, x0=noisy_transformAsVec, args=(),
                                               xtol=0.1, ftol=0.1,
                                               maxiter=self.maximum_iteration_number,
                                               maxfun=self.maximum_iteration_number, full_output=False, disp=False,
                                               retall=False,
                                               callback=None, initial_simplex=None)
            xopt = xopt_default
        if (self.OPT_NelderMeid == True):
            # Create bounds
            # if (self.Affine_Xform == True):
            #   dim = 6
            #  min_bound, max_bound = np.array([min_a1, min_a2, min_b1, min_a3, min_a4, min_b2]), np.array([max_a1, max_a2, max_b1, max_a3, max_a4, max_b2])
            # else:
            #   dim = 8
            #  min_bound, max_bound = np.array([min_a1, min_a2, min_b1, min_a3, min_a4, min_b2, min_c1, min_c2]), np.array([max_a1, max_a2, max_b1, max_a3, max_a4, max_b2, max_c1, max_c2])
            # bounds = (min_bound, max_bound)
            noisy_transformAsVec = np.array([max_a1, max_a2, max_b1, max_a3, max_a4, max_b2, max_c1, max_c2])
            xopt_nm = scipy.optimize.minimize(minimizationErrorFunction, x0=noisy_transformAsVec, args=(),
                                              method='Nelder-Mead', bounds=None, options={'disp': True})
            # ,bounds=None, tol=None, callback=None, options={'func': None, 'maxiter': None, 'maxfev': None,
            # 'disp': False, 'return_all': False,'initial_simplex': None, 'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False})
            xopt = xopt_nm.x
        if (self.OPT_PSO == True):
            pso_plot_cost = False
            pso_random_search = False
            pso_grid_search = False
            # Initialize swarm
            swarm = 40
            pso_iter = 20
            # options = {'w': 0.9, 'c1': 0.5, 'c2': 0.3} #options = {'w': 0.9, 'c1': 2, 'c2': 2} #options = {'w': 0.5069, 'c1': 2.5524, 'c2': 1.0056} #options = {'w': -0.3699, 'c1': -0.1207, 'c2': 3.3657}
            # options_local = {'c1': 1.4063935432825423, 'c2': 2.806884743926418, 'w': 0.07076560365339613, 'k': 6, 'p': 1}
            # options_global = {'w': -0.3699, 'c1': -0.1207, 'c2': 3.3657}
            options_global = {'w': -0.3699, 'c1': 2.05, 'c2': 3.3657}
            # options_global = {'w': 0.9, 'c1': 0.5, 'c2': 3.3657}
            # options_global = {'c1': 1.4063935432825423, 'c2': 2.806884743926418, 'w': 0.07076560365339613}
            # Create bounds
            # https: // pyswarms.readthedocs.io / en / latest / _modules / pyswarms / utils / search / random_search.html  # RandomSearch
            options_grid = {'c1': [-0.1207, 2.5524], 'c2': [0.3, 3.3657], 'w': [-0.3699, 0.9], 'k': [5, 15], 'p': 1}
            if (self.Affine_Xform == True):
                dim = 6
                min_bound, max_bound = .8 * GroundTruth[0:dim], 1.2 * GroundTruth[0:dim]
            else:
                if (self.optimize_x):
                    xopt = self.GroundTruth
                    dim = 1
                    max_b = np.array([max_b1])
                    min_b = np.array([min_b1])
                    bounds_x = (min_b, max_b)
                    optimizer = ps.single.GlobalBestPSO(n_particles=swarm, dimensions=dim, options=options,
                                                        bounds=bounds_x)
                    # optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=dim, options=options, bounds=bounds_xy)
                    cost, xopt_pso_global_x = optimizer.optimize(objective_func=minimizationErrorFunction_pso,
                                                                 iters=pso_iter)
                    if (pso_plot_cost):
                        plot_cost_history(cost_history=optimizer.cost_history)
                        plt.show()
                    xopt[2] = xopt_pso_global_x
                    # xopt[5] = xopt_pso_global_x[1]
                if (self.xy_grid_search == True and pso_grid_search == False and pso_random_search == False):
                    # xopt = self.GroundTruth
                    # dim = 2
                    # min_b, max_b = np.array([int(min_b1), int(min_b2)]), np.array([int(max_b1), int(max_b2)])
                    # bounds_xy = (min_b, max_b)
                    # print("bounds_xy options = %s" % str(bounds_xy))
                    print("xy_grid_search = True")
                    min_b1, max_b1, min_b2, max_b2 = int(min_b1.copy()), int(max_b1.copy()), int(min_b2.copy()), int(
                        max_b2.copy())
                    xv, yv = np.meshgrid(np.linspace(min_b1, max_b1, self.xy_grid_points),
                                         np.linspace(min_b2, max_b2, self.xy_grid_points), indexing='ij')
                    ny, nx = len(yv), len(xv)
                    mi_arr = np.zeros(ny * nx)
                    for i in range(ny):
                        for j in range(nx):
                            b1_tmp, b2_tmp = xv[i, j], yv[i, j]
                            # self.optimize_xy = True
                            xx = np.array([b1_tmp, b2_tmp])
                            mi_tmp = minimizationErrorFunction(xx)
                            mi_arr[i * nx + j] = -mi_tmp
                    mi_arr = np.float32(mi_arr)
                    # print("mi_arr = %s" % str(mi_arr))
                    max_value = max(mi_arr)
                    # print("mi_tmp_max = %s" % str(max_value))
                    mi_max_ind = [i for i, x in enumerate(mi_arr) if x == max_value][0]
                    # print("mi_max_ind = %s" % str(mi_max_ind))
                    ix, jy = mi_max_ind // nx, mi_max_ind % nx
                    # print("mi_tmp_max_index = %s %s" % (str(ix), str(jy)))
                    b1_grid, b2_grid = xv[ix, jy], yv[ix, jy]
                    self.b1_grid = b1_grid
                    self.b2_grid = b2_grid
                    # print("mi_tmp_max_val = %s %s" % (str(b1_grid), str(b2_grid)))
                    # options_grid_xy = {'b1': [min_b1, max_b1],
                    #               'b2': [min_b1, max_b2]}
                    # g = GridSearch(ps.single.GlobalBestPSO, n_particles=swarm, dimensions=dim, options=options_grid_xy,
                    #                objective_func=minimizationErrorFunction_pso, iters=pso_iter, bounds=bounds_xy)
                    # print("g options = %s" % str(g.options))

                    # best_score, best_options = g.search()
                    # print("best_score = %s" % str(best_score))
                    # print("best_options = %s" % str(best_options))
                    # optimizer = ps.single.GlobalBestPSO(n_particles=swarm, dimensions=dim, options=options_local, bounds=bounds_xy)
                    # optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=dim, options=options_local, bounds=bounds_xy)
                    # cost, xopt_pso_global_xy = optimizer.optimize(objective_func=minimizationErrorFunction_pso, iters=pso_iter)
                    # if (pso_plot_cost):
                    #  plot_cost_history(cost_history=optimizer.cost_history)
                    # plt.show()
                    # xopt[2] = xopt_pso_global_xy[0]
                    # xopt[5] = xopt_pso_global_xy[1]
                if (self.optimize_6DOF == True):
                    if (self.xy_grid_search == True):
                        print("xy_grid_search = True and optimize_6DOF = True")
                        print("b1 and b2 = %s %s " % (str(self.b1_grid.copy()), str(self.b2_grid.copy())))
                        xopt = self.GroundTruth.copy()
                        xopt[2] = self.b1_grid.copy()
                        xopt[5] = self.b2_grid.copy()
                        dim = 4
                        max_b = np.array([max_a1, max_a2, max_a3, max_a4])
                        min_b = np.array([min_a1, min_a2, min_a3, min_a4])
                        bounds_4DOF = (min_b, max_b)
                        optimizer = ps.single.GlobalBestPSO(n_particles=swarm, dimensions=dim, options=options_global,
                                                            bounds=bounds_4DOF)
                        # options_local = {'w': 0.5069, 'c1': 2.5524, 'c2': 1.0056, 'k': 10, 'p': 2}
                        # optimizer = ps.single.LocalBestPSO(n_particles=swarm, dimensions=dim, options=options_local, bounds=bounds_6DOF)
                        cost, xopt_pso_global_4DOF = optimizer.optimize(objective_func=minimizationErrorFunction_pso,
                                                                        iters=pso_iter)
                        xopt[:2] = xopt_pso_global_4DOF[:2]
                        xopt[3:5] = xopt_pso_global_4DOF[2:4]
                        print("xopt (xy_grid_search = True and optimize_6DOF = True) = %s" % str(xopt))

                    else:
                        print("xy_grid_search = False and optimize_6DOF = True")
                        xopt = self.GroundTruth
                        dim = 6
                        max_b = np.array([max_a1, max_a2, max_b1, max_a3, max_a4, max_b2])
                        min_b = np.array([min_a1, min_a2, min_b1, min_a3, min_a4, min_b2])
                        bounds_6DOF = (min_b, max_b)
                        optimizer = ps.single.GlobalBestPSO(n_particles=swarm, dimensions=dim, options=options_global,
                                                            bounds=bounds_6DOF)
                        # options_local = {'w': 0.5069, 'c1': 2.5524, 'c2': 1.0056, 'k': 10, 'p': 2}
                        # optimizer = ps.single.LocalBestPSO(n_particles=swarm, dimensions=dim, options=options_local, bounds=bounds_6DOF)
                        cost, xopt_pso_global_6DOF = optimizer.optimize(objective_func=minimizationErrorFunction_pso,
                                                                        iters=pso_iter)
                        if (pso_plot_cost):
                            plot_cost_history(cost_history=optimizer.cost_history)
                            plt.show()
                        xopt[:6] = xopt_pso_global_6DOF[:6]
                if (self.optimize_total == True):
                    if (self.xy_grid_search == True):
                        print("xy_grid_search = True and optimize_total = True")
                        xopt = self.GroundTruth
                        xopt[2] = self.b1_grid
                        xopt[5] = self.b2_grid
                        dim = 6
                        max_bound = np.array([max_a1, max_a2, max_a3, max_a4, max_c1, max_c2])
                        min_bound = np.array([min_a1, min_a2, min_a3, min_a4, min_c1, min_c2])
                        bounds = (min_bound, max_bound)
                        optimizer = ps.single.GlobalBestPSO(n_particles=swarm, dimensions=dim, options=options_global,
                                                            bounds=bounds)
                        cost, xopt_pso_global = optimizer.optimize(objective_func=minimizationErrorFunction_pso,
                                                                   iters=pso_iter)
                        xopt[:2] = xopt_pso_global[:2]
                        xopt[3:5] = xopt_pso_global[2:4]
                        if (math.isnan(xopt_pso_global[4])):
                            xopt[6] = 0
                        if (math.isnan(xopt_pso_global[5])):
                            xopt[7] = 0
                    else:
                        print("xy_grid_search = False and optimize_total = True")
                        dim = 8
                        max_bound = np.array([max_a1, max_a2, max_b1, max_a3, max_a4, max_b2, max_c1, max_c2])
                        min_bound = np.array([min_a1, min_a2, min_b1, min_a3, min_a4, min_b2, min_c1, min_c2])
                        bounds = (min_bound, max_bound)
                        # Call the instance of PSO
                        optimizer = ps.single.GlobalBestPSO(n_particles=swarm, dimensions=dim, options=options_global,
                                                            bounds=bounds)
                        # optimizer = ps.single.LocalBestPSO(n_particles=swarm, dimensions=dim, options=options_local, bounds=bounds)
                        # print("optimizer = %s" % str(optimizer))
                        # my_topology = Pyramid(static=False)
                        # Call instance of GlobalBestPSO
                        # optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=dim, options=options, topology=my_topology, bounds=bounds)
                        cost, xopt_pso_global = optimizer.optimize(objective_func=minimizationErrorFunction_pso,
                                                                   iters=pso_iter)
                        if (pso_plot_cost):
                            plot_cost_history(cost_history=optimizer.cost_history)
                            plt.show()
                        xopt = xopt_pso_global
                        if (math.isnan(xopt_pso_global[6])):
                            xopt[6] = 0
                        if (math.isnan(xopt_pso_global[7])):
                            xopt[7] = 0

                if (pso_random_search == True and self.optimize_xy == True and self.optimize_6DOF == True):
                    max_b = np.array([max_b1, max_b2])
                    min_b = np.array([min_b1, min_b2])
                    bounds_xy = (min_b, max_b)
                    dim = len(min_b)
                    bounds_xy = (min_b, max_b)
                    # n_selection_iters = number of iterations of random parameter selection
                    g = RandomSearch(ps.single.LocalBestPSO, n_particles=swarm, dimensions=dim, options=options_grid,
                                     objective_func=minimizationErrorFunction_pso, iters=pso_iter,
                                     n_selection_iters=20, bounds=bounds_xy)
                    print("g options = %s" % str(g.options))

                    best_score, best_options = g.search()
                    print("best_score = %s" % str(best_score))
                    print("best_options = %s" % str(best_options))

                    options_updated = {'w': best_options['w'], 'c1': best_options['c1'], 'c2': best_options['c2'],
                                       'k': best_options['k'], 'p': 1}
                    max_b = np.array([max_a1, max_a2, max_b1, max_a3, max_a4, max_b2])
                    min_b = np.array([min_a1, min_a2, min_b1, min_a3, min_a4, min_b2])
                    bounds_6DOF = (min_b, max_b)
                    dim_updated = len(min_b)
                    optimizer_updated = ps.single.LocalBestPSO(n_particles=swarm, dimensions=dim_updated,
                                                               options=options_updated, bounds=bounds_6DOF)
                    cost, xopt_pso_global_6DOF = optimizer_updated.optimize(
                        objective_func=minimizationErrorFunction_pso, iters=pso_iter)
                    xopt = self.GroundTruth
                    xopt[:6] = xopt_pso_global_6DOF[:6]

        if (self.Affine_Xform == True):
            xopt = np.append(xopt, [0, 0], axis=0)
        estTransformAsVec = np.append(xopt, [1.0], axis=0)
        Hest = np.reshape(estTransformAsVec, (3, 3))
        if (self.DEBUG == True):
            print("finalTransform = %s" % str(Hest))
        mask = None
        self.DEBUG = False
        return (Hest, mask)

    @staticmethod
    def rotation3D(pose_rotation_matrix):
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        """
        r = R.from_matrix(pose_rotation_matrix)
        return r.as_euler('zyx', degrees=True)

    def calculatePose_fim(self, transformAsVec):
        """
        :param transformAsVec: homography parameters
        :return: 6 DoF Pose
        """
        cameraK = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        bbox = self.bbox
        transformComplete = np.append(transformAsVec, [1.0], axis=0)
        Hrgb_estimated = np.reshape(transformComplete, (3, 3))
        (height, width) = (100, 100)
        pts_homogeneous_image_corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                                                 dtype=np.float32)

        pts_transformed_homogeneous_estimated = \
            cv2.perspectiveTransform(pts_homogeneous_image_corners[None, :, :], Hrgb_estimated)[0]
        # pts_transformed_homogeneous_estimated = pts_transformed_homogenous
        m_per_pixel_XY = np.asarray((float(self.reference_gazebo_model_size_x) / self.image_ref.shape[1],
                                     float(self.reference_gazebo_model_size_y) / self.image_ref.shape[0]),
                                    dtype=np.float32)

        center_of_image_XY = 0.5 * np.asarray((self.image_ref.shape[1], self.image_ref.shape[0]), dtype=np.float32)

        for idx in range(0, pts_transformed_homogeneous_estimated.shape[0]):
            pts_transformed_homogeneous_estimated[idx, :] += [bbox[0][0], bbox[1][0]]
            # compensate for world coordinate XY=(0,0) is referenced to the image center
            pts_transformed_homogeneous_estimated[idx, :] -= center_of_image_XY
            # pts_transformed_homogeneous_estimated[:2, idx] *= [m_per_pixel_y, m_per_pixel_x]
            pts_transformed_homogeneous_estimated[idx, :] = np.multiply(
                pts_transformed_homogeneous_estimated[idx, :],
                m_per_pixel_XY)
            # gazebo's Y axis is flipped
            pts_transformed_homogeneous_estimated[idx, 1] *= -1
        # concatenate the Z coordinate for all points and set it to 0
        pts_transformed_homogeneous_estimated = np.concatenate((pts_transformed_homogeneous_estimated.T,
                                                                np.zeros(
                                                                    (1, pts_transformed_homogeneous_estimated.shape[0]),
                                                                    dtype=np.float32)), axis=0)

        moving_image_xy_dims = (100, 100)
        image_corners = np.array([[0, 0], [moving_image_xy_dims[0] - 1, 0],
                                  [moving_image_xy_dims[0] - 1, moving_image_xy_dims[1] - 1],
                                  [0, moving_image_xy_dims[1] - 1]], dtype=np.float32)
        objPoints = pts_transformed_homogeneous_estimated.transpose()
        imagePoints = image_corners
        invPoseTransform_est_perspective_PSO = ROSImageRegistrationNode.OPENCV_SOLVEPNP(objPoints, imagePoints, cameraK,
                                                                                        cv2.SOLVEPNP_ITERATIVE)
        # print("invPoseTransform_est_perspective_PSO = %s" % str(invPoseTransform_est_perspective_PSO[:3, :4]))
        pose_rotation_matrix = invPoseTransform_est_perspective_PSO[:3, :3]
        pose_translation_vector = invPoseTransform_est_perspective_PSO[:3, 3]

        pose_YPR = ImageRegistrationFIM.rotation3D(pose_rotation_matrix)
        pose_YPR_XYZ = np.concatenate((pose_YPR, pose_translation_vector), axis=0)
        # print("pose_YPR_XYZ = %s " % str(pose_YPR_XYZ))
        # return np.reshape(invPoseTransform_est_perspective_PSO[:3, :4], (1,12))[0]
        return pose_YPR_XYZ

    def fisherInformation(self, image_moving, image_ref, xvals, uv_corners):
        """
        :param image_moving: Sensed SAR image
        :param image_ref: Reference RGB Image
        :param xvals: Estimated Homography
        :return: Fisher Information Matrix : $$I(\theta)]_{i,j} =
                        -E_{f(X;\theta)}[\frac{\delta^2}{\delta\theta_i\delta\theta_j}logf(X;\theta)$$
        """
        # print("xvals = %s " % str(xvals))
        if (len(image_ref.shape) > 2):
            image_ref_bw = ImageRegistration.convertImageColorSpace(image_ref)
        else:
            image_ref_bw = image_ref
        if (len(image_moving.shape) > 2):
            image_moving_bw = ImageRegistration.convertImageColorSpace(image_moving)
        else:
            image_moving_bw = image_moving

        FIM_MI = True
        FIM_POSE = True
        if FIM_MI == True:
            f = lambda x: math.log(self.calculateMutualInformation_fim(image_moving_bw, image_ref_bw, x))
            df = NumericalDiff(f, len(xvals), 1, 'central')
            sigmaHessian = df.hessian(xvals)
        if FIM_POSE == True:
            ref_poly = uv_corners[:, [0, 1, 3, 2]]
            bbox = ROSImageRegistrationNode.boundingBox(ref_poly)
            BBOX_MARGIN = 100
            bbox[1][0] = bbox[1][0] - BBOX_MARGIN
            bbox[0][0] = bbox[0][0] - BBOX_MARGIN
            bbox[1][1] = bbox[1][1] + BBOX_MARGIN
            bbox[0][1] = bbox[0][1] + BBOX_MARGIN
            self.image_ref = image_ref_bw
            self.bbox = bbox
            # f = lambda x: np.log(self.calculatePose_fim(x))
            ff = lambda x: self.calculatePose_fim(x)
            df = NumericalDiff(ff, len(xvals), 6, 'central')
            f_Jacobian = df.jacobian(xvals)

        FIM_sigmaHessian = sigmaHessian
        FIM_f_Jacobian = f_Jacobian
        return FIM_sigmaHessian, FIM_f_Jacobian

    def invMatrix(self, matrix):
        """
        :param matrix: matrix
        :return: inverse matrix
        """
        # check for square matrix ?
        return np.mat(matrix).I

    def SVD(self, FIM):
        """
        https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        :param FIM:
        :return:
        """
        # check for square matrix ?
        u, s, vh = np.linalg.svd(FIM, full_matrices=True, hermitian=True)
        return u, s, vh


# example run via console
#
# ./image_registration.py ../../im22.jpg ../../im2.jpg
#
if __name__ == '__main__':
    # import sys
    import os

    # if len(sys.argv) < 3:
    #   print('usage: %s image_moving image_reference' % sys.argv[0])
    #  sys.exit(1)

    DATADIR = "dataset_registration/dataset_rollar_coaster/"
    # image_moving_path = sys.argv[1]
    # image_reference_path = sys.argv[2]
    path_reference = os.path.join(DATADIR, "fixed")
    path_moving = os.path.join(DATADIR, "moving")
    reference_dataset = []
    moving_dataset = []
    for img in os.listdir(path_reference):
        img_ref_array = cv2.imread(os.path.join(path_reference, img), cv2.IMREAD_GRAYSCALE)
        reference_dataset.append(img_ref_array)
    for img in os.listdir(path_moving):
        img_mov_array = cv2.imread(os.path.join(path_moving, img), cv2.IMREAD_GRAYSCALE)
        moving_dataset.append(img_mov_array)
    file = 'gt_dataset_rollar_coaster.txt'
    filename = os.path.join(DATADIR, file)
    if file == 'gt_dataset_rollar_coaster.txt':
        cameraPoseArray = np.loadtxt(filename, delimiter=',',
                                     usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
        gTruthArray = np.loadtxt(filename, delimiter=',', usecols=(17, 18, 19, 20, 21, 22, 23, 24, 25))
        RefPolyArray = np.loadtxt(filename, delimiter=',', usecols=(26, 27, 28, 29, 30, 31, 32, 33))
    idx = 35
    gTruth = gTruthArray[idx][:8]
    ImageRegistrationFIMnode = ImageRegistrationFIM()

    # noisy_perturbation = np.array([0.2, 0.2, 0.001])
    # Hest, mask = ImageRegistrationFIMnode.registerImagePair(moving_dataset[idx], reference_dataset[idx], gTruth, noisy_perturbation, AFFINE = False, OPT_NM = False, OPT_PSO = True)
    # H_est = np.reshape(Hest.copy(), (1, 9))[0]
    H_est = gTruth.copy()
    np.savetxt('results_registration/estimatedHomography.txt', H_est, fmt='%1.4e')
    sigmaHessian, f_Jacobian = ImageRegistrationFIMnode.fisherInformation(moving_dataset[idx], reference_dataset[idx],
                                                                          H_est[:8],
                                                                          np.reshape(RefPolyArray[idx, :], (2, 4)))
    sigmaH = sigmaHessian.T[0].T
    np.savetxt('results_registration/sigmaHessian.txt', sigmaH, fmt='%1.4e')
    np.savetxt('results_registration/jacobian.txt', f_Jacobian, fmt='%1.4e')
    print('FIM:  sigmaHessian = %s ' % str(sigmaH))
    print('FIM: f_Jacobian = %s ' % str(f_Jacobian))
    H = np.matmul(f_Jacobian, np.matmul(sigmaH, f_Jacobian.T))
    np.savetxt('results_registration/Hessian.txt', H, fmt='%1.4e')
    print('FIM: H = %s ' % str(H))
    covM = ImageRegistrationFIMnode.invMatrix(H)
    np.savetxt('results_registration/covarianceMatrix.txt', covM, fmt='%1.4e')
    print('FIM: covM = %s ' % str(covM))
    # u, s, vh = ImageRegistrationFIMnode.SVD(covM)
    # print('U = %s ' % str(u))
    # print('S = %s ' % str(np.sqrt(s)))
    # print('VH = %s ' % str(vh))
    # vary_noise = True
    # if vary_noise == True:
    #     noise_idx = 10
    #     noise_perturbation = np.zeros(noise_idx)
    #     FisherI = np.zeros([noise_idx, 8])
    #     VarM = np.zeros([noise_idx, 8])
    #     for ii in range(noise_idx):
    #         noise_factor = (ii + 1 - noise_idx//2)*0.01
    #         noise_perturbation[ii] = noise_factor
    #         print('noise_factor = %s ' % str(noise_factor))
    #         gt = gTruth[:8].copy()
    #         noisy_gt = gt + gt*noise_factor
    #         print('noisy_gt = %s ' % str(noisy_gt))
    #         #print('RefPoly = %s ' % str(RefPolyArray[idx, :]))
    #         FIM = ImageRegistrationFIMnode.fisherInformation(moving_dataset[idx], reference_dataset[idx], noisy_gt, np.reshape(RefPolyArray[idx, :], (2, 4)))
    #         #covM = np.mat(FIM).I
    #         covM = ImageRegistrationFIMnode.invMatrix(FIM)
    #         u, s, vh = ImageRegistrationFIMnode.SVD(covM)
    #         VarM[ii, :] = s
    #         #for jj in range(8):
    #          #   FisherI[ii, jj] = FIM[jj,jj]
    #           #  VarM[ii, jj] = covM[jj, jj]
    #         #print('COV = %s ' % str(covM))
    #     Hparams = ['$a_1$', '$a_2$', '$b_1$', '$a_3$', '$a_4$', '$b_2$', '$c_1$', '$c_2$']
    #     FIM_COV_PLOT = True
    #     if (FIM_COV_PLOT == True):
    #         for kk in range(8):
    #             fig = plt.figure(figsize=(3, 2))
    #             #plt.plot(noise_perturbation, FisherI[:, kk]/max(FisherI[:, kk]), label=str(Hparams[kk]))
    #             #plt.plot(noise_perturbation, VarM[:, kk] / max(VarM[:, kk]), label=str(Hparams[kk]))
    #             plt.plot(np.arange(1,noise_idx+.01, 1), np.sqrt(VarM[:, kk]), label=str(Hparams[kk]))
    #             plt.xlabel('frame number', fontweight="bold", fontsize=10)
    #             plt.ylabel('standard deviation', fontweight="bold", fontsize=10)