#!/usr/bin/env python2

# ROS imports
# rospy for the subscriber
import math
import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import datetime as time
from enum import Enum
from georegistration_gazebo_msgs.srv import *
from georegistration_testbed.msg import *
from image_geometry import PinholeCameraModel
from image_registration_mi import *
from image_registration_cc import *
# from image_registration_mi_ocl import *
from image_registration_ocv_features import *
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import rosgraph_msgs.msg
import re
import timeit

class Algorithm(Enum):
    OPENCV_FEATURES = 1
    MUTUAL_INFORMATION = 2 
    MUTUAL_INFORMATION_OCL = 3
    CROSS_CORRELATION = 4

class ROSImageRegistrationNode(object):
    """An object representing a ROS Image Registration Node."""
    
    def __init__(self):
        # Debug and visualization flags
        self.DEBUG = True
        self.ROS_VISUALIZE_SENSOR_VIEW_FRUSTUM = True
        self.VISUALIZE_TRAJECTORY = True
        self.USE_REFERENCE_FOR_TRAJECTORY = False
        self.TRAJECTORY_LINE_SIZE = 4
        # Uncomment the line below for simulated EO images and OpenCV features for image alignment
        #self.ALGORITHM = Algorithm.OPENCV_FEATURES
        #self.ALGORITHM = Algorithm.MUTUAL_INFORMATION
        self.ALGORITHM = Algorithm.CROSS_CORRELATION
        # Uncomment the line below for simulated SAR images and mutual information for image alignment
        self.USE_SIMULATED_RGB_MOVING_IMAGE = False
        self.LOGGING = True

        self.ready = False
        super(ROSImageRegistrationNode, self).__init__()

        if (self.LOGGING == True):
            #datetime_str = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
            datetime_str = str(time.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            self.datafile = open('/tmp/rosgeoregistration_' + datetime_str + '.log', 'w')
        
        if (self.ALGORITHM == Algorithm.OPENCV_FEATURES):
            self.algorithm = ImageRegistrationWithOpenCVFeatures()
        elif (self.ALGORITHM == Algorithm.MUTUAL_INFORMATION):
            self.algorithm = ImageRegistrationWithMutualInformation()
        elif (self.ALGORITHM == Algorithm.MUTUAL_INFORMATION_OCL):
            self.algorithm = ImageRegistrationWithMutualInformationOpenCL()
        elif (self.ALGORITHM == Algorithm.CROSS_CORRELATION):
            self.algorithm = ImageRegistrationWithCrossCorrelation()    
            
        # Instantiate CvBridge
        self.bridge = CvBridge()
        # Define your image topic        
        self.hasCameraInfo = False
        self.reference_image_filename = rospy.get_param('~reference_image', None)
        self.rgb_sensor_image_filename = rospy.get_param('~rgb_sensor_image', None)
        self.reference_gazebo_model_name = rospy.get_param('~model_name', 'ground_plane')
        self.reference_gazebo_model_size_x = rospy.get_param('~model_size_x', 1.0)
        self.reference_gazebo_model_size_y = rospy.get_param('~model_size_y', 1.0)
        self.ALGORITHM_DETECT = rospy.get_param('~ALGORITHM_DETECT', 'ORB')
        self.ALGORITHM_EXTRACT = rospy.get_param('~ALGORITHM_EXTRACT', 'ORB')
        
        self.rgb_image_sub = rospy.Subscriber('camera/image_raw', Image, self.rgbImageCallback, queue_size=5)
        self.camera_info_sub = rospy.Subscriber('camera/camera_info', CameraInfo, self.rgbCameraInfoCallback, queue_size=1)
        self.sar_image_sub = rospy.Subscriber('sar/image_raw', Image, self.sarImageCallback, queue_size=5)
        self.sar_camera_view_sub = rospy.Subscriber('sar/camera_view', SARCameraView, self.sarCameraViewCallback, queue_size=5)
        self.waypoint_msg_sub = rospy.Subscriber('/rosout_agg', rosgraph_msgs.msg.Log, self.waypointMessageCallback, queue_size=1)
        
        self.prev_SAR_image_msg = None
        self.prev_RGB_image_msg = None
        self.register_target_image = None
        self.register_query_image = None
        self.prev_homography = None
        self.prev_nav_state = None
        self.prev_coordinate = None
        self.traj_image = None
        self.waypoint_num = "-1"
        
        self.ref_image_match_pub = rospy.Publisher('sar_registered/ref_image_match', Image, queue_size=1)
        self.register_moving_image_pub = rospy.Publisher('sar_registered/register_moving_image', Image, queue_size=1)
        self.register_fixed_image_pub = rospy.Publisher('sar_registered/register_fixed_image', Image, queue_size=1)
        self.rgb_truth_homography_pub = rospy.Publisher('sar_registered/truth_rgb_homography_image', Image, queue_size=1)
        self.registered_result_pub = rospy.Publisher('sar_registered/fused_registration_image', Image, queue_size=1)
        
        self.frame_id = 0

        self.pinhole_camera_model = PinholeCameraModel()
        rospy.loginfo('image_registration::Reference Inputfile = %s' % str(self.reference_image_filename))
        if (self.reference_image_filename):
            self.image_ref = cv2.imread(self.reference_image_filename)
            if (self.rgb_sensor_image_filename != None and self.rgb_sensor_image_filename != self.reference_image_filename):
                rospy.loginfo('image_registration::Sensor Inputfile = %s' % str(self.rgb_sensor_image_filename))
                self.image_sensed_src = cv2.imread(self.rgb_sensor_image_filename)
            else:
                self.image_sensed_src = self.image_ref
            #rospy.loginfo('shape = %s' % str(image_ref.shape[:3]))
            #self.image_ref = self.image_ref[:,range(0,image_ref.shape[1],10),:];

            if (self.USE_REFERENCE_FOR_TRAJECTORY == True):
                self.traj_image = self.image_ref.copy()
            else:
                self.traj_image = self.image_sensed_src.copy()

            rospy.loginfo('image_registration::Read reference image %s.' % self.reference_image_filename)
            self.ref_scale = 1.0
            self.image_ref = cv2.resize(self.image_ref, (0, 0), fx=self.ref_scale, fy=self.ref_scale, interpolation=cv2.INTER_NEAREST)
            self.ready = True
        else:
            rospy.logerr('image_registration::Error - no reference image found.')
    
        #self.modelsize = ROSImageRegistrationNode.getModelSize(self.reference_gazebo_model_name)
        #if (self.modelsize is None):
        #    self.modelsize = np.array([1, 1, 1]);      
            
    @staticmethod
    def computePoseError(T_ref, T_est):
        roll_yaw_pitch_deg_xyz = np.zeros((6,1))
        for cIdx in range(2, -1, -1):
            dotprod = np.dot(T_ref[:3,cIdx],T_est[:3,cIdx])
            if (abs(dotprod) > 1):
                roll_yaw_pitch_deg_xyz[cIdx] = 0;
            else:
                roll_yaw_pitch_deg_xyz[cIdx] = math.acos(dotprod)*180.0/np.pi
            out_of_plane_axis = np.cross(T_ref[:3,cIdx],T_est[:3,cIdx])
            if (np.dot(out_of_plane_axis,T_ref[:3,(cIdx+2)%3]) > 0):
                roll_yaw_pitch_deg_xyz[cIdx] *= -1
        # compute translation error
        roll_yaw_pitch_deg_xyz[3:6,0] = T_est[:3, 3] - T_ref[:3, 3]
        return roll_yaw_pitch_deg_xyz
    
    @staticmethod
    def boundingBox(coords2d):
        bbox = np.zeros((2, 2), dtype=np.uint32)
        bbox[:, 0] = np.min(coords2d, axis=1)
        bbox[:, 1] = np.max(coords2d, axis=1)
        return bbox

    @staticmethod
    def getModelSize(model_name, gazebo_namespace='/gazebo'):
        rospy.loginfo('image_registration::Waiting for ' + gazebo_namespace + '/get_model_size service to become available.')
        rospy.wait_for_service(gazebo_namespace + '/get_model_size', 5)
        try:
            model_geom_props = rospy.ServiceProxy(gazebo_namespace + '/get_model_size', GetModelGeomProperties)
            rospy.loginfo("image_registration::Calling service %s/get_model_size" % gazebo_namespace)            
            resp = model_geom_props(model_name)
            #rospy.loginfo('image_registration::received msg %s' % str(resp))
            if (len(resp.geom_sizes) > 0):
                rospy.loginfo('image_registration::Got model %s size = %s' % (model_name, str(resp.geom_sizes[0])))
                return np.array([resp.geom_sizes[0].x, resp.geom_sizes[0].y, resp.geom_sizes[0].z])
            else:
                rospy.logerr('image_registration::Got no response on model %s size.' % model_name)
                return None
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr('image_registration::Service ' + gazebo_namespace + '/get_model_size call failed: %s' % e)
            pass
        return None

    def logfileNewline(self):
        self.datafile.write('\n')
        
    def dumpAsStringToLog(self, val):
        self.datafile.write("\"%s\", " % str(val))
        
    def dumpNumpyDataToLog(self, matrix):
        numElements = np.prod(matrix.shape)
        outputArr = np.reshape(matrix, (1, numElements))
        lastElem = numElements - 1
        for idx in range(0, outputArr.shape[1]):
            self.datafile.write("%.6g" % outputArr[0, idx])
            #if (idx < lastElem):
            self.datafile.write(", ")
            #else:
            #    self.datafile.write("\n")

    def waypointMessageCallback(self, rosout_msg):
        '''
        Message style to look for
        STRAIGHT: Vehicle has reached waypoint 2
        Because there's no publishing of the waypoints, need to look at the rosout log for waypoints
        '''
        #print("RECEIVED MESSAGE: %s" % rosout_msg.msg)
        w = re.search(r'(\D+): Vehicle has reached waypoint (\d+)', rosout_msg.msg)
        if w:
            self.waypoint_num = w.group(2)
            #print("WAYPOINT RECEIVED: %d" % int(w.group(2)))

    def sarCameraViewCallback(self, camera_view_msg):
        MIN_IMAGE_DIMENSION = 50
        self.frame_id = camera_view_msg.header.seq

        try:
            if (self.image_ref is None):
                return
			#Timing for algorithms
            start = timeit.default_timer()
            #print("Got SAR camera view message")
            uv_corners = np.reshape(camera_view_msg.plane_uv_coords, (2, 4))
            #print("uv_corners = %s" % str(uv_corners))
            homography = np.reshape(camera_view_msg.homography, (3, 3))
            #print("homography = %s" % str(homography))
            xyz_corners = np.reshape(camera_view_msg.plane_xyz_coords, (3, 4))
            # swap the bottom right and bottom left corners for solvePnP and other functions
            xyz_corners[:, [3, 2]] = xyz_corners[:, [2, 3]]
            #print("xyz_corners = %s" % str(xyz_corners))
            cameraK = np.reshape(camera_view_msg.K, (3, 3))
            #print("camera_K = %s" % str(cameraK))
            cameraPose = np.reshape(camera_view_msg.pose, (4, 4))
            # Collect current coordinates
            current_coordinate = tuple(cameraPose[:2,3])
            if (self.DEBUG == True):
                print("camera_pose = %s" % str(cameraPose))
                        
            # compute the SAR sensor view polygon in the RGB reference image
            ref_poly = uv_corners[:, [0, 1, 3, 2]]
            for i in range(0, ref_poly.shape[1]):
                ref_poly[0, i] = ref_poly[0, i] * self.image_ref.shape[1]
                ref_poly[1, i] = ref_poly[1, i] * self.image_ref.shape[0]
                            
            # nadir view of image block from the rgb reference image using the bounding box of the homography
            bbox = ROSImageRegistrationNode.boundingBox(ref_poly)
            BBOX_MARGIN = 100
            bbox[1][0] = bbox[1][0] - BBOX_MARGIN
            bbox[0][0] = bbox[0][0] - BBOX_MARGIN
            bbox[1][1] = bbox[1][1] + BBOX_MARGIN
            bbox[0][1] = bbox[0][1] + BBOX_MARGIN
            uv_target_image = self.image_ref[bbox[1][0]:bbox[1][1], bbox[0][0]:bbox[0][1]].copy()

            bbox_poly = uv_corners[:, [0, 1, 3, 2]]            
            for i in range(0, bbox_poly.shape[1]):
                bbox_poly[0, i] = ref_poly[0, i] - bbox[0][0]
                bbox_poly[1, i] = ref_poly[1, i] - bbox[1][0]

            #print('ref_poly %s' % str(ref_poly))
            #print('bbox_poly %s' % str(bbox_poly))
                       
            # homography of bbox to the reference image
            if (self.USE_SIMULATED_RGB_MOVING_IMAGE == True):
                # RGB moving image
                RGB_SIMULATED_IMAGE_XY_DIMS = (700, 700)
                cameraK[0,:] = cameraK[0,:]*RGB_SIMULATED_IMAGE_XY_DIMS[1] / (2*cameraK[0,2])
                cameraK[1,:] = cameraK[1,:]*RGB_SIMULATED_IMAGE_XY_DIMS[0] / (2*cameraK[1,2])
                target_image_xy_dims = (RGB_SIMULATED_IMAGE_XY_DIMS[1], RGB_SIMULATED_IMAGE_XY_DIMS[0])
                image_corners = np.array([[0, 0], [target_image_xy_dims[0]-1, 0], 
                                         [target_image_xy_dims[0]-1, target_image_xy_dims[1]-1], 
                                         [0, target_image_xy_dims[1]-1]], dtype=np.float32)
                # compute reference image to simulated RGB image transform
                # compute the SAR sensor view polygon in the RGB reference image
                sensed_rgb_poly = uv_corners[:, [0, 1, 3, 2]]
                for i in range(0, sensed_rgb_poly.shape[1]):
                    sensed_rgb_poly[0, i] = sensed_rgb_poly[0, i] * self.image_sensed_src.shape[1]
                    sensed_rgb_poly[1, i] = sensed_rgb_poly[1, i] * self.image_sensed_src.shape[0]
                Hrgb_ref_truth = cv2.getPerspectiveTransform(np.float32(sensed_rgb_poly.transpose()), image_corners)
                # generate a simulated RGB image in the SAR sensor from the reference RGB image
                rgb_homography_target_image = cv2.warpPerspective(self.image_sensed_src, Hrgb_ref_truth, RGB_SIMULATED_IMAGE_XY_DIMS)
                self.register_moving_image = rgb_homography_target_image
                self.register_fixed_image = uv_target_image
            else:
                # SAR moving image
                #self.register_target_image = uv_target_image
                self.register_moving_image = self.register_query_image
                self.register_fixed_image = uv_target_image

            moving_image_xy_dims = (self.register_moving_image.shape[1], self.register_moving_image.shape[0])
            #print("moving_image_xy_dims = (%s,%s)" % (self.register_moving_image.shape[1], self.register_moving_image.shape[0]))
            image_corners = np.array([[0, 0], [moving_image_xy_dims[0]-1, 0], 
                                     [moving_image_xy_dims[0]-1, moving_image_xy_dims[1]-1], 
                                     [0, moving_image_xy_dims[1]-1]], dtype=np.float32)
                                     
            #Hrgb_ref_truth = cv2.getPerspectiveTransform(np.float32(ref_poly.transpose()), image_corners)
            #Hrgb_ref_truth = Hrgb_ref_truth / Hrgb_ref_truth[2, 2]
            Hrgb_bbox_truth = cv2.getPerspectiveTransform(np.float32(bbox_poly.transpose()), image_corners)
            Hrgb_bbox_truth = Hrgb_bbox_truth / Hrgb_bbox_truth[2, 2]
            
            #Haffine_rgb_ref_truth = cv2.getAffineTransform(np.float32(bbox_poly[:,:3].transpose()), image_corners[:3,:])
            #print("Hperspective = %s" % Hrgb_bbox_truth)
            #print("Haffine = %s" % Haffine_rgb_ref_truth)
            #rospy.loginfo('image_registration::Image[%s], H_perspective = %s' % (str(self.frame_id), str(Hrgb_bbox_truth)))
            #rospy.loginfo('image_registration::Image[%s], H_affine = %s' % (str(self.frame_id), str(Haffine_rgb_ref_truth)))

            (ret, rotation_vector_tru, translation_vector_tru) = cv2.solvePnP(objectPoints=np.ascontiguousarray(xyz_corners.transpose().reshape((4,1,3))),
                                                                      imagePoints=np.ascontiguousarray(image_corners).reshape((4,1,2)), cameraMatrix=cameraK, 
                                                                      distCoeffs=None, flags=cv2.SOLVEPNP_P3P) 
                                                                    # cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_P3P, 
                                                                    # cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_DLS


                #, useExtrinsicGuess=True, rvec=rvec0, tvec=T0)
            (poseRMatrix_tru, Rjacobian_tru) = cv2.Rodrigues(rotation_vector_tru)
            poseTransform_tru = np.eye(4)
            poseTransform_tru[:3,:3] = poseRMatrix_tru
            poseTransform_tru[:3, 3] = translation_vector_tru[:, 0]
            invPoseTransform_tru = np.linalg.inv(poseTransform_tru)
            invPoseTransform_tru[:3,:3] = np.transpose(invPoseTransform_tru[:3,:3]);
            invPoseTransform_tru[0,:3] = -invPoseTransform_tru[0,:3]
            invPoseTransform_tru[1,:3] = -invPoseTransform_tru[1,:3]
            #print("poseEstimate_tru = %s" % poseTransform_tru)
            yaw_pitch_roll_xyz_err_tru = ROSImageRegistrationNode.computePoseError(cameraPose, invPoseTransform_tru)
            if (self.DEBUG == True):
                print("invPoseEstimate_tru = %s" % invPoseTransform_tru)
                print("YPR_error_tru = %s" % yaw_pitch_roll_xyz_err_tru)

            if (self.register_moving_image is not None and 
                np.min(self.register_moving_image.shape[:2]) > MIN_IMAGE_DIMENSION and
                np.min(self.register_fixed_image.shape[:2]) > MIN_IMAGE_DIMENSION):
                    
                #print('Saving image registration_fixed_image_excerpt_%s.png' % str(self.frame_id))
                #cv2.imwrite('registration_fixed_image_excerpt_'+str(self.frame_id)+'.png', self.register_fixed_image)
                #print('Saving image '+'registration_moving_image_%s.png' % str(self.frame_id))
                #cv2.imwrite('registration_moving_image_'+str(self.frame_id)+'.png', self.register_moving_image)
                
                #detectList = ["ORB","GFTT","FAST","BRISK","SURF","SIFT"]
                #descriptList = ["ORB","SURF","SIFT","BRISK","BOOST"]
                if (self.ALGORITHM == Algorithm.OPENCV_FEATURES):
                    (Hrgb_estimated, mask) = self.algorithm.registerImagePair(self.register_moving_image, image_ref=self.register_fixed_image, 
                                                                              image_ref_keypoints=None, image_ref_descriptors=None,
                                                                              algorithm_detector=self.ALGORITHM_DETECT, algorithm_extractor=self.ALGORITHM_EXTRACT)                                             
                elif (self.ALGORITHM == Algorithm.MUTUAL_INFORMATION):
                    initialTransform = np.linalg.inv(Hrgb_bbox_truth)
                    initialTransform = initialTransform / initialTransform[2, 2]
                    (Hrgb_estimated, mask) = self.algorithm.registerImagePair(self.register_moving_image, self.register_fixed_image,
                                                                              initialTransform)                                             
                elif (self.ALGORITHM == Algorithm.MUTUAL_INFORMATION_OCL):
                    (Hrgb_estimated, mask) = self.algorithm.registerImagePair(self.register_moving_image, self.register_fixed_image,
                                                                              initialTransform)
                elif (self.ALGORITHM == Algorithm.CROSS_CORRELATION):
                    initialTransform = np.linalg.inv(Hrgb_bbox_truth)
                    initialTransform = initialTransform / initialTransform[2, 2]
                    (Hrgb_estimated, mask) = self.algorithm.registerImagePair(self.register_moving_image, self.register_fixed_image,
                                                                              initialTransform)
            
                #print('Hrgb_truth  = %s' % str(Hrgb_truth))
                #print('Hbbox_truth = %s' % str(Hrgb_bbox_truth))
                #print('Hrgb_estimated = %s' % str(Hrgb_ref_truth))
                Hrgb_bbox_truth_inv = np.linalg.inv(Hrgb_bbox_truth)
                Hrgb_bbox_truth_inv = Hrgb_bbox_truth_inv / Hrgb_bbox_truth_inv[2, 2]
                if (Hrgb_estimated is not None):
                    Herror = Hrgb_estimated - Hrgb_bbox_truth_inv
                    #print('Herror = %s' % str(Herror))                            
                else:
                    return
                
                (height, width) = self.register_moving_image.shape[:2]                       
                pts_homogeneous_image_corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
                pts_transformed_homogenous = cv2.perspectiveTransform(pts_homogeneous_image_corners[None,:,:], Hrgb_bbox_truth_inv)[0]
                
                pts_transformed_homogeneous_estimated = cv2.perspectiveTransform(pts_homogeneous_image_corners[None,:,:], Hrgb_estimated)[0]
                m_per_pixel_XY = np.asarray((float(self.reference_gazebo_model_size_x) / self.image_ref.shape[1], 
                                            float(self.reference_gazebo_model_size_y) / self.image_ref.shape[0]), dtype=np.float32)
                center_of_image_XY = 0.5 * np.asarray( (self.image_ref.shape[1], self.image_ref.shape[0]), dtype=np.float32)
                #print("pts_transformed_homogeneous_estimated = %s" % pts_transformed_homogeneous_estimated)
                #print("m_per_pixel_XY = %s" % m_per_pixel_XY)
                #print("center_of_image_XY = %s" % center_of_image_XY)
                for idx in range(0, pts_transformed_homogeneous_estimated.shape[0]):
                    pts_transformed_homogeneous_estimated[idx,:] += [bbox[0][0], bbox[1][0]]
                    # compensate for world coordinate XY=(0,0) is referenced to the image center
                    pts_transformed_homogeneous_estimated[idx,:] -= center_of_image_XY
                    #pts_transformed_homogeneous_estimated[:2, idx] *= [m_per_pixel_y, m_per_pixel_x]
                    pts_transformed_homogeneous_estimated[idx,:] = np.multiply(pts_transformed_homogeneous_estimated[idx,:], m_per_pixel_XY)
                    # gazebo's Y axis is flipped
                    pts_transformed_homogeneous_estimated[idx,1] *= -1;
                # concatenate the Z coordinate for all points and set it to 0
                pts_transformed_homogeneous_estimated = np.concatenate((pts_transformed_homogeneous_estimated.T, 
                                                                        np.zeros((1,pts_transformed_homogeneous_estimated.shape[0]), dtype=np.float32)), axis=0)
                
                #print("pts_transformed_homogeneous_estimated = %s" % pts_transformed_homogeneous_estimated)                
                # solvePnP 
                (ret, rotation_vector_est, translation_vector_est) = cv2.solvePnP(objectPoints=pts_transformed_homogeneous_estimated.transpose(),
                                                                      imagePoints=image_corners, cameraMatrix=cameraK, 
                                                                      distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE) 
                #, useExtrinsicGuess=True, rvec=rvec0, tvec=T0)
                (poseRMatrix_est, Rjacobian_est) = cv2.Rodrigues(rotation_vector_est)
                poseTransform_est = np.eye(4)
                poseTransform_est[:3,:3] = poseRMatrix_est
                poseTransform_est[:3, 3] = translation_vector_est[:, 0]
                invPoseTransform_est = np.linalg.inv(poseTransform_est)
                invPoseTransform_est[:3,:3] = np.transpose(invPoseTransform_est[:3,:3])
                invPoseTransform_est[0,:3] = -invPoseTransform_est[0,:3]
                invPoseTransform_est[1,:3] = -invPoseTransform_est[1,:3]
                # compare/subtract from (ret, rotation_vector, translation_vector) to estimate error
                # log to file
                # logs -> google sheets -> plot
                #print("poseEstimate_est = %s" % poseTransform_est)
                yaw_pitch_roll_xyz_err_est = ROSImageRegistrationNode.computePoseError(cameraPose, invPoseTransform_est)
                if (self.LOGGING == True):
                    datetime_str = str(time.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))                    
                    algorithm_str = self.algorithm.toString();
                    self.dumpAsStringToLog(datetime_str)
                    self.dumpAsStringToLog(algorithm_str)
                    self.dumpNumpyDataToLog(cameraPose)
                    self.dumpNumpyDataToLog(invPoseTransform_est)
                    self.dumpNumpyDataToLog(yaw_pitch_roll_xyz_err_est)
                    self.dumpNumpyDataToLog(Herror)
                    self.dumpAsStringToLog(self.waypoint_num)
                    self.dumpAsStringToLog(str(timeit.default_timer()-start))
                    self.logfileNewline()                               

                if (self.DEBUG == True):
                    print("invPoseEstimate_est = %s" % invPoseTransform_est)
                    print("YPR_error_est = %s" % yaw_pitch_roll_xyz_err_est)

                if (self.ROS_VISUALIZE_SENSOR_VIEW_FRUSTUM == True and uv_target_image is not None):
                    if (False):
                        img = uv_target_image.copy()
                        blue = img[:,:, 0].copy()
                        cv2.fillPoly(blue, [np.int_(pts_transformed_homogenous[:,:2])], color=228, lineType=8, shift=0)
                        img[:,:, 0] = blue
                    else:
                        fixed_bw = ImageRegistration.convertImageColorSpace(self.register_fixed_image)
                        moving_bw = ImageRegistration.convertImageColorSpace(self.register_moving_image)
                        moving_bw_xformed = cv2.warpPerspective(moving_bw, Hrgb_estimated, (fixed_bw.shape[1], fixed_bw.shape[0]))
                        blue = np.zeros(fixed_bw.shape[:2], dtype=np.uint8)
                        green = fixed_bw
                        red = moving_bw_xformed
                        #print("size (r,g,b)=(%s,%s,%s)" % (red.shape[:2], green.shape[:2], blue.shape[:2]))
                        cv2.fillPoly(blue, [np.int_(pts_transformed_homogenous[:,:2])], color=128, lineType=8, shift=0)
                        img = np.dstack((blue, green, red)).astype(np.uint8)

                    window_title = 'sensor-to-reference image matches'
                    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_title, img)
                    cv2.resizeWindow(window_title, 600, 600)
                    cv2.waitKey(5)
                    #print('Saving image '+'registration_result_image_'+str(self.frame_id)+'.png')
                    #cv2.imwrite('registration_result_image_'+str(self.frame_id)+'.png', img)

                if (self.VISUALIZE_TRAJECTORY == True):
                    self.visualizeTrajectory(current_coordinate)

            if (np.min(self.register_moving_image.shape[:2]) > 3):
                register_moving_image_message = self.bridge.cv2_to_imgmsg(self.register_moving_image, encoding="passthrough")
                self.register_moving_image_pub.publish(register_moving_image_message)
                register_fixed_image_message = self.bridge.cv2_to_imgmsg(self.register_fixed_image, encoding="passthrough")
                self.register_fixed_image_pub.publish(register_fixed_image_message)
                if(self.rgb_truth_homography_pub.get_num_connections() > 0):
                    # compute a visualization of the camera view in the RGB image as a green polygon
                    ref_img = self.image_ref.copy();
                    cv2.fillPoly(ref_img, pts=np.int32([polyPts.transpose()]), color=(0, 255, 0))
                    ref_image_truth_message = self.bridge.cv2_to_imgmsg(ref_img, encoding="passthrough")
                    self.rgb_truth_homography_pub.publish(ref_image_truth_message)
            
        except CvBridgeError, e:
            print(e)        
        
    def sarImageCallback(self, sar_image_msg):
        try:
            #print("Got SAR image")
            if (self.ready == False):
                return
            # Convert your ROS Image message to OpenCV2
            sar_sensor_img = self.bridge.imgmsg_to_cv2(sar_image_msg, "bgr8")
            self.register_query_image = sar_sensor_img.copy()
            #cv2.imwrite('sar_camera_image.jpeg', sar_sensor_img)
            self.prev_SAR_image_msg = sar_image_msg
        except CvBridgeError, e:
            print(e)

    def rgbImageCallback(self, rgb_image_msg):
        try:
            #print("Got RGB image")
            if (self.ready == False):
                return
            # Convert your ROS Image message to OpenCV2
            rgb_sensor_img = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            #if (self.register_target_image is not None and np.min(self.register_target_image.shape[:2]) > 25):
            #    (homography, mask) = self.registerImagePairOpenCV(rgb_sensor_img, image_ref=self.register_target_image,
            #                                                      image_ref_keypoints=None, image_ref_descriptors=None)
                #(homography, mask) = self.registerImagePairOpenCV(rgb_sensor_img, image_ref=None,
                #                                                  image_ref_keypoints=self.keypoints_ref, 
                #                                                  image_ref_descriptors=self.descriptors_ref)
                #print("est homography = %s" % homography)
            # homography maps pixels in the sensed image to pixels in the reference image (pixels to pixels)
            # compute the scaling transformation taking homography from the reference image space in pixels to meters
            # use the (x,y) location in meters to compute the location in the Web Mercator CRS
            # transform from Web Mercator CRS to WGS84 CRS
            #if (homography is not None and self.hasCameraInfo == True):
            #    rospy.loginfo('image_registration::Decomposing homography')
            #    _, rotArr, tranArr, normArr = cv2.decomposeHomographyMat(homography, self.pinhole_camera_model.intrinsicMatrix())
            #    for solutionIdx in range(0,len(rotArr)):
            #        rospy.loginfo('image_registration::rotations[%d] %s' % (solutionIdx, str(rotArr[0])))
            #        rospy.loginfo('image_registration::translations[%d] %s' % (solutionIdx, str(tranArr[0])))
            #        rospy.loginfo('image_registration::normals[%d] %s' % (solutionIdx, str(normArr[0])))
                
            #if (self.ROS_VISUALIZE_SENSOR_VIEW_FRUSTUM == True and homography is not None):
            #    self.visualizeHomography( rgb_sensor_img, self.image_ref, homography, mask)
            self.prev_RGB_image_msg = rgb_image_msg
        except CvBridgeError, e:
            print(e)
            
    def visualizeTrajectory(self, coord):
        (height, width) = self.traj_image.shape[:2]
        coord = np.asarray(coord, dtype=np.float32)
        coord[1] *= -1
        center_of_model_XY = 0.5 * np.asarray((self.reference_gazebo_model_size_x, self.reference_gazebo_model_size_y), dtype=np.float32)
        coord += center_of_model_XY
        pixel_coord = (coord[0]/self.reference_gazebo_model_size_x*width, coord[1]/self.reference_gazebo_model_size_y*height)
        pixel_coord = tuple(np.asarray(pixel_coord,dtype=np.int32))
        if (self.DEBUG == True):
            print('Image Shape: (%s, %s)' % (str(width),str(height)))
            print("Previous Coordinate = %s\nCurrent Coordinate = %s" % (str(self.prev_coordinate), str(pixel_coord)))

        if (self.traj_image is not None and self.prev_coordinate is not None):
            window_title = 'flight path'
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.line(self.traj_image,self.prev_coordinate, pixel_coord,(0,255,0),self.TRAJECTORY_LINE_SIZE)
            if (self.waypoint_num != "-1"):
                print("JUST TO CONFIRM: %d" % int(self.waypoint_num))
                font = cv2.FONT_HERSHEY_PLAIN
                text_size, _ = cv2.getTextSize(self.waypoint_num,font, 2,1)
                cv2.rectangle(self.traj_image,pixel_coord,(pixel_coord[0]+text_size[0],pixel_coord[1]+text_size[1]),(255,0,0),-1)
                self.traj_image = cv2.putText(self.traj_image,self.waypoint_num,(pixel_coord[0],pixel_coord[1]+text_size[1]),font,2,(0,0,0),1)
                self.waypoint_num = "-1"
            cv2.imshow(window_title,self.traj_image)
            cv2.resizeWindow(window_title, 600,600)
            cv2.waitKey(5)
        self.prev_coordinate = pixel_coord

    def visualizeHomography(self, sensor_img, image_ref, homography, mask):
        (width, height) = sensor_img.shape[:2]
        pts_homogeneous_image_corners = np.array([[0, width, width, 0],
                                                 [0, 0, height, height],
                                                 [1, 1, 1, 1]])
                        
        pts_transformed_homogenous = np.matmul(homography, pts_homogeneous_image_corners)
        rospy.loginfo('image_registration::homogeneous_pts = %s' % str(pts_transformed_homogenous))

        # #####################################
        # visualization of the matches
        if (self.ROS_VISUALIZE_SENSOR_VIEW_FRUSTUM == True and image_ref is not None):
            window_title = 'sensor-to-reference image matches'
            cv2.fillConvexPoly(img, pn, color, lineType=8, shift=0)
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.imshow(window_title, image_ref)
            cv2.resizeWindow(window_title, 600, 600)
            cv2.waitKey(5)        
        
    def rgbCameraInfoCallback(self, camera_info_msg):
        rospy.loginfo("image_registration::Received a camera info image!")
        try:
            self.pinhole_camera_model.fromCameraInfo(camera_info_msg)        
            self.hasCameraInfo = True
            # Process the camera info message
            self.camera_info_sub.unregister()            
        except rospy.ROSInterruptException, e:
            print(e)

if __name__ == '__main__':
    rospy.init_node('image_registration_node')
    try:
        ROSImageRegistrationNode()
        # Spin until ctrl + c
        while not rospy.is_shutdown():
            # wait for new messages and call the callback when they arrive
            rospy.spin()        
    except:
        rospy.ROSInterruptException
    pass
