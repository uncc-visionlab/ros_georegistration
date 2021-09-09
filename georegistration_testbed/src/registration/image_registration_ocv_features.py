#!/usr/bin/env python

import cv2
from image_registration import *
import numpy as np

class ImageRegistrationWithOpenCVFeatures(ImageRegistration):
    """An object that runs image registration algorithms on image data."""

    def __init__(self):
        ImageRegistration.__init__(self)
        self.detector_str = None
        self.extractor_str = None

    def toString(self):
        return "alg = OCV_FEATURE detector = %s extractor = %s" % (self.detector_str, self.extractor_str)

    def initializeFeatureRegistration(self, refImageFeatureCount=10000, movingImageFeatureCount=10000, detector='ORB', extractor='ORB'):
        self.detector_str = detector;
        if(detector == 'ORB'):
            self.kpDetector_ref = cv2.ORB_create(nfeatures=refImageFeatureCount, fastThreshold=0)
            self.kpDetector_moving = cv2.ORB_create(nfeatures=movingImageFeatureCount, fastThreshold=0)
        elif(detector == 'GFTT'):
            self.kpDetector_ref = cv2.GFTTDetector_create(maxCorners=refImageFeatureCount,minDistance=10)
            self.kpDetector_moving = cv2.GFTTDetector_create(maxCorners=movingImageFeatureCount,minDistance=10)
        elif(detector == 'FAST'):
            self.kpDetector_ref = cv2.FastFeatureDetector_create()
            self.kpDetector_moving = cv2.FastFeatureDetector_create()
        elif(detector == 'BRISK'):
            self.kpDetector_ref = cv2.BRISK_create()
            self.kpDetector_moving = cv2.BRISK_create()
        elif(detector == 'SURF'):
            self.kpDetector_ref = cv2.xfeatures2d.SURF_create(nOctaves=5)
            self.kpDetector_moving = cv2.xfeatures2d.SURF_create(nOctaves=5)
        elif(detector == 'SIFT'):
            self.kpDetector_ref = cv2.xfeatures2d.SIFT_create(nfeatures=refImageFeatureCount)
            self.kpDetector_moving = cv2.xfeatures2d.SIFT_create(nfeatures=movingImageFeatureCount)
        #self.descExtractor_ref = cv2.xfeatures2d.SURF_create()
        #self.descExtractor_ref = cv2.xfeatures2d.SIFT_create()
        #self.descExtractor_ref = cv2.ORB_create(nfeatures=refImageFeatureCount)
        else:
            #Use ORB as default in case of spelling error
            self.detector_str = 'ORB'
            self.kpDetector_ref = cv2.ORB_create(nfeatures=refImageFeatureCount)
            self.kpDetector_moving = cv2.ORB_create(nfeatures=movingImageFeatureCount)

        self.extractor_str = extractor;
        if(extractor == 'ORB'):
            self.descExtractor_ref = cv2.ORB_create(nfeatures=refImageFeatureCount, fastThreshold=0)
            self.descExtractor_moving = cv2.ORB_create(nfeatures=movingImageFeatureCount, fastThreshold=0)
        elif(extractor == 'SURF'):
            self.descExtractor_ref = cv2.xfeatures2d.SURF_create(nOctaves=5)
            self.descExtractor_moving = cv2.xfeatures2d.SURF_create(nOctaves=5)
        elif(extractor == 'SIFT'):
            self.descExtractor_ref = cv2.xfeatures2d.SIFT_create(nfeatures=refImageFeatureCount)
            self.descExtractor_moving = cv2.xfeatures2d.SIFT_create(nfeatures=movingImageFeatureCount)
        elif(extractor == 'BRISK'):
            self.descExtractor_ref = cv2.BRISK_create()
            self.descExtractor_moving = cv2.BRISK_create()
        elif(extractor == 'BOOST'):
            self.descExtractor_ref = cv2.xfeatures2d.BoostDesc_create()
            self.descExtractor_moving = cv2.xfeatures2d.BoostDesc_create()
        
        #self.kpDetector_moving = cv2.ORB_create(nfeatures=movingImageFeatureCount)
        #self.kpDetector_moving = cv2.GFTTDetector_create()
        #self.kpDetector_moving = cv2.FastFeatureDetector_create()
        #self.descExtractor_moving = cv2.xfeatures2d.SURF_create()
        #self.descExtractor_moving = cv2.xfeatures2d.SIFT_create()
        #self.descExtractor_moving = cv2.ORB_create(nfeatures=movingImageFeatureCount)
        #self.descExtractor_moving = self.kpDetector_moving
        else:
            #Use ORB as default in case of spelling error
            self.extractor_str = 'ORB'
            self.descExtractor_ref = cv2.ORB_create(nfeatures=refImageFeatureCount)
            self.descExtractor_moving = cv2.ORB_create(nfeatures=movingImageFeatureCount)
        #DEBUG
        print("Detector: {}\nExtractor: {}".format(type(self.kpDetector_ref),type(self.descExtractor_moving)))
        self.initialized = True
                                
    def multiScaleImageRegistration(self, image_moving, image_ref,algorithm_detector='ORB', algorithm_extractor='ORB'):
        tileSize = np.array([100, 100])
        featureDensity = 0.01
        MINFEATURES = 500
        MAXFEATURES = 5e6
        
        homography = None
        mask = None
        numMovingFeatures = min(MINFEATURES, image_moving.size * featureDensity)
        numMovingFeatures = int(max(MAXFEATURES, numMovingFeatures))
        numReferenceFeatures = min(MINFEATURES, image_moving.size * featureDensity)
        numReferenceFeatures = int(max(MAXFEATURES, numReferenceFeatures))
        if (self.initialized == False):
            self.initializeFeatureRegistration(numReferenceFeatures, numMovingFeatures, algorithm_detector, algorithm_extractor)
        if (image_moving is None):
            print('Null moving image.')
            return (homography, mask)
        # convert moving image to gray then detect features and compute descriptors
        image_moving_bw = ImageRegistration.convertImageColorSpace(image_moving)
        keypoints_moving = self.kpDetector_moving.detect(image_moving_bw)
        (keypoints_moving, descriptors_moving) = self.descExtractor_moving.compute(image_moving_bw, keypoints_moving)
        
        image_ref_bw = ImageRegistration.convertImageColorSpace(image_ref)
        (refImg_height, refImg_width) = image_ref_bw.shape[:2]
        imgDims = np.array([refImg_height, refImg_width])
        numTiles = np.int_(np.ceil(imgDims / tileSize))
        tileCoords = np.zeros((4), dtype=int)
        print('numTiles', numTiles)
        for tileYIdx in range(0, numTiles[0]):
            for tileXIdx in range(0, numTiles[1]):
                
                if (tileYIdx == numTiles[0]-1 and tileXIdx == numTiles[1]-1):
                    tileCoords = [tileYIdx * tileSize[0], refImg_height, 
                        tileXIdx * tileSize[1], refImg_width]
                elif (tileYIdx == numTiles[0]-1):
                    tileCoords = [tileYIdx * tileSize[0], refImg_height, 
                        tileXIdx * tileSize[1], (tileXIdx + 1) * tileSize[1]]               
                elif (tileXIdx == numTiles[1]-1):
                    tileCoords = [tileYIdx * tileSize[0], (tileYIdx + 1) * tileSize[0], 
                        tileXIdx * tileSize[1], refImg_width]
                else:
                    tileCoords = [tileYIdx * tileSize[0], (tileYIdx + 1) * tileSize[0], 
                        tileXIdx * tileSize[1], (tileXIdx + 1) * tileSize[1]]
                tile = image_ref_bw[tileCoords[0]:tileCoords[1], tileCoords[2]:tileCoords[3]]
                keypoints_ref = self.kpDetector_ref.detect(tile)
                print('tileCoords', tileCoords)
                print('keypoints_ref', keypoints_ref)
                (keypoints_ref, descriptors_ref) = self.descExtractor_ref.compute(image_ref_bw, keypoints_ref)
        
        return (homography, mask)
                                
    def registerImagePair(self, image_moving, image_ref=None,
                          image_ref_keypoints=None, image_ref_descriptors=None,algorithm_detector='ORB',algorithm_extractor='ORB'):
                                    
        homography = None
        mask = None
        if (self.initialized == False):
            self.initializeFeatureRegistration(detector=algorithm_detector, extractor=algorithm_extractor)
        if (image_moving is None):
            print('Null moving image.')
            return (homography, mask)
        # convert moving image to gray then detect features and compute descriptors
        image_moving_bw = ImageRegistration.convertImageColorSpace(image_moving)
        if (self.kpDetector_moving != self.descExtractor_moving):
            keypoints_moving = self.kpDetector_moving.detect(image_moving_bw)
            descriptors_moving = self.descExtractor_moving.compute(image_moving_bw, keypoints_moving)[1]
        else:
            (keypoints_moving, descriptors_moving) = self.descExtractor_moving.detectAndCompute(image_moving_bw, None)
            
        if (image_ref_keypoints is None or image_ref_descriptors is None):
            if (image_ref is None):
                print('Error: Cannot register image pair. No reference image or reference image (feature, descriptor) pairs provided.')
                return (homography, mask)
            # convert new image to gray then detect features and compute descriptors
            image_ref_bw = ImageRegistration.convertImageColorSpace(image_ref)
            #image_ref_bw = cv2.GaussianBlur(image_ref_bw, (15, 15), 0)
            if (self.kpDetector_ref != self.descExtractor_ref):
                keypoints_ref = self.kpDetector_ref.detect(image_ref_bw)
                descriptors_ref = self.descExtractor_ref.compute(image_ref_bw, keypoints_ref)[1]
            else:
                (keypoints_ref, descriptors_ref) = self.descExtractor_ref.detectAndCompute(image_ref_bw, None)
        else:
            # keypoints and descriptors of the reference image are provided
            # the image reference may be stored in the image registration object
            keypoints_ref = image_ref_keypoints
            descriptors_ref = image_ref_descriptors
            image_ref = self.image_ref

        if (len(keypoints_moving) < self.MIN_FEATURES_FOR_ALIGNMENT):
            print('Not enough features in moving image. Feature count = %s' % len(keypoints_moving))
            return (homography, mask)
        if (len(keypoints_ref) < self.MIN_FEATURES_FOR_ALIGNMENT):
            print('Not enough features in reference image. Feature count = %s' % len(keypoints_ref))
            return (homography, mask)

        # Match features between the two images. 
        # We create a Brute Force matcher with  
        # Hamming distance as measurement mode. 
        print ('keypoints (reference, moving) (%d,%d)' % (len(keypoints_ref), len(keypoints_moving)))
        print ('descriptors (reference, moving) (%d,%d)' % (len(descriptors_ref), len(descriptors_moving)))
        #matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
        if (algorithm_extractor == "SIFT" or algorithm_extractor == "SURF"):
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match the two sets of descriptors. 
        matches = matcher.match(descriptors_ref, descriptors_moving) 
        # Sort matches on the basis of their Hamming distance. 
        matches.sort(key=lambda x: x.distance) 

        # visualize the matches
        print ('#matches: %d' % len(matches))
        dist = [m.distance for m in matches]
        if (len(matches) < 3):
            print('Not enough features matches to estimate homography. Match count = %s' % len(matches))
            return (homography, mask)
        print('distance min : %.3f' % min(dist))
        print('distance mean: %.3f' % (sum(dist) / len(dist)))
        print('distance max : %.3f' % max(dist))

        # threshold: half the mean
        thres_dist = (sum(dist) / len(dist)) * 0.5

        # keep only the reasonable matches
        sel_matches = [m for m in matches if m.distance < thres_dist]

        # Take the top 90 % matches forward. 
        sel_matches = matches[:int(len(matches) * 0.90)] 
        no_of_matches = len(sel_matches) 

        print('#selected matches: %d' % len(sel_matches))

        # Define empty matrices of shape no_of_matches * 2. 
        p1 = np.zeros((no_of_matches, 2)) 
        p2 = np.zeros((no_of_matches, 2)) 

        for i in range(len(sel_matches)): 
            p1[i,:] = keypoints_ref[sel_matches[i].queryIdx].pt 
            p2[i,:] = keypoints_moving[sel_matches[i].trainIdx].pt 

        # Find the homography matrix. 
        homography, mask = cv2.findHomography(p2, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0) 
        #print("homography = %s" % str(homography))
        if (np.linalg.matrix_rank(homography) < 3):
            print("homography matrix is singular.")
            return (homography, mask)

        if (self.SAVE_WARPEDIMAGE == True):
            # Use this matrix to transform the 
            # colored image wrt the reference image. 
            (height, width, channels) = image_moving.shape[:3]
            transformed_img = cv2.warpPerspective(image_moving, 
                                                  homography, (width, height))
            # Save the output. 
            cv2.imwrite('output.jpg', transformed_img) 

        # #####################################
        # visualization of the matches
        if (self.OPENCV_VISUALIZE_MATCHES == True and image_ref is not None):
            image_moving_disp = image_moving.copy()
            image_ref_disp = image_ref.copy()
            h1, w1 = image_moving_disp.shape[:2]
            h2, w2 = image_ref_disp.shape[:2] 
            scale_moving = 1.0
            scale_ref = 1.0            

            if (h1 < h2):
                scale_moving = float(h2) / h1
                image_moving_sc = cv2.resize(image_moving_disp, (int(scale_moving * h1), int(scale_moving * w1)))
                image_moving_disp = image_moving_sc
                h1, w1 = image_moving_disp.shape[:2] 
            elif (h2 < h1):
                scale_ref = float(h1) / h2
                image_ref_sc = cv2.resize(image_ref_disp, (int(scale_ref * h2), int(scale_ref * w2)))
                image_ref_disp = image_ref_sc
                h2, w2 = image_ref_disp.shape[:2] 
            
            GAP_SIZE = 10
            GAP_COLOR = np.array([255, 255, 255], dtype=np.uint8)
            #print("heights (%d,%d)" % (h1, h2))
            view = np.zeros((max(h1, h2), w1 + GAP_SIZE + w2, 3), np.uint8)
            view[:h1,:w1,:] = image_moving_disp
            view[:h1, w1:(w1 + GAP_SIZE),:] = GAP_COLOR
            view[:h2, (w1 + GAP_SIZE):,:] = image_ref_disp
            SHOW_NUM_KEYPOINTS = 20.0
            for mIdx in range(0, len(sel_matches), int(np.ceil(len(sel_matches) / SHOW_NUM_KEYPOINTS))):
                m = sel_matches[mIdx]
                # draw the keypoints
                # print m.queryIdx, m.trainIdx, m.distance
                color = tuple([np.random.randint(128, 255) for _ in xrange(3)])
                k1 = keypoints_moving
                k2 = keypoints_ref
                keypoint1_int = (int(k1[m.trainIdx].pt[0] * scale_moving), int(k1[m.trainIdx].pt[1] * scale_moving));
                keypoint2_int = (int(k2[m.queryIdx].pt[0] * scale_ref + w1 + GAP_SIZE), int(k2[m.queryIdx].pt[1] * scale_ref))
                # Draw Selected points
                #print('%s %s %s' % (str(k1[m.queryIdx].pt), str(k1[m.queryIdx].octave), str(color)))
                cv2.circle(view, keypoint1_int, 15, color, thickness=3)
                cv2.circle(view, keypoint2_int, 15, color, thickness=3)

                cv2.line(view, keypoint1_int, keypoint2_int, color, thickness=3)
            cv2.namedWindow('view matched features', cv2.WINDOW_NORMAL)
            cv2.imshow('view matched features', view)
            cv2.resizeWindow('view matched features', 600, 600)
            cv2.waitKey(5)        
            
        if (self.OPENCV_SHOW_HOMOGRAPHY_BOUNDS_ON_REFERENCE == True and image_ref is not None):
            if (not hasattr(self, 'scaled_image_ref')):
                (self.scaled_image_ref, self.scalef) = ImageRegistration.resizeImage(image_ref, self.OPENCV_SHOW_HOMOGRAPHY_MAXDIM)
                
            (h1, w1) = image_moving.shape[:2]
            print('Scale factor : ', self.scalef)
            h1 = h1
            w1 = w1
            image_bounds = np.array(([0, 0, 1], [w1, 0, 1], [w1, h1, 1], [0, h1, 1]), dtype=np.float32)
            image_bounds = image_bounds.transpose()
            (dimpt, numpoints) = image_bounds.shape[:2]
            scaled_homography = homography.copy()
            scaled_homography[:, 2] = scaled_homography[:, 2] * self.scalef
            scaled_homography = np.linalg.inv(scaled_homography)
            #scaled_homography[2,:] = scaled_homography[2,:] / self.scalef
            #print('Scaled homography :', scaled_homography)
            xformed_image_bounds = np.matmul(scaled_homography, image_bounds)
            for pointIdx in range(0, numpoints):
                xformed_image_bounds[:, pointIdx] = xformed_image_bounds[:, pointIdx] / xformed_image_bounds[2, pointIdx]
                #print('point : ', xformed_image_bounds[:,pointIdx])
            #print('Original points : ', image_bounds)
            #print('New points : ', xformed_image_bounds)
            #cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 125, 125)); 
            #alpha = 0.5;
            #image2show = np.zeros(self.scaled_image_ref.shape, np.uint8) # empty image
            #image2show = cv2.cvtColor(self.scaled_image_ref, cv2.COLOR_BGR2BGRA)
            #print('pts', np.int_(xformed_image_bounds[:2,:]))
            #cv2.fillPoly(image2show, [np.int_(xformed_image_bounds[:2,:]).transpose()], color = [228,0,0], lineType = 8, shift = 0)
            #image2show = cv2.addWeighted(image2show, alpha, self.scaled_image_ref, 1.0 - alpha, 0)
            #cv2.namedWindow('view homography bounds',cv2.WINDOW_NORMAL)
            #cv2.imshow('view homography bounds', image2show)
            #cv2.resizeWindow('view homography bounds', 1600,1600)
            #cv2.waitKey() 

        return (homography, mask)

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
    
    image_registration = ImageRegistrationWithOpenCVFeatures()
    image_registration.SAVE_WARPEDIMAGE = True
    image_registration.OPENCV_SHOW_HOMOGRAPHY_BOUNDS_ON_REFERENCE = True
    #image_registration.OPENCV_SHOW_HOMOGRAPHY_MAXDIM = 512
    image_registration.registerImagePair(image_moving, image_ref=image_reference)
