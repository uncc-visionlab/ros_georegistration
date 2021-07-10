#!/usr/bin/env python

import cv2
from image_registration_mi import *
import numpy as np
import pyopencl as cl

class ImageRegistrationWithMutualInformationOpenCL(ImageRegistrationWithMutualInformation):
    """An object that runs image registration algorithms on image data."""
    def __init__(self):
        ImageRegistrationWithMutualInformation.__init__(self)
        self.numIntensityBins = 64
        #self.iteration_number = 0
        #Declare maximum iteration number for mutual information optimization
        self.maximum_iteration_number = 50
        self.EMBARRASSINGLY_PARALLEL = True
        if (self.EMBARRASSINGLY_PARALLEL == False):   
            # setup OpenCL
            platforms = cl.get_platforms()
            platform = platforms[0]
            devices = platform.get_devices(cl.device_type.GPU)
            device = devices[0]  # take first GPU
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context, device)  # create command queue for selected GPU and context
            #declare OpenCL
            self.src = open('imageTransformation_MI.cl').read()
            # load and compile OpenCL program
            self.program = cl.Program(self.context, self.src).build()
            
    @staticmethod
    def applyTransformation(self, imgIn, shape_imgOut, initialTransform):
        """
        Input = Image
        Output = Transformed Image
        """
        #self.iteration_number = self.iteration_number + 1
        #print('Image_Registration::Applying Transformation for iteration = ' + str(self.iteration_number))
        
        theta_rad = initialTransform[0]
        #shape_imgOut = imgRef.T.shape
        imgOut_width = shape_imgOut[0]
        imgOut_height = shape_imgOut[1]
        #print('imgOut_width = ' + str(imgOut_width))
        #print('imgOut_height = ' + str(imgOut_height))
        #os.listdir()
        if (self.EMBARRASSINGLY_PARALLEL == True):
            print('Image_Registration::Parallel Processing ON')
            # setup OpenCL
            platforms = cl.get_platforms()
            platform = platforms[0]
            devices = platform.get_devices(cl.device_type.GPU)
            device = devices[0]  # take first GPU
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context, device)  # create command queue for selected GPU and context
            #declare OpenCL
            #self.src = open('ros_ws/src/rosgeoregistration/georegistration_testbed/src/registration/imageTransformation_MI.cl').read()
            #Kernel Function
            #https://wiki.tiker.net/PyOpenCL/Examples/MedianFilter/
            self.src = '''
            __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
            __kernel void img_translate(__write_only image2d_t transformedImage, __read_only image2d_t inputImage, const int W, const int H,
                            const float sinTheta, const float cosTheta, const float shearX, const float shearY,
                             const float translationX, const float translationY, const int Wout, const int Hout){
            //Work-item gets its index within index space
            const int ix = get_global_id(0);
            const int iy = get_global_id(1);
            int2 coords = {ix, iy};
            //Calculate location of data to move into (ix,iy)
            //Output decomposition as mentioned
            //https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983

            int xpos = (int)(cosTheta*ix - sinTheta*shearX*iy + translationX);
            int ypos = (int)(shearY*sinTheta*ix + cosTheta*iy + translationY);
            int2 transCoords = {xpos, ypos};

            if((xpos >=0) && (xpos < Wout) &&
                (ypos >=0) && (ypos < Hout)){
                //Read (ix,iy) src data
                float4 inputPixel = read_imagef(inputImage, sampler, coords);
                    write_imagef(transformedImage, transCoords, inputPixel);
                }
            }
            '''
            # load and compile OpenCL program
            self.program = cl.Program(self.context, self.src).build()
            
            cosTheta = cos(theta_rad)
            sinTheta = sin(theta_rad)
            shearx = initialTransform[1]
            sheary = initialTransform[2]
            tx = initialTransform[3]
            ty = initialTransform[4]

            
            # get shape of Input image
            #imgIn = self.imageMoving
            shape_imgIn = imgIn.T.shape
            #print('shape image Input = ' + str(shape_imgIn))
            imgIn_width = shape_imgIn[0]
            imgIn_height = shape_imgIn[1]

            # create input and output image buffer
            imgInBuf = cl.Image(self.context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape_imgIn)
            imgOutBuf = cl.Image(self.context, cl.mem_flags.WRITE_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape_imgOut)
            #create transformation matrix (2 x3) and buffer
            kernel = cl.Kernel(self.program, 'img_translate')
            kernel.set_arg(0, imgOutBuf)
            kernel.set_arg(1, imgInBuf)
            kernel.set_arg(2, np.uint32(imgIn_width))
            kernel.set_arg(3, np.uint32(imgIn_height))
            kernel.set_arg(4, np.float32(sinTheta))
            kernel.set_arg(5, np.float32(cosTheta))
            kernel.set_arg(6, np.float32(shearx))
            kernel.set_arg(7, np.float32(sheary))
            kernel.set_arg(8, np.float32(tx))
            kernel.set_arg(9, np.float32(ty))
            kernel.set_arg(10, np.uint32(imgOut_width))
            kernel.set_arg(11, np.uint32(imgOut_height))
            #self.program.img_translate(self.queue, shape_imgIn, None, imgOutBuf, imgInBuf, np.uint32(imgIn_width), np.uint32(imgIn_height), np.uint32(imgOut_width),
            #						   translationBuf, np.uint32(imgOut_height))
            #copy image to device
            cl.enqueue_copy(self.queue, imgInBuf, imgIn, origin=(0, 0), region=shape_imgIn, is_blocking=False)
            #cl.enqueue_copy(self.queue, translationBuf, translation_matrix, is_blocking=False)
            cl.enqueue_nd_range_kernel(self.queue, kernel, shape_imgIn, None)

            #imgOut = np.zeros((imgOut_width, imgOut_height))
            imgOut = np.zeros([imgOut_width, imgOut_height], dtype=float)
            cl.enqueue_copy(self.queue, imgOut, imgOutBuf, origin=(0, 0), region=shape_imgOut, is_blocking=True)

            #imgOut = np.empty_like(imgRef)
            #imgOut = np.empty_like(self.imageFixed)
        
            """
            https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
            T = [[cos(theta) -s_x*sin(theta) t_x],
            [s_y*sin(theta) cos(theta) t_y],
            [0 0 1]]
            """
            #    translation_matrix = np.float32([[cosTheta, -shearx*sinTheta, tx], [sheary*sinTheta, cosTheta, ty]])
        else:
            translation_matrix = np.float32([[cos(theta_rad), -initialTransform[1] * sin(theta_rad), initialTransform[3]],
                                            [initialTransform[2] * sin(theta_rad), cos(theta_rad), initialTransform[4]]])
            #imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags = cv2.INTER_CUBIC)
            imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags=cv2.INTER_NEAREST)
            #imgOut = cv2.warpAffine(imgIn, translation_matrix, (imgOut_width, imgOut_height), flags = cv2.INTER_LINEAR)
        return imgOut
    
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
    
    image_registration = ImageRegistration()
    image_registration.SAVE_WARPEDIMAGE = True
    image_registration.OPENCV_SHOW_HOMOGRAPHY_BOUNDS_ON_REFERENCE = True
    #image_registration.OPENCV_SHOW_HOMOGRAPHY_MAXDIM = 512
    image_registration.registerImagePair(image_moving, image_ref=image_reference)
