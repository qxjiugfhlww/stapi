"""
 This sample shows how to use callback function to acquire image data from camera.
 The following points will be demonstrated in this sample code:
 - Initialize StApi
 - Connect to camera
 - Register and use callback function with StApi
 - Acquire image data via callback function
"""

#import stapipy as st
import cv2
import numpy as np
import os
import threading

import time
from datetime import date
import datetime

import argparse
from skimage.morphology import skeletonize, binary_closing, binary_opening
from skimage import data
from skimage.util import invert
import matplotlib.pyplot as plt
DISPLAY_RESIZE_FACTOR = 1

class CMyCallback:
    def __init__(self):
        self._image = None
        self._lock = threading.Lock()

    @property
    def image(self):
        duplicate = None
        self._lock.acquire()
        if self._image is not None:
            duplicate = self._image.copy()
        self._lock.release()
        return duplicate

    def datastream_callback(self, handle=None, context=None):
        """
        Callback to handle events from DataStream.

        :param handle: handle that trigger the callback.
        :param context: user data passed on during callback registration.
        """
        # Image scale when displaying using OpenCV.
        
        st_datastream = handle.module
        if st_datastream:
            with st_datastream.retrieve_buffer() as st_buffer:
                # Check if the acquired data contains image data.
                if st_buffer.info.is_image_present:
                    # Create an image object.
                    st_image = st_buffer.get_image()

                    # Create an image object and convert
                    st_image = st_converter_pixelformat.convert(
                        st_converter_reverse.convert(st_image))

                    # Get raw image data.
                    data = st_image.get_image_data()
                    nparr = np.frombuffer(data, np.uint8)

                    # Process image for displaying the BGR8 image.
                    nparr = nparr.reshape(st_image.height, st_image.width, 3)
                    
                    # Resize image.and display.
                    nparr = cv2.resize(nparr, None, fx=DISPLAY_RESIZE_FACTOR, fy=DISPLAY_RESIZE_FACTOR)
                    
                    self._lock.acquire()
                    self._image = nparr
                    self._lock.release()

                    '''
                    cv2.imshow('image', nparr)
                    key_input = cv2.waitKey(1)
                    if key_input != -1:
                        # Stop the image acquisition of the camera side
                        st_device.acquisition_stop()

                        # Stop the image acquisition of the host side
                        st_datastream.stop_acquisition()
                        os.system(exit)
                    '''
                    
                    #print("Image data exist.")
                else:
                    ...
                    # If the acquired data contains no image data.
                    #print("Image data does not exist.")

if __name__ == "__main__":

    # Get the callback function:
    # my_callback = CMyCallback()
    # cb_func = my_callback.datastream_callback

    try:
        # # Initialize StApi before using.
        # st.initialize()

        # # Create a system object for device scan and connection.
        # st_system = st.create_system()

        # # Connect to first detected device.
        # st_device = st_system.create_first_device()

        # # Display DisplayName of the device.
        # print('Device=', st_device.info.display_name)
        
        # # Create a converter object for vertical reverse.
        # st_converter_reverse = st.create_converter(st.EStConverterType.Reverse)
        # st_converter_reverse.reverse_y = True

        # # Create a converter object for converting pixel format to BGR8.
        # st_converter_pixelformat = \
        #     st.create_converter(st.EStConverterType.PixelFormat)
        # st_converter_pixelformat.destination_pixel_format = \
        #     st.EStPixelFormatNamingConvention.BGR8

        # # Create a datastream object for handling image stream data.
        # st_datastream = st_device.create_datastream()

        # # Register callback for datastream
        # callback = st_datastream.register_callback(cb_func)

        # # Start the image acquisition of the host (local machine) side.
        # st_datastream.start_acquisition()

        # # Start the image acquisition of the camera side.
        # st_device.acquisition_start()

        # # Press enter to terminate.
        # #input("Press enter to terminate")

        # #print("To terminate, focus on the OpenCV window and press any key.")
        # count = 0
        # curr_time = 0
        # prev_time = datetime.datetime.now()


        # # construct the argument parse and parse the arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-i", "--image", help = "path to the image file")
        # ap.add_argument("-r", "--radius", type = int, help = "radius of Gaussian blur; must be odd")
        # args = vars(ap.parse_args())
        # # load the image and convert it to grayscale
       

    

        image = cv2.imread("laser-5.jpg", -1)
        print(image)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        image[50:60,50:60] = [10,150,10]

        def on_click(event, x, y, p1, p2):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x,y,p1,p2)
                #cv2.circle(lastImage, (x, y), 3, (255, 0, 0), -1)
        f, (ax0, ax1, ax2) = plt.subplots(3, 1)

        # call the video source


        w = 1920  # Obtain video dimension x
        h = 1200  # Obtain video dimension y

        # search area
        x1 = 500
        x2 = 800
        y1 = 750
        y2 = 800

        start_frame = 1000
        stop_frame = 1250

        def rotate_image(image, angle):
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result
        '''
        def find_skeleton3(full_img, img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converted to gray
            #gray = cv2.GaussianBlur(gray,(15, 15), 15)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)    
            inv = invert(thresh)
            #cv2.imshow('thresh', inv)
            # perform skeletonization
            inv[inv == 255] = 1
            #skeleton = skeletonize(binary_closing(inv))
            #skeleton2 = skeletonize(binary_opening(binary_closing(inv)))
            skeleton3 = binary_closing(skeletonize(binary_opening(binary_closing(inv))))
            skel_img = skeleton3.copy().astype('uint8')*255

            right_lines_num = 0
            lines = cv2.HoughLinesP(skel_img, cv2.HOUGH_PROBABILISTIC, theta = 1*np.pi/180, threshold = 1, minLineLength = 30, maxLineGap = 0)

            curv_f=0
            min_y = y2 - y1
            max_y = 0
            if lines is not None:
                for x in range(0, len(lines)):
                    for x1_,y1_,x2_,y2_ in lines[x]:
                        #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
                        pts = np.array([[x1_, y1_ ], [x2_ , y2_]], np.int32)
                        arctan = slope(x1_, x2_, y1_, y2_)
                        if y1_>max_y:
                            max_y = y1_
                        if y2_>max_y:
                            max_y = y2_
                        if y1_<min_y:
                            min_y = y1_
                        if y2_<min_y:
                            min_y = y2_
                        if (abs(arctan) <= 315):
                            curv_f+=abs(arctan)
                        cv2.polylines(img, [pts], True, (0,255,0))
                curv_f += max_y - min_y
                
                for i in range(len(lines)):
                    lx1 = lines[i][0][0]
                    ly1 = lines[i][0][1]

                    lx2 = lines[i][0][2]
                    ly2 = lines[i][0][3]
                    if (round(lx2-lx1) != 0):
                        arctan = slope(lx1, lx2, ly1, ly2)
                        if (round(arctan >= round(-315)) and round(arctan <= round(315))):
                            cv2.line(img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                            right_lines_num += 1
                

            #cv2.imshow('all', img)
            ax0.imshow(full_img)

            ax1.imshow(img)
            ax1.set_title(str(curv_f))
            ax2.imshow(skeleton3)
            plt.draw()
            plt.pause(100)
            return skeleton3
        '''
        cv2.imshow("image", image)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_click)


        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        #lower red
        lower_red = np.array([10,50,50])
        upper_red = np.array([30,255,255])

        print("hsv:", hsv)

        #upper red
        lower_red2 = np.array([170,50,50])
        upper_red2 = np.array([180,255,255])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(image,image, mask= mask)
        
        '''
        # Not optimized
        x_arr = []
        for i in range(len(res)):
            x_arr.append([])
        y_arr = []

        for i in range(len(res)):
            #print("i:", i)
            for j in range(len(res[0])):
                #print("j:", j)
                if np.any(res[i,j] != 0):
                    
                    x_arr[i].append([i, j])
            sum_y = 0
            sum_x = 0
            if (len(x_arr[i]) == 0):
                continue
            for j in range(len(x_arr[i])):
                sum_x += x_arr[i][j][0]
                sum_y += x_arr[i][j][1]
            y_arr.append([int(sum_x/len(x_arr[i])), int(sum_y/len(x_arr[i]))])
            #print(i)
            
        for i in range(len(y_arr)):
            print(y_arr[i][0], y_arr[i][1])
            image[y_arr[i][0], y_arr[i][1]] = [10,240,10]

        for i in range(len(y_arr)):
            image[i, 50] = [0,255,0]

        '''

        #print(x_arr)


        # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        # res2 = cv2.bitwise_and(image,image, mask= mask2)

        # img3 = res+res2
        # img4 = cv2.add(res,res2)
        # img5 = cv2.addWeighted(res,0.5,res2,0.5,0)


        kernel = np.ones((15,15),np.float32)/225
        smoothed = cv2.filter2D(res,-1,kernel)
        # smoothed2 = cv2.filter2D(img3,-1,kernel)




        #print(cv2.imread(res,cv2.IMREAD_COLOR))
        #print(res[0][0])
        cv2.imshow('Original',image)

        cv2.imshow('Averaging',smoothed)
        cv2.imshow('mask',mask)
        

        #res = rotate_image(res, 20)
        cv2.imshow('res',res)

        '''
        SKELETON
        '''
        #frame = image
        #frameRect = frame
        #find_skeleton3(frame,frameRect)  
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # while True:
        #     output_image = my_callback.image
        #     if output_image is not None:
        #         image = output_image

        #         hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        #         #lower red
        #         lower_red = np.array([0,50,50])
        #         upper_red = np.array([10,255,255])


        #         #upper red
        #         lower_red2 = np.array([170,50,50])
        #         upper_red2 = np.array([180,255,255])

        #         mask = cv2.inRange(hsv, lower_red, upper_red)
        #         res = cv2.bitwise_and(image,image, mask= mask)
        #         #print(res)

        #         # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        #         # res2 = cv2.bitwise_and(image,image, mask= mask2)

        #         # img3 = res+res2
        #         # img4 = cv2.add(res,res2)
        #         # img5 = cv2.addWeighted(res,0.5,res2,0.5,0)


        #         kernel = np.ones((15,15),np.float32)/225
        #         smoothed = cv2.filter2D(res,-1,kernel)
        #         # smoothed2 = cv2.filter2D(img3,-1,kernel)




        #         #print(cv2.imread(res,cv2.IMREAD_COLOR))
        #         #print(res[0][0])
        #         cv2.imshow('Original',image)
        #         cv2.imshow('Averaging',smoothed)
        #         cv2.imshow('mask',mask)
                
        #         cv2.imshow('res',res)
        #         # cv2.imshow('mask2',mask2)
        #         # cv2.imshow('res2',res2)
        #         # cv2.imshow('res3',img3)
        #         # cv2.imshow('res4',img4)
        #         # cv2.imshow('res5',img5)
        #         # cv2.imshow('smooth2',smoothed2)


        #         # orig = image.copy()
        #         # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #         # delta_time =  datetime.datetime.now() - prev_time
        #         # prev_time = datetime.datetime.now()
        #         #cv2.imshow('image!', output_image)

        #         # # perform a naive attempt to find the (x, y) coordinates of
        #         # # the area of the image with the largest intensity value
        #         # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        #         # cv2.circle(image, maxLoc, 5, (255, 0, 0), 2)
        #         # # display the results of the naive attempt
        #         # cv2.imshow("Naive", image)

        #         # img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #         # # lower mask (0-10)
        #         # lower_red = np.array([0,50,50])
        #         # upper_red = np.array([10,255,255])
        #         # mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        #         # # upper mask (170-180)
        #         # lower_red = np.array([170,50,50])
        #         # upper_red = np.array([180,255,255])
        #         # mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        #         # # join my masks
        #         # mask = mask0+mask1

        #         # # set my output img to zero everywhere except my mask
        #         # output_img = image.copy()
        #         # output_img[np.where(mask==0)] = 0

        #         # # or your HSV image, which I *believe* is what you want
        #         # output_hsv = img_hsv.copy()
        #         # output_hsv[np.where(mask==0)] = 0

        #         #print("fps:", 1000000/delta_time.microseconds)
        #     else:
        #         print("image is None")
        #     key_input = cv2.waitKey(1)
        #     if key_input != -1:
        #         break

        # Stop the image acquisition of the camera side
        # st_device.acquisition_stop()

        # # Stop the image acquisition of the host side
        # st_datastream.stop_acquisition()

    except Exception as exception:
        print(exception)
