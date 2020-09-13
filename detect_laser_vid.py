import cv2
import numpy as np
import os
import threading

from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

import time
from datetime import date
import datetime

# Invert the horse image

def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    out = np.zeros_like(skel)
    out[np.where(filtered==11)] = 1
    return out


cap = cv2.VideoCapture('vid.mp4')

start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    #cv2.imshow('frame',frame)
    
    image = frame
    counter+=1
    if (time.time() - start_time) > x :
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()

    #image = cv2.resize(image, (1200, 900))
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    #lower red
    lower_red = np.array([0,50,50])
    upper_red = np.array([40,255,255])

    #upper red
    lower_red2 = np.array([155,-8,199])
    upper_red2 = np.array([184,89,292])
    mask = cv2.inRange(hsv, lower_red2, upper_red2)

    res = cv2.bitwise_and(image,image, mask= mask)

    kernel = np.ones((5,5),np.uint8)
    res = cv2.dilate(res,kernel,iterations = 2)


    res_blur = cv2.GaussianBlur(res, (11,1), 1)

    #cv2.imshow('res_blur', res_blur)
    #cv2.imshow('image', image)       
    #cv2.imshow('res', res)
    # perform skeletonization
    skeleton = skeletonize(res)

    indices = np.where(skeleton==255)
    skeleton[indices[0], indices[1], :] = [0, 255, 0]
    #cv2.imshow('skeleton', skeleton)

    # skel_endp = skeleton_endpoints(skeleton)



    # redImg = np.zeros(image.shape, image.dtype)
    # redImg[:,:] = (0, 0, 255)
    # redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
    # cv2.addWeighted(redMask, 1, image, 1, 0, image)




    # out_img = cv2.add(image,skeleton)
    # #img1[0:rows, 0:cols ] = out_img

    # cv2.imshow('image2', out_img)  
    # #cv2.imshow('image3', img1) 



    # rows,cols,channels = image.shape
    # print(rows,cols,channels)
    # roi = image[0:rows, 0:cols ]
    # print(roi)



    # img2gray = cv2.cvtColor(skeleton,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img2gray', img2gray)

    # mask_inv = cv2.bitwise_not(img2gray)

    # cv2.imshow('mask_inv', mask_inv)


    # img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # cv2.imshow('img1_bg', img1_bg)


        
    #out_img = cv2.add(img1_bg,img2_fg)
    #img1[0:rows, 0:cols ] = out_img





    img=cv2.addWeighted(image,0.5,skeleton,0.5,0)
    cv2.imshow('img', img)


    # # display results
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
    #                          sharex=True, sharey=True)

    # ax = axes.ravel()

    # ax[0].imshow(image, cmap=plt.cm.gray)
    # ax[0].axis('off')
    # ax[0].set_title('original', fontsize=20)

    # ax[1].imshow(skeleton, cmap=plt.cm.gray)
    # ax[1].axis('off')
    # ax[1].set_title('skeleton', fontsize=20)

    # fig.tight_layout()
    # plt.show()



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


