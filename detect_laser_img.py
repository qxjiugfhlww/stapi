import cv2
import numpy as np
import os
import threading

from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

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

image = cv2.imread("curve.jpg", -1)
'''
# Detecting curve line (not working yet)

inputImageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(inputImageGray,150,200,apertureSize = 3)
minLineLength = 50
maxLineGap = 3
lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
        pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
        cv2.polylines(image, [pts], True, (0,255,0))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image,"Tracks Detected", (500, 250), font, 0.5, 255)
cv2.imshow("Trolley_Problem_Result", image)
cv2.imshow('edge', edges)
'''



def on_click(event, x, y, p1, p2):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y,p1,p2)
cv2.namedWindow('image')
cv2.setMouseCallback('image', on_click)

#image = cv2.resize(image, (1200, 900))
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#lower red
lower_red = np.array([0,50,50])
upper_red = np.array([40,255,255])

#upper red
lower_red2 = np.array([-1,129,215])
upper_red2 = np.array([32,187,295])
mask = cv2.inRange(hsv, lower_red2, upper_red2)

res = cv2.bitwise_and(image,image, mask= mask)
kernel = np.ones((5,5),np.uint8)
res = cv2.dilate(res,kernel,iterations = 2)




res_blur = cv2.GaussianBlur(res, (11,1), 1)

cv2.imshow('res_blur', res_blur)
      
cv2.imshow('res', res)
# perform skeletonization
skeleton = skeletonize(res)


# Load image, convert to grayscale, threshold and find contours
gray = cv2.cvtColor(skeleton,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours,hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

# then apply fitline() function
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

# Now find two extreme points on the line to draw line
lefty = int((-x*vy/vx) + y)
righty = int(((gray.shape[1]-x)*vy/vx)+y)

print("lefty", lefty)
print("righty", righty)

#Finally draw the line
cv2.line(image,(gray.shape[1]-1,righty),(0,lefty),255,1)
print("lol", (gray.shape[1]-1,righty),(0,lefty))
cv2.imshow('img_line',image)


#image = cv2.resize(image, (1200, 900))
hsv_blue = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#lower red [120 255 255] [110 245 215] [130 265 295]
lower_blue = np.array([110, 245, 215])
upper_blue = np.array([130, 265, 295])

mask_blue = cv2.inRange(hsv_blue, lower_blue, upper_blue)

#cv2.fillPoly(mask_blue, [np.array([ [gray.shape[1], 0], [gray.shape[1]-1,righty],[0,lefty], [0, gray.shape[0]], [0,0]])], (255, 255, 255))






res_blue = cv2.bitwise_and(image,image, mask= mask_blue)
kernel = np.ones((5,5),np.uint8)
# print(res_blue[760,237])
# for i in range(gray.shape[0]-1):
#     for j in range(gray.shape[1]-2):
#         if (np.all(res_blue[i,j+1] == [255,0,0])):
#             continue
#         else:
#             res_blue[i,j] = (255,255,255)
cv2.imshow('res_blue',res_blue)

cv2.imshow('mask_blue',mask_blue)

mask_blue_inv = cv2.bitwise_not(mask_blue)

cv2.imshow('mask_blue_inv', mask_blue_inv)


### Finding endpoints of skeleton
# Find row and column locations that are non-zero
cv2.imshow('skeleton', skeleton)
(rows,cols, chan) = np.nonzero(skeleton)
# Initialize empty list of co-ordinates
skel_coords = []

# For each non-zero pixel...
for (r,c) in zip(rows,cols):

    # Extract an 8-connected neighbourhood
    (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
    

    # Cast to int to index into image
    col_neigh = col_neigh.astype('int')
    row_neigh = row_neigh.astype('int')

    # Convert into a single 1D array and check for non-zero locations
    pix_neighbourhood = skeleton[row_neigh,col_neigh].ravel() != 0

    # If the number of non-zero locations equals 2, add this to 
    # our list of co-ordinates
    if np.sum(pix_neighbourhood) == 2:
        skel_coords.append((r,c))
print("".join(["(" + str(r) + "," + str(c) + ")\n" for (r,c) in skel_coords]))

for (r,c) in skel_coords:
    image = cv2.circle(image, (c,r), 1, (0, 255, 0), 5)


indices = np.where(skeleton==255)
skeleton[indices[0], indices[1], :] = [255, 255, 0]
cv2.imshow('skeleton', skeleton)

skel_endp = skeleton_endpoints(skeleton)



redImg = np.zeros(image.shape, image.dtype)
redImg[:,:] = (0, 0, 255)
redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
cv2.addWeighted(redMask, 1, image, 1, 0, image)




out_img = cv2.add(image,skeleton)
#img1[0:rows, 0:cols ] = out_img

cv2.imshow('image2', out_img)  
#cv2.imshow('image3', img1) 



rows,cols,channels = image.shape
print(rows,cols,channels)
roi = image[0:rows, 0:cols ]



img2gray = cv2.cvtColor(skeleton,cv2.COLOR_BGR2GRAY)
cv2.imshow('img2gray', img2gray)

mask_inv = cv2.bitwise_not(img2gray)

cv2.imshow('mask_inv', mask_inv)

added_image = cv2.addWeighted(mask_inv,0.9,mask_blue_inv,0.1,9)





cv2.line(mask_inv,(gray.shape[1]-1,righty),(0,lefty),0,1)
cv2.imshow('mask_with_line', mask_inv)

cv2.imshow('added_image', added_image)
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow('img1_bg', img1_bg)


#convert img to grey
#set a thresh
thresh = 100
#get threshold image
ret,thresh_img = cv2.threshold(added_image, thresh, 255, cv2.THRESH_BINARY)
#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(image.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
#save image
cv2.imshow('img_contours',img_contours) 


#gray = cv2.cvtColor(mask5, cv2.COLOR_BGR2GRAY)
# frame = cv2.imread("mask4.jpg")
# static_back = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# while(True):

#     diff_frame = cv2.absdiff(static_back, gray)

#     thresh_frame = cv2.threshold(diff_frame, 10, 255, cv2.THRESH_BINARY)[1])

#     thresh_frame = cv2.dilate(thresh_frame, None, iterations = 1)

#     (cnts, h) = cv2.findContours(thresh_frame.copy(), cv2.RETR_CCOMP, 
#     cv2.CHAIN_APPROX_NONE)
#     for contour in cnts:
#         if cv2.contourArea(contour) > 10000:
#             cv2.drawContours(frame, contour, -1, (0, 255, 0), 5)    
#     result = imutils.resize(frame, width=320)
#     cv2.imshow("Frame", result)
#     cv2.imwrite("Frame.jpg", result)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break


    
#out_img = cv2.add(img1_bg,img2_fg)
#img1[0:rows, 0:cols ] = out_img





img=cv2.addWeighted(image,0.5,skeleton,0.5,0)
cv2.imshow('img', img)
cv2.waitKey(0)

cv2.destroyAllWindows()



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
