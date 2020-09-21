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

image = cv2.imread("frame917.jpg", -1)
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
cv2.imshow('image',image)
cv2.namedWindow('image')
cv2.setMouseCallback('image', on_click)

#image = cv2.resize(image, (1200, 900))
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#lower red
lower_red = np.array([0,50,50])
upper_red = np.array([40,255,255])

#upper red
lower_red2 = np.array([-1,65,215])
upper_red2 = np.array([32,187,295])
mask = cv2.inRange(hsv, lower_red2, upper_red2)

res = cv2.bitwise_and(image,image, mask= mask)
kernel = np.ones((5,5),np.uint8)
res = cv2.dilate(res,kernel,iterations = 2)




res_blur = cv2.GaussianBlur(res, (11,1), 1)

cv2.imshow('res_blur', res_blur)
      
cv2.imshow('res1', res)
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
cv2.line(image,(gray.shape[1]-1,righty),(0,lefty),(255,255,0),1)
print("lol", (gray.shape[1]-1,righty),(0,lefty))
cv2.imshow('img_line',image)


#image = cv2.resize(image, (1200, 900))
hsv_blue = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#lower red [120 255 255] [110 245 215] [130 265 295]
# 255_255_0 [ 90 255 255] [ 80 245 215] [100 265 295]
lower_blue = np.array([ 90, 255, 255])
upper_blue = np.array([ 90, 255, 255])

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
print("skel_coords", skel_coords)




redImg = np.zeros(image.shape, image.dtype)
redImg[:,:] = (0, 255, 255)
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

cv2.imshow('added_image', added_image)

# <1 Fill res_blue2 with 2 diff colors on the sides of skelet line

color2_fill_skeleton = skeleton.copy()


coord=cv2.findNonZero(img2gray)
print(len(coord))
print(skel_coords[1][0])
print(coord[0][0][0])

print(res_blue.shape)


counter = 0
for i in range(skel_coords[0][0], skel_coords[1][0]):
    for j in range(0, coord[counter][0][0]):
        #print(i, j, coord[i][0][0])
        color2_fill_skeleton[i,j] = [0, 50, 255]
    counter += 1

counter = 0
for i in range(skel_coords[0][0], skel_coords[1][0]):
    for j in range(coord[counter][0][0], res_blue.shape[1]):
        #print(i, j, coord[i][0][0])
        color2_fill_skeleton[i,j] = [0, 10, 255]
    counter += 1

cv2.imshow('color2_fill_skeleton', color2_fill_skeleton)
# 1>

# <2 Fill res_blue2 with 2 diff colors on the sides of straight line
color2_fill_straight = res_blue.copy()



fit_straight = np.zeros(shape=[mask_blue.shape[0], mask_blue.shape[1], 1], dtype=np.uint8)
# draw staright line
cv2.line(fit_straight,(skel_coords[0][1],skel_coords[0][0]),(skel_coords[1][1],skel_coords[1][0]),255,1)
coord=cv2.findNonZero(fit_straight)
# for i in range(coord[0][0][1], skel_coords[0][0]): 
#     print("coord[i][0][0]", coord[i][0][0])
#     color2_fill_straight[i, coord[i][0][0]] = [0,255,0]

print("len(coord)", len(coord))
print("skel_coords", skel_coords)
#print("coord", coord)

counter = 0
for i in range(skel_coords[0][0], skel_coords[1][0]):
    #print(coord[counter][0][0])
    for j in range(0, coord[counter][0][0]):
        #print(i, j, coord[i][0][0])
        color2_fill_straight[i,j] = [100, 150, 100]
    counter += 1

counter = 0
for i in range(skel_coords[0][0], skel_coords[1][0]):
    for j in range(coord[counter][0][0], res_blue.shape[1]):
        #print(i, j, coord[i][0][0])
        color2_fill_straight[i,j] = [100, 150, 150]
    counter += 1

cv2.imshow('color2_fill_straight', color2_fill_straight)
# 2>



# color2_fill_straight[color2_fill_straight[:, :, 1:].all(axis=-1)] = 0
# color2_fill_skeleton[color2_fill_skeleton[:, :, 1:].all(axis=-1)] = 0

# overlay_straight_skelet_filled = cv2.addWeighted(color2_fill_straight, 1, color2_fill_skeleton, 1, 0)

# cv2.imshow('overlay_straight_skelet_filled', overlay_straight_skelet_filled)




overlay_straight_skelet_filled = cv2.addWeighted(color2_fill_straight,0.5,color2_fill_skeleton,0.5,0)
cv2.imshow('overlay_straight_skelet_filled', overlay_straight_skelet_filled)

# [ 10 192 202] [  0 182 162] [ 20 202 242]
# [  7 183 178] [ -3 173 138] [ 17 193 218]


# define the list of boundaries
boundaries = [
	([ 7, 183, 178], [ 10, 192, 202])
]


# boundaries = [
# 	([ 10, 192, 202], [ 10, 192, 202]),
#     ([ 7, 183, 178], [ 7, 183, 178])
# ]

# hsv_blue = cv2.cvtColor(overlay_straight_skelet_filled,cv2.COLOR_BGR2HSV)

# mask = cv2.inRange(hsv_blue, np.array([ 10, 192, 202]), np.array([ 10, 192, 202]))
# output = cv2.bitwise_and(hsv_blue, overlay_straight_skelet_filled, mask = mask)
# # show the images
# cv2.imshow("test", np.hstack([overlay_straight_skelet_filled, output]))
# cv2.imshow("test1", mask)
# cv2.imshow("test2", output)
# cv2.imshow("test3", overlay_straight_skelet_filled)

counter = 0
prev_output = None
overlay_masks = None
output_mask_rect = None
for (lower, upper) in boundaries:
    hsv_blue = cv2.cvtColor(overlay_straight_skelet_filled,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_blue, np.array(lower), np.array(upper))
    output = cv2.bitwise_and(overlay_straight_skelet_filled, hsv_blue, mask = mask)
    if (np.all(prev_output != None)):
        overlay_masks = cv2.addWeighted(output,0.5,prev_output,0.5,0)
    prev_output = output
	

    # remove unnecessary contours
    ## draw staright line
    cv2.line(output,(skel_coords[0][1],skel_coords[0][0]),(skel_coords[1][1],skel_coords[1][0]),(0,0,0),2)
    cv2.line(mask,(skel_coords[0][1],skel_coords[0][0]),(skel_coords[1][1],skel_coords[1][0]),(0,0,0),2)
    ## draw skelet line
    output[indices[0], indices[1]] = [0,0,0]

    # for i in range(len(mask)):
    #     for j in range(len(mask[0])):
    #         print(indices[0][i], indices[1][j])
    #         mask[indices[0][i]][indices[1][j]] = [0,0,0]

    # print(len(mask))

    # for i in range(236, 579):
    #     print(mask[i])

    # mask[indices[0]][indices[1]] = 0

    for i in range(len(indices[0])):
        mask[indices[0][i]][indices[1][i]] = 0
        mask[indices[0][i]+1][indices[1][i]] = 0
        mask[indices[0][i]-1][indices[1][i]] = 0

    cv2.imshow("test1"+str(counter), mask)
    cv2.imshow("test2"+str(counter), output)


    ret,thresh = cv2.threshold(mask, 40, 255, 0)
    if (int(cv2.__version__[0]) > 3):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        #cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),1)
        output_mask_rect = output

    # show the images
    cv2.imshow("Result", output)

    counter += 1
counter = 0

# cv2.imshow("overlay_masks", overlay_masks)






img_bwa = cv2.bitwise_and(color2_fill_straight,color2_fill_skeleton)
img_bwo = cv2.bitwise_or(color2_fill_straight,color2_fill_skeleton)
img_bwx = cv2.bitwise_xor(color2_fill_straight,color2_fill_skeleton)
cv2.imshow("Bitwise AND of Image 1 and 2", img_bwa)
cv2.imshow("Bitwise OR of Image 1 and 2", img_bwo)
cv2.imshow("Bitwise XOR of Image 1 and 2", img_bwx)




added_image = cv2.addWeighted(mask_inv,0.9,mask_blue_inv,0.1,9)





cv2.line(mask_inv,(gray.shape[1]-1,righty),(0,lefty),0,1)
cv2.imshow('mask_with_line', mask_inv)

cv2.imshow('added_image', added_image)
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow('img1_bg', img1_bg)


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



res = cv2.addWeighted(output_mask_rect,0.5,img,0.5,0)
cv2.imshow('res', res)

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
