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
cv2.imshow('image', image)       
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

#Finally draw the line
cv2.line(image,(gray.shape[1]-1,righty),(0,lefty),255,2)
cv2.imshow('img_line',image)



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
print(roi)



img2gray = cv2.cvtColor(skeleton,cv2.COLOR_BGR2GRAY)
cv2.imshow('img2gray', img2gray)

mask_inv = cv2.bitwise_not(img2gray)

cv2.imshow('mask_inv', mask_inv)


img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow('img1_bg', img1_bg)


    
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
