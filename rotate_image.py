import numpy as np
import cv2
import time

import argparse
import imutils



img = cv2.imread("circle.png", 1)
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray1, (3, 3), 0)
edged = cv2.Canny(gray, 20, 100)

cv2.imshow("gray1", gray1)
cv2.imshow("gray", gray)
cv2.imshow("edged", edged)


# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

if len(cnts) > 0:
	# grab the largest contour, then draw a mask for the pill
	c = max(cnts, key=cv2.contourArea)
	mask = np.zeros(gray.shape, dtype="uint8")
	cv2.drawContours(mask, [c], -1, 255, -1)
	# compute its bounding box of pill, then extract the ROI,
	# and apply the mask
	(x, y, w, h) = cv2.boundingRect(c)
	imageROI = img[y:y + h, x:x + w]
	maskROI = mask[y:y + h, x:x + w]
	imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)

cv2.imshow("mask", mask)
cv2.imshow("imageROI", imageROI)


# loop over the rotation angles
for angle in np.arange(0, 10000, 15):
    rotated = imutils.rotate(imageROI, angle)
    cv2.imshow("Rotated (Problematic)", rotated)
    cv2.waitKey(0)
# loop over the rotation angles again, this time ensure the
# entire pill is still within the ROI after rotation
for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate_bound(imageROI, angle)
    cv2.imshow("Rotated (Correct)", rotated)
    cv2.waitKey(0)




def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result



cv2.imshow("img", img)

count = 0
center = (int(img.shape[1]/2), int(img.shape[0]/2))
for d in range(360):
    #img = rotate_image(img, count+2)
    #cv2.imshow("img", img)
    M = cv2.getRotationMatrix2D(center, count+2, 1.0)
    img = cv2.warpAffine(img, M, center)
    cv2.imshow("img", img)

    key = cv2.waitKey(100)


cv2.waitKey(0)
cv2.destroyAllWindows()