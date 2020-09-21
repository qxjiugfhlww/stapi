import cv2 
import numpy as np
    
import matplotlib.pyplot as plt

# Reading an image in default mode 
points = np.array([[[100,100], [70, 200], [150, 270], [230, 200], [200, 100]]], np.int32)

#img = np.zeros((500,500,1), np.uint8)

img = np.zeros([512,512,3],dtype=np.uint8)
img.fill(255)

img = cv2.fillPoly(img, points, (0,0,0), lineType=cv2.LINE_AA)

#img = cv2.polylines(img, [points], True, (0,0,0),1)

'''
for i in range(len(points)):
    #cv2.ellipse(img, points[i], (2,2), 0, 0, 360, (0,0,0), -1) 
    if (i != len(points)-1):
        cv2.line(img, points[i], points[i+1], (0, 0, 0), thickness=1)
    else:
        cv2.line(img, points[i], points[0], (0, 0, 0), thickness=1)

#img = cv2.ellipse(img, (300, 300), (50, 50), 30, 0, 360, (0, 0, 0), -1) 
'''





from skimage.draw import ellipse
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon





# hand = np.array(points)

# # subdivide polygon using 2nd degree B-Splines
# new_hand = hand.copy()
# for _ in range(5):
#     new_hand = subdivide_polygon(new_hand, degree=2, preserve_ends=True)

# cv2.imshow("new_hand", new_hand)

# # approximate subdivided polygon with Douglas-Peucker algorithm
# appr_hand = approximate_polygon(new_hand, tolerance=0.02)


# cv2.imshow("appr_hand", appr_hand)

# print("Number of coordinates:", len(hand), len(new_hand), len(appr_hand))

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))

# ax1.plot(hand[:, 0], hand[:, 1])
# ax1.plot(new_hand[:, 0], new_hand[:, 1])
# ax1.plot(appr_hand[:, 0], appr_hand[:, 1])


# # create two ellipses in image
# img = np.zeros((800, 800), 'int32')
# rr, cc = ellipse(250, 250, 180, 230, img.shape)
# img[rr, cc] = 1
# rr, cc = ellipse(600, 600, 150, 90, img.shape)
# img[rr, cc] = 1

# plt.gray()
# ax2.imshow(img)

# # approximate / simplify coordinates of the two ellipses
# for contour in find_contours(img, 0):
#     coords = approximate_polygon(contour, tolerance=2.5)
#     ax2.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
#     coords2 = approximate_polygon(contour, tolerance=39.5)
#     ax2.plot(coords2[:, 1], coords2[:, 0], '-g', linewidth=2)
#     print("Number of coordinates:", len(contour), len(coords), len(coords2))

# ax2.axis((0, 800, 0, 800))

# plt.show()


gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

ret,thresh = cv2.threshold(gray,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
print(cnt)
M = cv2.moments(cnt)
print(M)


cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])


area = cv2.contourArea(cnt)

perimeter = cv2.arcLength(cnt,True)

epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True) 


hull = cv2.convexHull(cnt)

k = cv2.isContourConvex(cnt)

x,y,w,h = cv2.boundingRect(cnt)
#img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
#img = cv2.drawContours(img,[box],0,(0,0,255),2)


(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,0),1)

print(cnt)

ellipse = cv2.fitEllipse(cnt)
img = cv2.ellipse(img,ellipse,(0,0,255),1)

# Displaying the image 
cv2.imshow('img', img)  

cv2.waitKey(0)
cv2.destroyAllWindows()