import numpy as np
import cv2
import time 

import math



img = 255 * np.ones(shape=[480, 640, 3], dtype=np.uint8)

img_copy_1 = img.copy()


img = cv2.circle(img, (100, 100), 40, (150,200,50), 3)

img = cv2.circle(img, (100, 100), 2, (110,220,50), 2)

img_copy = img.copy()


# find circle (cv2.HoughCircles)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=10,param2=5,minRadius=35,maxRadius=45)
circle_points = None
print("circles", circles)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img_copy, (x, y), r, (255, 0, 0), 1)
        hsv = cv2.cvtColor(img_copy,cv2.COLOR_BGR2HSV)

        lower_blue = np.array([120,255,255])
        upper_blue = np.array([120,255,255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        #get all non zero values
        circle_points = cv2.findNonZero(mask)

        # for i in range(y-r, y+r):
        #     for j in range(x-r, x+r):
        #         if (img_copy[i,j] == (255,0,0)):
        cv2.rectangle(img_copy, (x+3, y+3), (x-3, y-3), (0, 128, 255), -1)
    cv2.imshow("output", np.hstack([img_copy]))



img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



# get edges
high_thresh, thresh_img = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("thresh_img", thresh_img)
lowThresh = 0.5*high_thresh
edged = cv2.Canny(thresh_img, lowThresh, high_thresh) 
#cv2.imshow("edged", edged)

# ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("len(contours)", len(contours))

print("fgfghfg", contours[0][25][0][1])
# get center of contour
M = cv2.moments(contours[0])
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

radius = math.sqrt((cX - contours[0][100][0][0])**2 + (cY - contours[0][100][0][1])**2)  

print("r", radius)


cv2.circle(img, (cX, cY), 3, (0, 0, 0), -1)

print(len(contours[0]))
#cv2.drawContours(img, contours[0], -1, (255,0,0), 5)
#cv2.drawContours(img, contours[1], -1, (0,0,255), 4)
#cv2.drawContours(img, contours[2], -1, (0,255,255), 3)
#cv2.drawContours(img, contours[3], -1, (0,255,0), 2)

cv2.drawContours(img, contours, -1, (0,0,0), 1)


red_point = (contours[0][115][0][0], contours[0][115][0][1])
blue_point_1 = (contours[0][100][0][0], contours[0][100][0][1])
blue_point_2 = (contours[0][25][0][0], contours[0][25][0][1])


# draw points of view camera(blue) and laser(red)
cv2.circle(img, red_point, 2, (0, 0, 255), -1)
cv2.circle(img, blue_point_1, 2, (255, 0, 0), -1)
cv2.circle(img, blue_point_2, 2, (255, 0, 0), -1)


# get length of line between 2 blue points (camera view)
dist_cam = math.sqrt((contours[0][25][0][0] - contours[0][100][0][0])**2 + (contours[0][25][0][1] - contours[0][100][0][1])**2)  
print("dist_cam", dist_cam)
# draw line between blue points
img = cv2.line(img, (contours[0][100][0][0], contours[0][100][0][1]), (contours[0][25][0][0], contours[0][25][0][1]), (255, 0, 0), 1) 





img_copy_1 = cv2.line(img_copy_1, (contours[0][100][0][0], contours[0][100][0][1]), (contours[0][25][0][0], contours[0][25][0][1]), (255, 0, 0), 1) 




img_gray_1 = cv2.cvtColor(img_copy_1,cv2.COLOR_BGR2GRAY)
#cv2.imshow("img_gray_1", img_gray_1)


# get edges
high_thresh, thresh_img_1 = cv2.threshold(img_gray_1, 150, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("thresh_img_1", thresh_img_1)
lowThresh = 0.5*high_thresh
edged_1 = cv2.Canny(thresh_img_1, lowThresh, high_thresh) 
#cv2.imshow("edged-1", edged_1)

# ret,thresh = cv2.threshold(imgray,127,255,0)
contours_1, hierarchy = cv2.findContours(edged_1,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
print("len(contours)", len(contours_1))
print("(contours)", (contours_1))


cv2.drawContours(img_copy_1, contours_1[2], -1, (0,0,0), 2)

cv2.imshow("img_copy_1", img_copy_1)

#cv2.imshow("img_copy_1-1", img_copy_1)


M = cv2.moments(contours_1[2])


c_x_blue_line = int(M["m10"] / M["m00"])
c_y_blue_line = int(M["m01"] / M["m00"])


cv2.circle(img, (c_x_blue_line, c_y_blue_line), 3, (255, 0, 0), -1)

'''
# get porpendicular to blue line
def getPerpCoord(aX, aY, bX, bY, length):
    vX = bX-aX
    vY = bY-aY
    #print(str(vX)+" "+str(vY))
    #if(vX == 0 or vY == 0):
    #    return 0, 0, 0, 0
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    return int(cX), int(cY), int(dX), int(dY)


perp_coord = getPerpCoord(contours[0][100][0][0], contours[0][100][0][1], c_x_blue_line, c_y_blue_line, dist_cam)

# draw perpendicular to blue line
img = cv2.line(img, (perp_coord[0], perp_coord[1]), (perp_coord[2], perp_coord[3]), (255, 0, 0), 1) 

cv2.circle(img, (perp_coord[0], perp_coord[1]), 3, (255, 100, 30), -1)
cv2.circle(img, (perp_coord[2], perp_coord[3]), 3, (255, 150, 30), -1)
'''

'''
from skimage.draw import line
# being start and end two points (x1,y1), (x2,y2)
discrete_line = list(zip(*line(*(perp_coord[0], perp_coord[1]), *(perp_coord[2], perp_coord[3]))))

print(discrete_line)
'''

print(cX, cY, radius)
img = cv2.circle(img, (cX, cY), int(radius), (100,100,255), 1)




# find y(z) of blue points

for i in range(red_point[1]+1, cY+int(radius)):
    if (np.all(img_copy_1[i, blue_point_1[0]] == [255,0,0])):
        blue_point_1 = (blue_point_1[0], i)
for i in range(red_point[1]+1, cY+int(radius)):
    if (np.all(img_copy_1[i, blue_point_2[0]] == [255,0,0])):
        blue_point_2 = (blue_point_2[0], i)

print(blue_point_1)
print(blue_point_2)


img_copy_1 = cv2.line(img_copy_1, (blue_point_1[0], red_point[1]), (blue_point_2[0], red_point[1]), (255, 0, 0), 1) 



def findAngle(px1, py1, px2, py2, cx1, cy1):
    dist1 = math.sqrt( (px1-cx1)*(px1-cx1) + (py1-cy1)*(py1-cy1) )
    dist2 = math.sqrt(  (px2-cx1)*(px2-cx1) + (py2-cy1)*(py2-cy1) )

    Ax = Ay = 0
    Bx = By = 0
    Cx = Cy = 0

    #find closest point to C
    #print("dist = %lf %lf\n", dist1, dist2);

    Cx = cx1
    Cy = cy1
    if (dist1 < dist2):
        Bx = px1
        By = py1
        Ax = px2
        Ay = py2
    else:
        Bx = px2
        By = py2
        Ax = px1
        Ay = py1

    Q1 = Cx - Ax
    Q2 = Cy - Ay
    P1 = Bx - Ax
    P2 = By - Ay

    A = math.acos( (P1*Q1 + P2*Q2) / ( math.sqrt(P1*P1+P2*P2) * math.sqrt(Q1*Q1+Q2*Q2) ) )
    A = A*180/math.pi
    return A


angle_br = findAngle(blue_point_1[0], blue_point_1[1], red_point[0], red_point[1], cX, cY)

print(angle_br)


angle_rb = findAngle(blue_point_2[0], blue_point_2[1], red_point[0], red_point[1], cX, cY)

print(angle_rb)



cv2.imshow("img_copy_1-2", img)



cv2.waitKey(0)
cv2.destroyAllWindows()

