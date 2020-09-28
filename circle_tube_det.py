import numpy as np
import cv2
import time 

import math



img = 255 * np.ones(shape=[700, 800, 3], dtype=np.uint8)

img_copy_1 = img.copy()


img_copy_2 = img.copy()

img_copy_3 = img.copy()

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


encoder = -0.8

angle_bc = None


imageAngle4 = 40

if (encoder == 0):
    angle_bc = imageAngle4


x1 = math.cos(90)*radius
y1 = math.sin(90)*radius
   
x2 = math.cos(90+imageAngle4)
y2 = math.cos(90+imageAngle4)

print(x2, y2)


import random
import decimal



n = 10
k = 9




N = 9



r_arr = [[float(decimal.Decimal(random.randrange(2800, 3300))/100) for i in range(N)] for j in range(N) ]

# find min of min and max of max

min_r_arr = []
max_r_arr = []
for j in range(N):
    min_r_arr.append([(i, x) for i, x in enumerate(r_arr[j]) if x == min(r_arr[j])])
    max_r_arr.append([(i, x) for i, x in enumerate(r_arr[j]) if x == max(r_arr[j])])

tmp_r = []
tmp_i = []

for i, x in enumerate(min_r_arr):
    tmp_r.append(x[0][1])
    tmp_i.append(x[0][0])

    print("x", i, x[0][0], x[0][1])
    for j, y in enumerate(x):
        print(min(x))
        print("y", j, y)


print("tmp_r", tmp_r)
print("tmp_i", tmp_i)


min_min = [(i,x) for i, x in enumerate(tmp_r) if x == min(tmp_r)]
min_min = [i, min_min[0][0], min_min[0][1]]

print("min_min:", i, min_min)

tmp_r = []
tmp_i = []

for i, x in enumerate(max_r_arr):
    tmp_r.append(x[0][1])
    tmp_i.append(x[0][0])

    print("x", i, x[0][0], x[0][1])
    for j, y in enumerate(x):
        print(max(x))
        print("y", j, y)


print("tmp_r", tmp_r)
print("tmp_i", tmp_i)


max_max = [(i,x) for i, x in enumerate(tmp_r) if x == max(tmp_r)]
max_max = [i, max_max[0][0], max_max[0][1]]
print("max_max:", max_max)





#r_arr = [float(decimal.Decimal(random.randrange(2800, 3300))/100) for i in range(n)]

#r_arr = [31.65, 32.98, 31.96, 31.45, 30.34, 31.38, 28.16, 32.8, 28.71, 28.32]

start_x = 35
draw_center_xy = [[start_x,50] for i in range(N) ]
center_xy = [[start_x+70*i,50] for i in range(10) ]

#center_xy = [50,50]

print("center_xy", center_xy)


# Step #6


draw_points = []
points = []

points_all = []
draw_points_all = []

draw_x_offsets = [70*i for i in range(N) ]

for j in range(N):
    cv2.circle(img_copy_2, (center_xy[j][0], center_xy[j][1]), 2, (255, 0, 0), -1)
    points = []
    for i in range(N):
        theta = i*(360/N)
        theta *= np.pi/180.0
        points.append([int(center_xy[j][0]+np.cos(theta)*r_arr[j][i]),int(center_xy[j][1]-np.sin(theta)*r_arr[j][i])])
        draw_points.append([int(draw_center_xy[j][0]+np.cos(theta)*r_arr[j][i]),int(draw_center_xy[j][1]-np.sin(theta)*r_arr[j][i])])
        cv2.line(img_copy_2, (draw_center_xy[j][0], draw_center_xy[j][1]),(draw_points[len(draw_points)-1][0], draw_points[len(points)-1][1]), 255, 1)
        #(row,col) = np.nonzero(np.logical_and(img_copy_2, ))
        #cv2.line(img_copy_2, (center_xy[j][0], center_xy[j][1]), (col[0],row[0]), 0, 1)
    points_all.append(points)
    draw_points_all.append(draw_points)
    cv2.drawContours(img_copy_3, [np.array(points)], contourIdx=-1, color=(0,0,0),thickness=-1)

cv2.circle(img_copy_2, (50, 50), 2, (255, 0, 0), -1)


draw_points = np.array([draw_points], np.int32)

#print("points", points)






imgray = cv2.cvtColor(img_copy_3, cv2.COLOR_BGR2LAB)[...,0]

ret, thresh = cv2.threshold(imgray, 20, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

print("len(contours)", len(contours))
#print("contours", contours)


maxArea = 0
best = None


ma_ar = []

xy_centers = []
MA_arr = []
ma_arr = []
coef_arr = []
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    (x, y), (MA, ma), angle = cv2.fitEllipse(contours[i])
    cv2.ellipse(img_copy_3, ((300, 300), (MA, ma), angle), (0,255,0),1)
    xy_centers.append([x,y])
    MA_arr.append(MA)
    ma_arr.append(ma)

    ellipse = cv2.fitEllipse(contours[i])
    ma_ar.append([MA, ma])

    coef_arr.append(min(ma_ar[len(ma_ar)-1])/max(ma_ar[len(ma_ar)-1]))

    # Using cv2.putText() method 
    if (coef_arr[i] >= 0.95):
        img_copy_3 = cv2.putText(img_copy_3, str(round(coef_arr[i], 2)), (center_xy[i][0], center_xy[i][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA) 
    else:
        img_copy_3 = cv2.putText(img_copy_3, str(round(coef_arr[i], 2)), (center_xy[i][0], center_xy[i][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA)
    cv2.fitEllipse(contours[i])

    if (max_max[1] == i):
        print("max")
        img_copy_3 = cv2.putText(img_copy_3, "max r"+str(max_max[1])+": "+str(max_max[2]), (center_xy[i][0], center_xy[i][1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA) 
    if (min_min[1] == i):
        print("min")
        img_copy_3 = cv2.putText(img_copy_3, "min r"+str(min_min[1])+": "+str(min_min[2]), (center_xy[i][0], center_xy[i][1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA) 
    cv2.ellipse(img_copy_3, ellipse, (0,255,0),1)
    #if area > maxArea :
    #    maxArea = area
    #    best = contour
    #ellipse = cv2.fitEllipse(best)


MA_max = [(i,x) for i, x in enumerate(MA_arr) if x == max(MA_arr)]
ma_min = [(i,x) for i, x in enumerate(ma_arr) if x == min(ma_arr)]

print("MA_max", MA_max)
print("ma_min", ma_min)

for i in range(len(contours)):
    if (MA_max[0][0] == i):
        print("max")
        img_copy_3 = cv2.putText(img_copy_3, "max d"+str(MA_max[0][0])+": "+str(round(MA_max[0][1],2)), (center_xy[i][0], center_xy[i][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA) 
    if (ma_min[0][0] == i):
        print("min")
        img_copy_3 = cv2.putText(img_copy_3, "min d"+str(ma_min[0][0])+": "+str(round(ma_min[0][1],2)), (center_xy[i][0], center_xy[i][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA) 



img_copy_3 = cv2.putText(img_copy_3, "avg_coef_ellips: "+str(round(np.mean(coef_arr),2)), (center_xy[0][0], center_xy[0][1] + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA) 

img_copy_3 = cv2.putText(img_copy_3, "r_min/r_max: "+str(round(min_min[2]/max_max[2],2)), (center_xy[0][0], center_xy[0][1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA) 


img_copy_3 = cv2.putText(img_copy_3, "d_min/d_max: "+str(round(ma_min[0][1]/MA_max[0][1],2)), (center_xy[0][0], center_xy[0][1] + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA) 




xy_centers = sorted(xy_centers, key=lambda x: x[0])

xy_centers_tmp = [[xy_centers[i][0]-70*i,xy_centers[i][1]] for i in range(len(xy_centers))]

print(xy_centers_tmp)

xy_centers = [[int(xy_centers[i][0]-70*i), int(xy_centers[i][1])] for i in range(len(xy_centers))]



for i in range(len(xy_centers)):
    print(xy_centers[i])
    cv2.circle(img_copy_3, (int(xy_centers[i][0])+10, int(xy_centers[i][1])+150), 1, (0,0,255), -1)


all_dist = []
max_dist = []

for i in range(len(xy_centers)):
    print(i)
    all_dist.append([])
    for j in range(len(xy_centers)):
        all_dist[i].append(np.sqrt((xy_centers[i][0]-xy_centers[j][0])**2+(xy_centers[i][1]-xy_centers[j][1])**2))
        print(" " + str(np.sqrt((xy_centers[i][0]-xy_centers[j][0])**2+(xy_centers[i][1]-xy_centers[j][1])**2)))
    print(all_dist[i])
    max_dist.append( [(k,x) for k, x in enumerate(all_dist[i]) if x == max(all_dist[i])] )

print(all_dist)
print(max_dist)





dist_tmp_i =  0
for i in range(len(max_dist)):
    img_copy_3 = cv2.putText(img_copy_3, str(i), (80, 180+dist_tmp_i), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA) 
    for j in range(len(max_dist[i])):
        img_copy_3 = cv2.putText(img_copy_3, str(max_dist[i][j][0]) +" "+str(round(max_dist[i][j][1], 2)), (90, 180+dist_tmp_i + 10*j), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA) 
    dist_tmp_i += len(max_dist[i])*10









# 0.95





#cv2.polylines(img_copy_3, [points], True, (0,0,0), thickness=1)



import matplotlib.pyplot as plt
import numpy as np




rand_x_start = random.uniform(-0.1,0.05)

print("rand_x_start:", rand_x_start)

x = [i for i in np.arange(rand_x_start,0.15,(0.15-rand_x_start)/9)]

rand_curve_range_1 = [np.cos(i) for i in x]

print("x:", x)
print("rand_curve_range_1:", rand_curve_range_1)




rand_x_start_2 = random.uniform(-0.01,0.01)

print("rand_x_start:", rand_x_start_2)

x_2 = [i for i in np.arange(rand_x_start_2,0.15,(0.15-rand_x_start)/9)]



rand_curve_range_2 = [np.sin(i+np.pi/2) for i in x_2]

x_2 = [int(i*1000)+100 for i in x_2]


print("x_2:", x_2)



rand_curve_range_2 = [int(i*1000)-600 for i in rand_curve_range_2]
print("rand_curve_range_2:", rand_curve_range_2)


img_copy_3[rand_curve_range_2,x_2] = [0,0,0]

# x = [int(i*1.2) for i in x]



#img_copy_3[rand_curve_range_1,x] = [0,0,0]





'''
x = [i for i in range(300,400,1)]

rand_curve_range_1 = [10*np.sin(i*np.pi*0.01) for i in x]


rand_curve_range_1 = [int(i+400) for i in rand_curve_range_1]

print(x)
print(rand_curve_range_1)

x = [int(i*1.2) for i in x]


rand_curve_range_1 = [int(i*1.2) for i in rand_curve_range_1]

print(x)
print(rand_curve_range_1)

img_copy_3[rand_curve_range_1,x] = [0,0,0]
'''


rand_curve_range_1 = [int(i*1000)-500 for i in rand_curve_range_1]

x = [int(i*1000)+100 for i in x]

print("x:", x)
print("rand_curve_range_1:", rand_curve_range_1)

img_copy_3[rand_curve_range_1,x] = [0,0,0]






'''
pos = [35, 60]
for j in range(len(r_array)): 
    for i in range(len(r_array[j])):
        img_copy_2 = cv2.ellipse(img_copy_2, (pos[0], pos[1]), (int(r_array[j][i]), int(r_array[j][i])), 360, 0, 360, (random.randrange(0, 255),random.randrange(0, 255),random.randrange(0, 255)), 1) 
    pos[0] = pos[0]+70
'''


cv2.imshow("img_copy_2", img_copy_2)

cv2.imshow("img_copy_3", img_copy_3)


cv2.imshow("img_copy_1-2", img)







cv2.waitKey(0)
cv2.destroyAllWindows()

