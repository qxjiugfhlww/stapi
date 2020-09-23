import numpy as np
import cv2
import time 
import math


height_1 = 50
height_2 = 100


x_l = 0
y_l = 5

x_b = -5
y_b = 0

x = -2

y =  ((x-x_l)*(y_b-y_l))/(x_b-x_l)+y_l

print(y)

f= open("save_height.txt","w+")

f.write(str(y))

f.close() 
