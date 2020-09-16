import numpy as np
import cv2

import cv2
import numpy as np
import time 

so = cv2.imread("curve.jpg", 0)


coord=cv2.findNonZero(so)



coord2=np.nonzero(so)

print(coord)
print(coord2)