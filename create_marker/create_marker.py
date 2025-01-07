import numpy as np
import cv2
from cv2 import aruco


aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
ar_id = 3
img_size = 400
ar_img = aruco.drawMarker(aruco_dict, ar_id, img_size)

while True:
    cv2.imshow("aruco_marker", ar_img)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cv2.imwrite("marker.png", ar_img)
