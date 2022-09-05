import numpy as np
import sys
import cv2

# You should replace these 3 lines with the output in calibration step
DIM=(1280, 800)
K=np.array([[570.745531298554, 0.0, 634.754590876372], [0.0, 571.2493277542346, 383.15948388971617], [0.0, 0.0, 1.0]])
D=np.array([[-0.026460843727798892], [0.12044929366868225], [-0.16522872093012023], [0.07746898581641035]])

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
#    for p in sys.argv[1:]:
    undistort("D:\Code\OVP800_IFM\myfilename3D.png")