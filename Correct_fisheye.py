import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# You should replace these 3 lines with the output in calibration step
DIM=(960, 740)
K=np.array([[572.632377821686, 0.0, 502.3727883845666], [0.0, 571.8920800894838, 353.64151728979175], [0.0, 0.0, 1.0]])
D=np.array([[-0.023857951257414998], [0.06540923457587092], [-0.059851021477875854], [0.0081709509065374]])

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #cv2.imshow("undistorted", undistorted_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite('undistorded0.png', undistorted_img)
    image = Image.open('undistorted0.png')
    image.show()
if __name__ == '__main__':
#    for p in sys.argv[1:]:
    undistort("bouchon26.png")