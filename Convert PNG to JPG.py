
from imutils import paths
import cv2

i=1
for imagePath in paths.list_images("D:\Code\OVP800_IFM"):
    # Load .png image
    image = cv2.imread(imagePath)

    # Save .jpg image
    cv2.imwrite('bouchon%s.jpg' %i, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    i += 1