import cv2
img = cv2.imread("myfilename2D0.jpg")
x=160
y=30
w=960
h=740
crop_img = img[y:y+h, x:x+w]
cv2.imwrite('1.JPG',crop_img)