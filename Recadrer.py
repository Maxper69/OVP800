import cv2
img = cv2.imread("undistorded0.png")
x=125 # decalage de 35px entre capteur 2D et 3D
y=30
w=960
h=740
crop_img = img[y:y+h, x:x+w]
cv2.imwrite('bouchon26.png',crop_img)