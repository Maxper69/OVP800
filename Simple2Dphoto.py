print("Importing library...")
import ifm3dpy
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import wget
import numpy as np
from matplotlib import pyplot as plt
from imutils import paths
import keyboard
from PIL import Image


from ifm3dpy import O3RCamera, FrameGrabber, ImageBuffer
print("Library Imported")


# Test connection if no message it's ok
o3r = O3RCamera()
config = o3r.get() #get the configuration saved on the VPU

# Set camera 2D PORT0 mode to RUN
config['ports']['port0']['state'] = "RUN" #Expecting a head on Port 0
o3r.set(config)

i =0
#print(im.distance_image())
while True:  # making a loop
    print("loop")
    if keyboard.is_pressed('t'):  # if key 't' is pressed trigger photo    
        

        fg = FrameGrabber(o3r, pcic_port=50010) #Expecting a head on Port 0 (Port 0 == 50010)
        im = ImageBuffer()
        #print(im.distance_image())

        if fg.wait_for_frame(im, 1000):

            #2D Data
            jpegdec =cv2.imdecode(im.jpeg_image(), cv2.IMREAD_UNCHANGED)
            plt.imshow(jpegdec)
            #plt.savefig('myfilename2D.png', dpi=100)
            cv2.imwrite('D:\Code\OVP800_IFM\myfilename2D'+ str(i) + '.png', jpegdec) 
            
            print("2D photo Saved")

            ###### CONVERT TO JPEG######
                # Load .png image
            image = cv2.imread('D:\Code\OVP800_IFM\myfilename2D'+ str(i) + '.png')

            # Save .jpg image
            cv2.imwrite('D:\Code\OVP800_IFM\myfilename2D'+ str(i) + '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            i=i+1