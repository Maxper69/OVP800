import ifm3dpy
import json
import matplotlib.pyplot as plt
import cv2
#for test to read img
import matplotlib.image as mpimg

from ifm3dpy import O3RCamera, FrameGrabber, ImageBuffer

#Test installation
print(ifm3dpy.__version__)


# Test connection if no message it's ok
o3r = O3RCamera()
config = o3r.get() #get the configuration saved on the VPU
#print test config in json
print(json.dumps(config, indent=4))

# Set camera 2D PORT0 mode to RUN
config['ports']['port0']['state'] = "RUN" #Expecting a head on Port 0
o3r.set(config)

# Set camera 3D PORT2 mode to RUN (OR CONF)
config['ports']['port2']['state'] = "RUN" #Expecting a head on Port 2
config['ports']['port2']['mode'] = "standard_range2m" #Expecting a head on Port 2
config['ports']['port2']['processing']['diParam']['mixedPixelThresholdRad'] = 0.05 #Expecting a head on Port 0
o3r.set(config)

fg = FrameGrabber(o3r, pcic_port=50012) #Expecting a head on Port 2 (Port 2 == 50012)
im = ImageBuffer()
#print(im.distance_image())

if fg.wait_for_frame(im, 1000):
    # 3D Data
    img3D = im.distance_image()
    plt.imshow(img3D)
    plt.savefig('myfilename3D.png', dpi=100)
    plt.show()
    

    dist = im.distance_image()
    (width, height) = dist.shape
    print(dist[width//2,height//2])


fg = FrameGrabber(o3r, pcic_port=50010) #Expecting a head on Port 0 (Port 0 == 50010)
im = ImageBuffer()
#print(im.distance_image())

if fg.wait_for_frame(im, 1000):

    #2D Data
    jpegdec =cv2.imdecode(im.jpeg_image(), cv2.IMREAD_UNCHANGED)
    plt.imshow(jpegdec)
    #plt.savefig('myfilename2D.png', dpi=100)
    cv2.imwrite('D:\Code\OVP800_IFM\myfilename2D.png', jpegdec) 
    plt.show()
    

    # Read the distance image and display a pixel in the center


config['ports']['port2']['state'] = "IDLE" #Expecting a head on Port 0
o3r.set(config)

