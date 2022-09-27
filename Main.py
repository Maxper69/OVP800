print("Importing library...")
import ifm3dpy
import json
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import sys
import wget
import tensorflow as tf
import object_detection
import numpy as np
import keyboard
from numpy import sqrt
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from matplotlib import pyplot as plt
from imutils import paths
from PIL import Image
from ifm3dpy import O3RCamera, FrameGrabber, ImageBuffer

print("Library Imported")

print("Function Definition...")
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def undistort(img_path):
    """ Permet de supprimer l'effet Fisheye natif sur une photo grace aux parametre K et D defini a l'exterieur de la fonction
    img_path : chemin de l'image .jpg ou .png """
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #cv2.imshow("undistorted", undistorted_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite('undistorded.png', undistorted_img)
    #image = Image.open('undistorted0.png')
    #image.show()

def distancePx3DTomm(x,y):
    convpx3Dtopx2D = 4.2857
    convpx2Dtomm = 1.51
    """Permet de recuperer les coordonnées d'un object sur l'image 3D en x et y et de calculer la distance en mm depuis le centre de l'image
    x = nb de pixel en x 
    y = nb de pixel en y """
    xpx3DFromCenter = abs(x -112)
    ypx3DFromCenter = abs(y -86)
    print(xpx3DFromCenter)
    print(ypx3DFromCenter)
    #convertion des pixelx 3D en pixels 2D
    xpx2DFromCenter = xpx3DFromCenter*convpx3Dtopx2D
    ypx2DFromCenter = ypx3DFromCenter*convpx3Dtopx2D

    #convertion des pixel 2D en mm
    xmmFromCenter = xpx2DFromCenter / convpx2Dtomm
    ymmFromCenter = ypx2DFromCenter / convpx2Dtomm
    print(xmmFromCenter)
    print(ymmFromCenter)
    #on calcule la distance en mm
    distmm = sqrt((xmmFromCenter**2)+(ymmFromCenter**2))
    print(distmm)
    return distmm

def hauteurZTomm(x3D,y3D,hBrut):
    distX3D = abs(x3D - 112)
    distY3D = abs(y3D - 86)
    distPx3D = sqrt((distX3D**2)+(distY3D**2))
    print(distPx3D)
    deformation = 0.00001466260826 * ((distPx3D)*(distPx3D))
    print(hBrut)
    print(deformation)
    hauteur=hBrut-deformation
    print (hauteur)
    return hauteur

print("Function Defined")

print("Brain loading...")
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

im_width=960
im_height=740

#Full 2D photo
DIM=(1280, 800)
K=np.array([[571.7451510174436, 0.0, 628.5786510869697], [0.0, 571.7523510309238, 382.0204664058281], [0.0, 0.0, 1.0]])
D=np.array([[-0.0190351011633013], [0.04800875479592394], [-0.035426669766489324], [0.00697316430866599]])


 # Load pipeline config and build a detection mode
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-29')).expect_partial()
print("Brain loaded")

# Test connection if no message it's ok
o3r = O3RCamera()
config = o3r.get() #get the configuration saved on the VPU

# Set camera 2D PORT0 mode to RUN
config['ports']['port0']['state'] = "RUN" #Expecting a head on Port 0
config['ports']['port2']['state'] = "RUN" #Expecting a head on Port 2
config['ports']['port2']['mode'] = "standard_range2m" #Expecting a head on Port 2
config['ports']['port2']['processing']['diParam']['mixedPixelThresholdRad'] = 0.05 #Expecting a head on Port 0
o3r.set(config)


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
            cv2.imwrite('D:\Code\OVP800_IFM\myfilename2D.png', jpegdec) 
            print("2D photo Saved")

            ###### CONVERT TO JPEG######
            # Load .png image
            image = cv2.imread('D:\Code\OVP800_IFM\myfilename2D.png')



            # Save .jpg image
            cv2.imwrite('cropped.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            #correction de la distortion 2D
            undistort("cropped.jpg")

            img = cv2.imread("undistorded.png")
            x=125 # decalage de 35px entre capteur 2D et 3D
            y=30
            w=960
            h=740
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite('bouchon26.png',crop_img)

            ###### Apply object recognition to the taken photo ######
            category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
            IMAGE_PATH = os.path.join('D:/Code/OVP800_IFM/bouchon26.png')
            print(IMAGE_PATH)

            img = cv2.imread(IMAGE_PATH)
            image_np = np.array(img)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            #print(detections)
            detections['num_detections'] = num_detections
            print(detections['num_detections'])

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            print(detections['detection_classes'])

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=50,
                        min_score_thresh=.8,
                        agnostic_mode=False)

            plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
            print(detections['detection_scores'])
            #(Ymin , Xmin, YMax, Xmax )
            print(detections['detection_boxes'])
            print(detections['detection_boxes'][0][0])
            print(detections['detection_boxes'][0][1])
            print(detections['detection_boxes'][0][2])
            print(detections['detection_boxes'][0][3])
            objbb = ((detections['detection_boxes'][0][0])*im_height),((detections['detection_boxes'][0][1])*im_width),((detections['detection_boxes'][0][2])*im_height),((detections['detection_boxes'][0][3])*im_width)

            print(objbb)
            objcenter = (((objbb[0]+objbb[2])/2),((objbb[1]+objbb[3])/2))
            

            imagein = cv2.circle(image_np_with_detections, ((int(objcenter[1])),(int(objcenter[0]))), radius=4, color=(0, 0, 255), thickness=4)

            cv2.imwrite('D:/Code/OVP800_IFM/bouchon26.jpg', imagein)
            image = Image.open('D:/Code/OVP800_IFM/bouchon26.jpg')
            image.show()

        fg = FrameGrabber(o3r, pcic_port=50012) #Expecting a head on Port 2 (Port 2 == 50012)
        im = ImageBuffer()     
        if fg.wait_for_frame(im, 1000):
            # 3D Data
            img3D = im.distance_image()

            #try to save in img
            #img = Image.fromarray(img3D, 'L')
            #img.save('my.png')
            #img.show()

            plt.imshow(img3D)
            plt.savefig('myfilename3D.png', dpi=100)
            print("3D photo Saved")

            h = 223
            l = 171
            im = Image.new('L',(224,172))
            for y in range(h):
                for x in range(l):
                    #print(img3D[x,y])
                    px = img3D[x,y]*1000
                    px = px-200
                    if px > 255:
                        px = 255
                    if px < 0:
                        px =0
                    im.putpixel((y,x),int(px))  #inversion de x et des y pour faire une rotation a 90°
            
            im.show()
            im.save(r'D:\Code\OVP800_IFM\3D.png') 

            #Resize image to 
            
            new_image = im.resize((960, 740))
            new_image.save('image_9603D.jpg')



            #correction de la distortion 3D
            #3D fisheye
            DIM=(960, 740)
            K=np.array([[572.632377821686, 0.0, 502.3727883845666], [0.0, 571.8920800894838, 353.64151728979175], [0.0, 0.0, 1.0]])
            D=np.array([[-0.023857951257414998], [0.06540923457587092], [-0.059851021477875854], [0.0081709509065374]])
            undistort("image_9603D.jpg")
            # Read the distance image and display a pixel in the center
            #dist = im.distance_image()
            #(width, height) = dist.shape
            #print(dist.shape)
            #print(dist[width//2,height//2])
            
            #xaskz = round((objcenter[0]-148)/4.4)
            #yaskz = round((objcenter[1]-22)/4.4)
            #print(yaskz)
            #print(dist[xaskz,yaskz])
            #print(dist[65,87])
            IMAGE_PATH = os.path.join('D:/Code/OVP800_IFM/undistorded.png')
            print(IMAGE_PATH)

            img = cv2.imread(IMAGE_PATH)
            image_np = np.array(img)

            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=50,
                        min_score_thresh=.6,
                        agnostic_mode=False)

            plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
            imagein = cv2.circle(image_np_with_detections, ((int(objcenter[1])),(int(objcenter[0]))), radius=4, color=(0, 0, 255), thickness=4)
            cv2.imwrite('D:/Code/OVP800_IFM/end.jpg', imagein)
            image = Image.open('D:/Code/OVP800_IFM/end.jpg')
            image.show()

            yaskz = round((objcenter[0]/4.2857))
            xaskz = round((objcenter[1]/4.2857))
            print(xaskz)
            print(yaskz)
            print(distancePx3DTomm(xaskz,yaskz))
            print("distance Z brut", img3D[yaskz,xaskz],"m")

            print(hauteurZTomm(yaskz,xaskz,img3D[yaskz,xaskz]))
            #print(img3D[xaskz,yaskz])

            '''print(img3D[yaskz,1])
            print(img3D[yaskz,2])
            print(img3D[yaskz,3])
            print(img3D[yaskz,4])
            print(img3D[yaskz,5])
            print(img3D[yaskz,6])
            print(img3D[yaskz,7])
            print(img3D[yaskz,8])
            print(img3D[yaskz,9])
            print(img3D[yaskz,10])
            print(img3D[yaskz,11])
            print(img3D[yaskz,12])
            print(img3D[yaskz,13])
            print(img3D[yaskz,14])
            print(img3D[yaskz,15])
            print(img3D[yaskz,16])
            print(img3D[yaskz,17])
            print(img3D[yaskz,18])
            print(img3D[yaskz,19])
            print(img3D[yaskz,20])
            print(img3D[yaskz,21])
            print(img3D[yaskz,22])
            print(img3D[yaskz,23])
            print(img3D[yaskz,24])
            print(img3D[yaskz,25])
            print(img3D[yaskz,26])
            print(img3D[yaskz,27])
            print(img3D[yaskz,28])
            print(img3D[yaskz,29])
            print(img3D[yaskz,30])
            print(img3D[yaskz,31])
            print(img3D[yaskz,32])
            print(img3D[yaskz,33])
            print(img3D[yaskz,34])
            print(img3D[yaskz,35])
            print(img3D[yaskz,36])
            print(img3D[yaskz,37])
            print(img3D[yaskz,38])
            print(img3D[yaskz,39])
            print(img3D[yaskz,40])
            print(img3D[yaskz,41])
            print(img3D[yaskz,42])
            print(img3D[yaskz,43])
            print(img3D[yaskz,44])
            print(img3D[yaskz,45])
            print(img3D[yaskz,46])
            print(img3D[yaskz,47])
            print(img3D[yaskz,48])
            print(img3D[yaskz,49])
            print(img3D[yaskz,50])
            print(img3D[yaskz,51])
            print(img3D[yaskz,52])
            print(img3D[yaskz,53])
            print(img3D[yaskz,54])
            print(img3D[yaskz,55])
            print(img3D[yaskz,56])
            print(img3D[yaskz,57])
            print(img3D[yaskz,58])
            print(img3D[yaskz,59])
            print(img3D[yaskz,60])
            print(img3D[yaskz,61])
            print(img3D[yaskz,62])
            print(img3D[yaskz,63])
            print(img3D[yaskz,64])
            print(img3D[yaskz,65])
            print(img3D[yaskz,66])
            print(img3D[yaskz,67])
            print(img3D[yaskz,68])
            print(img3D[yaskz,69])
            print(img3D[yaskz,70])
            print(img3D[yaskz,71])
            print(img3D[yaskz,72])
            print(img3D[yaskz,73])
            print(img3D[yaskz,74])
            print(img3D[yaskz,75])
            print(img3D[yaskz,76])
            print(img3D[yaskz,77])
            print(img3D[yaskz,78])
            print(img3D[yaskz,79])
            print(img3D[yaskz,80])
            print(img3D[yaskz,81])
            print(img3D[yaskz,82])
            print(img3D[yaskz,83])
            print(img3D[yaskz,84])
            print(img3D[yaskz,85])
            print(img3D[yaskz,86])
            print(img3D[yaskz,87])
            print(img3D[yaskz,88])
            print(img3D[yaskz,89])
            print(img3D[yaskz,90])
            print(img3D[yaskz,91])
            print(img3D[yaskz,92])
            print(img3D[yaskz,93])
            print(img3D[yaskz,94])
            print(img3D[yaskz,95])
            print(img3D[yaskz,96])
            print(img3D[yaskz,97])
            print(img3D[yaskz,98])
            print(img3D[yaskz,99])
            print(img3D[yaskz,100])
            print(img3D[yaskz,101])
            print(img3D[yaskz,102])
            print(img3D[yaskz,103])
            print(img3D[yaskz,104])
            print(img3D[yaskz,105])
            print(img3D[yaskz,106])
            print(img3D[yaskz,107])
            print(img3D[yaskz,108])
            print(img3D[yaskz,109])
            print(img3D[yaskz,110])
            print(img3D[yaskz,111])
            print(img3D[yaskz,112])
            print(img3D[yaskz,113])
            print(img3D[yaskz,114])
            print(img3D[yaskz,115])
            print(img3D[yaskz,116])
            print(img3D[yaskz,117])
            print(img3D[yaskz,118])
            print(img3D[yaskz,119])
            print(img3D[yaskz,120])
            print(img3D[yaskz,121])
            print(img3D[yaskz,122])
            print(img3D[yaskz,123])
            print(img3D[yaskz,124])
            print(img3D[yaskz,125])
            print(img3D[yaskz,126])
            print(img3D[yaskz,127])
            print(img3D[yaskz,128])
            print(img3D[yaskz,129])
            print(img3D[yaskz,130])
            print(img3D[yaskz,131])
            print(img3D[yaskz,132])
            print(img3D[yaskz,133])
            print(img3D[yaskz,134])
            print(img3D[yaskz,135])
            print(img3D[yaskz,136])
            print(img3D[yaskz,137])
            print(img3D[yaskz,138])
            print(img3D[yaskz,139])
            print(img3D[yaskz,140])
            print(img3D[yaskz,141])
            print(img3D[yaskz,142])
            print(img3D[yaskz,143])
            print(img3D[yaskz,144])
            print(img3D[yaskz,145])
            print(img3D[yaskz,146])
            print(img3D[yaskz,147])
            print(img3D[yaskz,148])
            print(img3D[yaskz,149])
            print(img3D[yaskz,150])
            print(img3D[yaskz,151])
            print(img3D[yaskz,152])
            print(img3D[yaskz,153])
            print(img3D[yaskz,154])
            print(img3D[yaskz,155])
            print(img3D[yaskz,156])
            print(img3D[yaskz,157])
            print(img3D[yaskz,158])
            print(img3D[yaskz,159])
            print(img3D[yaskz,160])
            print(img3D[yaskz,161])
            print(img3D[yaskz,162])
            print(img3D[yaskz,163])
            print(img3D[yaskz,164])
            print(img3D[yaskz,165])
            print(img3D[yaskz,166])
            print(img3D[yaskz,167])
            print(img3D[yaskz,168])
            print(img3D[yaskz,169])
            print(img3D[yaskz,170])
            print(img3D[yaskz,171])
            print(img3D[yaskz,172])
            print(img3D[yaskz,173])
            print(img3D[yaskz,174])
            print(img3D[yaskz,175])
            print(img3D[yaskz,176])
            print(img3D[yaskz,177])
            print(img3D[yaskz,178])
            print(img3D[yaskz,179])
            print(img3D[yaskz,180])
            print(img3D[yaskz,181])
            print(img3D[yaskz,182])
            print(img3D[yaskz,183])
            print(img3D[yaskz,184])
            print(img3D[yaskz,185])
            print(img3D[yaskz,186])
            print(img3D[yaskz,187])
            print(img3D[yaskz,188])
            print(img3D[yaskz,189])
            print(img3D[yaskz,190])
            print(img3D[yaskz,191])
            print(img3D[yaskz,192])
            print(img3D[yaskz,193])
            print(img3D[yaskz,194])
            print(img3D[yaskz,195])
            print(img3D[yaskz,196])
            print(img3D[yaskz,197])
            print(img3D[yaskz,198])
            print(img3D[yaskz,199])'''
            print(img3D[1,112])
            print(img3D[2,112])
            print(img3D[3,112])
            print(img3D[4,112])
            print(img3D[5,112])
            print(img3D[6,112])
            print(img3D[7,112])
            print(img3D[8,112])
            print(img3D[9,112])
            print(img3D[10,112])
            print(img3D[11,112])
            print(img3D[12,112])
            print(img3D[13,112])
            print(img3D[14,112])
            print(img3D[15,112])
            print(img3D[16,112])
            print(img3D[17,112])
            print(img3D[18,112])
            print(img3D[19,112])
            print(img3D[20,112])
            print(img3D[21,112])
            print(img3D[22,112])
            print(img3D[23,112])
            print(img3D[24,112])
            print(img3D[25,112])
            print(img3D[26,112])
            print(img3D[27,112])
            print(img3D[28,112])
            print(img3D[29,112])
            print(img3D[30,112])
            print(img3D[31,112])
            print(img3D[32,112])
            print(img3D[33,112])
            print(img3D[34,112])
            print(img3D[35,112])
            print(img3D[36,112])
            print(img3D[37,112])
            print(img3D[38,112])
            print(img3D[39,112])
            print(img3D[40,112])
            print(img3D[41,112])
            print(img3D[42,112])
            print(img3D[43,112])
            print(img3D[44,112])
            print(img3D[45,112])
            print(img3D[46,112])
            print(img3D[47,112])
            print(img3D[48,112])
            print(img3D[49,112])
            print(img3D[50,112])
            print(img3D[51,112])
            print(img3D[52,112])
            print(img3D[53,112])
            print(img3D[54,112])
            print(img3D[55,112])
            print(img3D[56,112])
            print(img3D[57,112])
            print(img3D[58,112])
            print(img3D[59,112])
            print(img3D[60,112])
            print(img3D[61,112])
            print(img3D[62,112])
            print(img3D[63,112])
            print(img3D[64,112])
            print(img3D[65,112])
            print(img3D[66,112])
            print(img3D[67,112])
            print(img3D[68,112])
            print(img3D[69,112])
            print(img3D[70,112])
            print(img3D[71,112])
            print(img3D[72,112])
            print(img3D[73,112])
            print(img3D[74,112])
            print(img3D[75,112])
            print(img3D[76,112])
            print(img3D[77,112])
            print(img3D[78,112])
            print(img3D[79,112])
            print(img3D[80,112])
            print(img3D[81,112])
            print(img3D[82,112])
            print(img3D[83,112])
            print(img3D[84,112])
            print(img3D[85,112])
            print(img3D[86,112])
            print(img3D[87,112])
            print(img3D[88,112])
            print(img3D[89,112])
            print(img3D[90,112])
            print(img3D[91,112])
            print(img3D[92,112])
            print(img3D[93,112])
            print(img3D[94,112])
            print(img3D[95,112])
            print(img3D[96,112])
            print(img3D[97,112])
            print(img3D[98,112])
            print(img3D[99,112])
            print(img3D[100,112])
            print(img3D[101,112])
            print(img3D[102,112])
            print(img3D[103,112])
            print(img3D[104,112])
            print(img3D[105,112])
            print(img3D[106,112])
            print(img3D[107,112])
            print(img3D[108,112])
            print(img3D[109,112])
            print(img3D[110,112])
            print(img3D[111,112])
            print(img3D[112,112])
            print(img3D[113,112])
            print(img3D[114,112])
            print(img3D[115,112])
            print(img3D[116,112])
            print(img3D[117,112])
            print(img3D[118,112])
            print(img3D[119,112])
            print(img3D[120,112])
            print(img3D[121,112])
            print(img3D[122,112])
            print(img3D[123,112])
            print(img3D[124,112])
            print(img3D[125,112])
            print(img3D[126,112])
            print(img3D[127,112])
            print(img3D[128,112])
            print(img3D[129,112])
            print(img3D[130,112])
            print(img3D[131,112])
            print(img3D[132,112])
            print(img3D[133,112])
            print(img3D[134,112])
            print(img3D[135,112])
            print(img3D[136,112])
            print(img3D[137,112])
            print(img3D[138,112])
            print(img3D[139,112])
            print(img3D[140,112])

            

    #if keyboard.is_pressed('q'):  # if key 'q' is pressed t
            print('Exiting...')
            # while quit
            # close port to avoid overheating
            config['ports']['port2']['state'] = "IDLE" #Expecting a head on Port 0
            o3r.set(config)
            break  # finishing the loop