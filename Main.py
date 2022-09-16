print("Importing library...")
import ifm3dpy
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import sys
import wget
import tensorflow as tf
import object_detection
import numpy as np
import keyboard
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
                        min_score_thresh=.8,
                        agnostic_mode=False)

            plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
            imagein = cv2.circle(image_np_with_detections, ((int(objcenter[1])),(int(objcenter[0]))), radius=4, color=(0, 0, 255), thickness=4)
            cv2.imwrite('D:/Code/OVP800_IFM/end.jpg', imagein)
            image = Image.open('D:/Code/OVP800_IFM/end.jpg')
            image.show()

            xaskz = round((objcenter[0]/4.2857))
            yaskz = round((objcenter[1]/4.2857))
            print(yaskz)
            print(xaskz)
            print(img3D[xaskz,yaskz])


    #if keyboard.is_pressed('q'):  # if key 'q' is pressed t
            print('Exiting...')
            # while quit
            # close port to avoid overheating
            config['ports']['port2']['state'] = "IDLE" #Expecting a head on Port 0
            o3r.set(config)
            break  # finishing the loop