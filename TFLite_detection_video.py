# Import packages
from pyzbar import pyzbar
from PIL import Image, ImageEnhance
from BarcodeScan import BarcodeScan
import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--blur', help='Minimum blur threshold for displaying detected objects',
                    type=float, default=100.0,)
parser.add_argument('--cw', help='maximum width threshold of detected ptag to save',
                    type=float, default=200.0,)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

#Barcode Scan
#barcode_scan = BarcodeScan().start()

while(video.isOpened()):
    
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
          print('Reached the end of the video!')
          #barcode_scan.stop()
          break
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray_img)
    print("blur value : {}", format(fm))
    #barcode_scan.fm = fm
    #barcode_scan.stop()
    #Allow only non-blurry frames for processing
    if fm > args.blur:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    
    
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
            
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
                ROI = frame[ymin:ymax, xmin:xmax]
                print(xmax - xmin,ymax - ymin)
                if((xmax - xmin) > args.cw):
                    #barcode_scan.start()
                    #barcode_scan.frame = ROI
                    img = Image.fromarray(np.array(ROI)) 
                    
                    img_contrast = ImageEnhance.Contrast(img)
                    img_enhance = img_contrast.enhance(2)
                    
                    img_sharp = ImageEnhance.Sharpness(img_enhance)
                    img_enhance = img_sharp.enhance(3)
                    
                    #img_enhance.save('snap/Sharp-img_{}.png'.format(cv2.getTickCount()))
                    
                    
                    #barcode detection
                    img_bar = np.array(img_enhance)
                    print("imW:{}".format(xmax - xmin),"imH:{}".format(ymax - ymin))

                    img_bar = cv2.resize(img_bar,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
                    
                    kernel = np.ones((5,5),np.uint8)
                    img_bar = cv2.cvtColor(img_bar, cv2.COLOR_BGR2GRAY)
                    # grab the (x, y) coordinates of all pixel values that
                    # are greater than zero, then use these coordinates to
                    # compute a rotated bounding box that contains all
                    # coordinates
                    coords = np.column_stack(np.where(img_bar > 0))
                    angle = cv2.minAreaRect(coords)[-1]
                    # the `cv2.minAreaRect` function returns values in the
                    # range [-90, 0); as the rectangle rotates clockwise the
                    # returned angle trends to 0 -- in this special case we
                    # need to add 90 degrees to the angle
                    #print("Skew angle : {}",format(angle))
                    if angle < -45:
                        angle = -(90 + angle)
                    # otherwise, just take the inverse of the angle to make
                    # it positive
                    else:
                        angle = -angle
    
                    #print("[INFO] angle: {:.3f}".format(angle))
                    # rotate the image to deskew it
                    (h, w) = img_bar.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img_bar = cv2.warpAffine(img_bar, M, (w, h),flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
                    if fm < 100 :
                        blur_text = "Blur"
                    else:
                        blur_text = "No Blur"
                
                    print("Image has : "+ blur_text)
                    # find the barcodes in the image and decode each of the barcodes
                    barcodes = pyzbar.decode(img_bar)
            
                    # loop over the detected barcodes
                    for barcode in barcodes:
                        # extract the bounding box location of the barcode and draw the
                        # bounding box surrounding the barcode on the image
                        (x, y, w, h) = barcode.rect
                        cv2.rectangle(img_bar, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # the barcode data is a bytes object so if we want to draw it on
                        # our output image we need to convert it to a string first
                        barcodeData = barcode.data.decode("utf-8")
                        barcodeType = barcode.type
                        # draw the barcode data and barcode type on the image
                        text = "{} ({})".format(barcodeData, barcodeType)
                        cv2.putText(img_bar, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            
                        #cv2.putText(img_bar,'{}'.format(blur_text),(280,150),cv2.FONT_HERSHEY_SIMPLEX,1,(100,220,255),1,cv2.LINE_AA)
                        #cv2.putText(img_bar,'thresh:{0:.2f}'.format(fm),(280,180),cv2.FONT_HERSHEY_SIMPLEX,1,(100,220,255),1,cv2.LINE_AA)
    
                        # print the barcode type and data to the terminal
                        print("[INFO] Found barcode" + text)
                        cv2.imwrite('snap/test/Bar-Img_{}.png'.format(cv2.getTickCount()), img_bar)
                    
            # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,'Blurry',(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()