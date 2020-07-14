# pTag

Ptag detector is used to detect price tag and scan for barcode.

# Running pTag on Video!
```sh
python3 TFLite_detection_video.py --video=tflite-model/eval/aisle08.mp4 --threshold=0.7 --blur=110.0 --cw=150 --modeldir=tflite-model/ --edgetpu
```

# Running pTag on Image!
```sh
python3 TFLite_detection_image.py --imagedir=tflite-model/eval/images/ --threshold=0.6 --modeldir=tflite-model/ --edgetpu
```

# Running pTag on Webcam!
```sh
python3 TFLite_detection_video.py --threshold=0.7 --blur=80.0 --cw=180 --resolution=640x480 --modeldir=tflite-model/ --edgetpu
```

### Features

- '--modeldir', help='Folder the .tflite file is located in',
                    required=True)
- '--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
- '--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
- '--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
- '--blur', help='Minimum blur threshold for displaying detected objects',
                    type=float, default=100.0)
- '--cw', help='maximum width threshold of detected ptag to save',
                    type=float, default=200.0)
- '--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

- '--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                                        default='1280x720'(webcam detector only)

- '--video', help='Name of the video file', default='test.mp4'(video detector only)

- '--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None(image detector only)
                    
                    
### Changes if desired

#### print the barcode type and data to the terminal and location to save the found barcode.
- cv2.imwrite('snap/test/Bar-Img_{}.png'.format(cv2.getTickCount()), img_bar) (line # 279)

