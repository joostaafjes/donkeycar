import numpy as np
import cv2
import time
import random
import collections
#from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects

from PIL import Image
from matplotlib import cm
import os
import urllib.request

import cv2

class StopSignDetector(object):
    '''
    Requires an EdgeTPU for this part to work

    This part will run a EdgeTPU optimized model to run object detection to detect a stop sign.
    We are just using a pre-trained model (MobileNet V2 SSD) provided by Google.
    '''

    def download_file(self, url, filename):
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url, filename)

    def __init__(self, min_score, show_bounding_box, debug=True):
        MODEL_FILE_NAME = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
        LABEL_FILE_NAME = "coco_labels.txt"

        MODEL_URL = "https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
        LABEL_URL = "https://dl.google.com/coral/canned_models/coco_labels.txt"

        self.download_file(MODEL_URL, MODEL_FILE_NAME)
        self.download_file(LABEL_URL, LABEL_FILE_NAME)

        self.last_5_scores = collections.deque(np.zeros(5), maxlen=5)
        #self.engine = DetectionEngine(MODEL_FILE_NAME)
        self.interpreter = make_interpreter(MODEL_FILE_NAME)
        self.interpreter.allocate_tensors()
        self.inference_size = input_size(self.interpreter)
        print(f"inference_size: {self.inference_size}")
        
        self.labels = dataset_utils.read_label_file(LABEL_FILE_NAME)

        self.STOP_SIGN_CLASS_ID = 12
        self.min_score = min_score
        self.show_bounding_box = show_bounding_box
        self.debug = debug

    def convertImageArrayToPILImage(self, img_arr):
        #img = Image.fromarray(img_arr.astype('uint8'), 'RGB')

        #print(f"type img_arr: {type(img_arr)}")
        #print(f"type img: {type(img)}")
        #print(f"size before: {img.size}")
        #img_resized = img.resize(self.inference_size)
        #print(f"size after: {img_resized.size}")

        return img_resized

    '''
    Return an object if there is a traffic light in the frame
    '''
    def detect_stop_sign (self, img_arr):
        #img = self.convertImageArrayToPILImage(img_arr)
        #print(f'stop sign detector -> img_arr.shape:{img_arr.shape}')
        img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.inference_size)
        

        #print('detect stop sign')
        #ans = self.engine.detect_with_image(img,
        #                                  threshold=self.min_score,
        #                                  keep_aspect_ratio=True,
        #                                  relative_coord=False,
        #                                  top_k=3)
        run_inference(self.interpreter, img.tobytes())
        ans = get_objects(self.interpreter, self.min_score)[:3]
        
        max_score = 0
        traffic_light_obj = None
        if ans:
            for obj in ans:
                if (obj.id == self.STOP_SIGN_CLASS_ID):
                    #print(f'obj:{obj} -> score:{obj.score}')
                    if self.debug:
                        print("stop sign detected, score = {}".format(obj.score))
                    if (obj.score > max_score):
                        #print(obj.bounding_box)
                        traffic_light_obj = obj
                        max_score = obj.score

        # if traffic_light_obj:
        #     self.last_5_scores.append(traffic_light_obj.score)
        #     sum_of_last_5_score = sum(list(self.last_5_scores))
        #     # print("sum of last 5 score = ", sum_of_last_5_score)

        #     if sum_of_last_5_score > self.LAST_5_SCORE_THRESHOLD:
        #         return traffic_light_obj
        #     else:
        #         print("Not reaching last 5 score threshold")
        #         return None
        # else:
        #     self.last_5_scores.append(0)
        #     return None

        return traffic_light_obj

    def draw_bounding_box_new(self, obj, img):
        height, width, channels = img.shape
        scale_x, scale_y = width / self.inference_size[0], height / self.inference_size[1]
        
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, self.labels.get(obj.id, obj.id))

        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(img, label, (x0, y0+30),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        

    def run(self, img_arr, throttle, debug=False):
        if img_arr is None:
            return throttle, img_arr 

        # Detect traffic light object
        traffic_light_obj = self.detect_stop_sign(img_arr)

        if traffic_light_obj:
            if self.show_bounding_box:
                self.draw_bounding_box_new(traffic_light_obj, img_arr)
            return 0, img_arr
        else:
            return throttle, img_arr
