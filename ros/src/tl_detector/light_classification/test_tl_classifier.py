import tensorflow as tf
import numpy as np


#from PIL import ImageDraw

#import time
from timeit import default_timer as timer
import cv2

CLASS_TRAFFIC_LIGHT = 10


# Frozen inference graph files.

SSD_MOBILENET = 'models/ssd_mobilenet_v1.pb'
SSD_INCEPTION = 'models/ssd_inception_v2.pb'

# This one has a GOOD accuracy - for a budget around or below 100 ms
FASTER_RCNN_INCEPTION = 'models/faster_rcnn_inception_v2.pb'

OBJDET_GRAPH = FASTER_RCNN_INCEPTION
#OBJDET_GRAPH = SSD_INCEPTION
#OBJDET_GRAPH = SSD_MOBILENET

#                         GTX 1080 TI
# SSD_MOBILENET        :    32 ms
# SSD_INCEPTION        :    38 ms
# FASTER_RCNN_INCEPTION:    69 ms



def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

def select_lighton_real(img):
    """Applies color selection for high L and S """
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    lower = np.array([ 50,   150, 150], dtype="uint8")
    upper = np.array([ 100, 255, 255], dtype="uint8")
    tl_mask = cv2.inRange(hls_img, lower, upper)
    return cv2.bitwise_and(img, img, mask = tl_mask)


def select_red_simu(img):
    # BGR
    lower = np.array([ 0,   0, 200], dtype="uint8")
    upper = np.array([ 50, 50, 255], dtype="uint8")
    red_mask = cv2.inRange(img, lower, upper)
    return cv2.bitwise_and(img, img, mask = red_mask)


def classify_red_tl(image_np, boxes, classes, scores, thickness=4):
    """Draw bounding boxes on the image"""
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        score = scores[i]
        if class_id == CLASS_TRAFFIC_LIGHT:
            h = top - bot
            w = right - left
            if h <= 1.5 * w:
                continue # Truncated Traffic Ligth box
            cv2.rectangle(image_np,(left, top), (right, bot), (255, 0, 0), thickness) # BGR format for color
            tl_img = image_np[int(bot):int(top), int(left):int(right)]

            tl_img_simu = select_red_simu(tl_img) # SELECT RED
            tl_img_real = select_lighton_real(tl_img) # SELECT TL
            tl_img = (tl_img_simu + tl_img_real) / 2

            gray_tl_img = cv2.cvtColor(tl_img, cv2.COLOR_RGB2GRAY)
            nrows, ncols = gray_tl_img.shape[0], gray_tl_img.shape[1]

            # compute center of mass of RED points
            mean_row = 0
            mean_col = 0
            npoints = 0
            for row in range(nrows):
                for col in range(ncols):
                    if (gray_tl_img[row, col] > 0): 
                        mean_row += row
                        mean_col += col
                        npoints += 1
            if npoints > 0:
              mean_row = float(mean_row / npoints) / nrows
              mean_col = float(mean_col / npoints) / ncols
              print(mean_row, mean_col, npoints)

              # if normalized center of mass of RED points 
              # is in the upper part of detected Traffic Light Box
              # THEN it is a RED traffic light
              if npoints > 10 and mean_row < 0.33:
                  text = "RED LIGHT score=%.3f" % score
                  cv2.putText(image_np, text, (int(left), int(bot)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, lineType=cv2.LINE_AA)
                  return image_np, True

            return image_np, False
            #cv2.imshow("image", image_np)
            #cv2.waitKey(0)
        

def load_graph(graph_file, use_xla=False):
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess, ops


sess, _ = load_graph(OBJDET_GRAPH)
detection_graph = sess.graph

# The input placeholder for the image.
# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
# The classification of the object (integer id).
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


# Load a sample image.
#image = Image.open('./assets/sample1.jpg')

#test_image = scipy.misc.imread("assets/sample1.jpg")
#print(test_image.shape)


#def predict_image(image, image_np):
def detect_tl(image):
    start = timer()
    trt_image = np.copy(image)
    image_np = np.expand_dims(np.asarray(trt_image, dtype=np.uint8), 0)
    #print(image_np.shape)

    start_inference = timer()
    # Actual detection.
    (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], 
                                        feed_dict={image_tensor: image_np})
    end_inference = timer()

    # Remove unnecessary dimensions
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    confidence_cutoff = 0.8
    #confidence_cutoff = 0.5
    # Filter boxes with a confidence score less than `confidence_cutoff`
    boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

    # The current box coordinates are normalized to a range between 0 and 1.
    # This converts the coordinates actual location on the image.
    image_np = np.squeeze(image_np)
    width = image_np.shape[1]
    height = image_np.shape[0]
    box_coords = to_image_coords(boxes, height, width)

    # Each class with be represented by a differently colored box
    image_np, is_red = classify_red_tl(image_np, box_coords, classes, scores)

    end = timer()

    if dump_time:
        time_inference = end_inference - start_inference
        time_img_processing = (end - start) - time_inference
        print("time: inference {:.6f} post-processing {:.6f}".format(time_inference, time_img_processing))
    return image_np, is_red


dump_time = True

#test_image = cv2.imread('img/left0000.jpg')
test_image = cv2.imread('img/left0144.jpg')
#test_image = cv2.imread('img/image1.png')
#test_image = cv2.imread('img/image2.png')
#test_image = cv2.imread('img/image3.png')
#test_image = cv2.imread('img/image4.png')
#test_image = cv2.imread('img/image210.jpg')
#test_image = cv2.imread('img/image214.jpg')
#test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

for i in range(5):
    pred_image, is_red = detect_tl(test_image)
    if is_red:
        print("RED")
    else:
        print("NOT RED")
cv2.imwrite("img/pred_image.png", pred_image)
