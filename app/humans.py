from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import sys

import tensorflow as tf
from PIL import Image
import numpy as np


sys.path.append("../models")
sys.path.append("../models/object_detection")

import utils.visualization_utils as vis_util
from utils import label_map_util
import model_download



log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)



PATH_TO_LABELS = '../models/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 100 # person has index 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)

log.debug("building graph")

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_download.PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

log.debug("loaded graph into memory")


def load_image_into_numpy_array(image_path):
    image = Image.open(image_path)
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def find_people(image_path, sess):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image_path)
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    boxes, scores, classes, num_detections = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return boxes, scores, classes, image_np


def get_people(img, sess, min_score_thresh=0.5):
    boxes, scores, classes, image_np = find_people(img, sess)
    scores_above_threshold = [s for s in scores[0] if s > min_score_thresh]
    
    return [
        {"score": score, "box": box}
        for box, score in zip(boxes[0], scores_above_threshold)
    ]


def benchmark():
    total_people = 0
    total_time = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            imgs = ["sc%i.png" % i for i in range(1,11)]
            for img in imgs:
                then = datetime.now().timestamp()

                log.debug("processing %s", img)
                people = get_people(img, sess)
                log.debug(people)

                how_long_it_took = datetime.now().timestamp() - then
                total_time = how_long_it_took + total_time
                total_people = total_people + len(people)

                log.debug("Done in %s secs", how_long_it_took)
    
    log.debug("Identified %i people in %i seconds from %i photos",
          total_people, total_time, len(imgs))


def draw_on_image(image_path, outfile, session):
    boxes, scores, classes, image_np = find_people(image_path, session)
    vis_util.visualize_boxes_and_labels_on_image_array(
                      image_np,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,
                      line_thickness=8)
    plt.imsave(outfile, image_np)

if __name__ == "__main__":
    benchmark()

