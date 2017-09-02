MODELS_DIR = "/home/ubuntu/detector"
OBJECT_DETECTION_DIR = MODELS_DIR + "/object_detection"

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = OBJECT_DETECTION_DIR + "/" + MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = OBJECT_DETECTION_DIR + '/data/mscoco_label_map.pbtxt'

#MODEL_NAME = 'rfcn_resnet101_coco_11_06_2017'
