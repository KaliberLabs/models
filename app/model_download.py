import tarfile
import os

import six.moves.urllib as urllib

import config

IMAGE_SIZE = (24, 16)

if __name__ == "__main__":

    opener = urllib.request.URLopener()
    opener.retrieve(config.DOWNLOAD_BASE + config.MODEL_FILE, config.MODEL_FILE)
    tar_file = tarfile.open(config.MODEL_FILE)

    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
