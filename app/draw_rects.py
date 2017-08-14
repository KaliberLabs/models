#!/usr/bin/env
import glob

import humans

# ASSIGN THESE TO DIRECTORIES OF INPUT FRAMES AND A DIRECTORY
# OF TO OUTPUT IMAGES WITH DRAWN BOUNDING BOXES
IMAGE_GLOB = "../data/images/frames/*.png"
OUT_DIR = "../data/images/rects/"


def list_images():
    return sorted(glob.glob(IMAGE_GLOB))


if __name__ == "__main__":
    with humans.detection_graph.as_default():
        with humans.tf.Session(graph=humans.detection_graph) as session:
            imgs = list(list_images())
            for index, img in enumerate(imgs):
                out_name = img.split("/")[-1]
                print(img, index, "out of", len(imgs))
                humans.draw_on_image(img, OUT_DIR + out_name, session)

