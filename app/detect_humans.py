#!/usr/bin/env python
import argparse
import logging
import csv
import os

import humans

argparser = argparse.ArgumentParser(
    description="Run human detection on a list of images and dump csv to outcsv")
argparser.add_argument("outcsv", help="output csv file")
argparser.add_argument("image_files", nargs="*", help="list of frames")

HEADERS = "filename confidence ymin xmin ymax xmax".split(" ")


def main(args):
    with open(args.outcsv, "w") as output:
        writer = csv.writer(output)
        writer.writerow(HEADERS)
        writer.writerows(person_generator(args.image_files))


def person_generator(image_files):
    for filename in image_files:
        people = humans.get_people(filename, session)
        for person in people:
            yield (os.path.basename(filename), person["score"], *person["box"])


if __name__ == "__main__":
    logging.debug("Starting App")
    with humans.detection_graph.as_default():
        with humans.tf.Session(graph=humans.detection_graph) as session:
            main(argparser.parse_args())
