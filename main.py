#!/usr/bin/env python3
import argparse
import logging
import time
import numpy as np
from waggle.plugin import Plugin
from waggle.data.vision import Camera


def process_frame(frame):
    # we assume frame shape is (H, W, 3) for an RGB image
    mean = np.mean(frame, (0, 1)).astype(float)
    min = np.min(frame, (0, 1)).astype(float)
    max = np.max(frame, (0, 1)).astype(float)
    return {
        "mean": mean,
        "min": min,
        "max": max,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=0, help="camera device to use")
    parser.add_argument("--interval", default=10, type=float, help="sampling interval in seconds")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    logging.info("starting plugin. will process a frame every %ss", args.interval)

    with Plugin() as plugin:
        cam = Camera(args.device)

        for sample in cam.stream():
            logging.info("processing frame")
            results = process_frame(sample.data)

            logging.info("image summary")
            logging.info("mean: %s", results["mean"])
            logging.info("min: %s", results["min"])
            logging.info("max: %s", results["max"])

            plugin.publish("image.mean.red", results["mean"][0])
            plugin.publish("image.mean.green", results["mean"][1])
            plugin.publish("image.mean.blue", results["mean"][2])

            plugin.publish("image.min.red", results["min"][0])
            plugin.publish("image.min.green", results["min"][1])
            plugin.publish("image.min.blue", results["min"][2])

            plugin.publish("image.max.red", results["max"][0])
            plugin.publish("image.max.green", results["max"][1])
            plugin.publish("image.max.blue", results["max"][2])

            logging.info("published summary")

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
