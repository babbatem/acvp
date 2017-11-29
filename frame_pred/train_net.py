#!/usr/bin/python

import tensorflow as tf
import numpy as np
import cv2

import sys
import os

import argparse
import time

from ACVPnet import *

def main():
    parser = argparse.ArgumentParser(description="Train the ACVP image generation network.")
    parser.add_argument("dirs", metavar="dir", type=str, nargs='+',
        help="Directory containing subdirectories corresponding to each episode")
    parser.add_argument("--network", choices=["cnn", "rnn", "naff"], default="cnn",
        help="types of models are cnn, rnn, and naff")
    parser.add_argument("--model_file", type=str, default=str(time.time()),
        help="filename (with directory) to save model and average pixel values in")

    args = parser.parse_args()
    print "Using network type", args.network
    model_path_list = args.model_file.split('/')[:-1]
    model_path = ""
    for subpath in model_path_list:
        model_path += subpath + "/"
    if model_path == '':
        model_path = "."
    if not os.path.exists(model_path):
        print "Given path", model_path, "does not exist, storing model in current directory"
        model_path = "."
    model_name = args.network + "_" + args.model_file.split('/')[-1]
    network_model_name = model_name if model_name[-5:] == ".ckpt" else model_name + ".ckpt"
    pixel_avg_model_name = "avg_pixels_" + model_name + ".txt"
    print("Saving model in directory {} and files {} and {}".format(model_path if model_path != "" else '.', \
        network_model_name, pixel_avg_model_name))

    # Walk through the given directory, pull out list of lists (one per episode) of filenames,
    # and list of list of corresponding actions
    (frame_lists, action_lists) = readFilenamesAndActions(args.dirs)

    # Calculate average pixel values, save to file
    pixels_avg = calcAvgPixel(frame_lists)
    np.savetxt(model_path + '/' + pixel_avg_model_name, pixels_avg)

    network = ACVPnet(args.network, pixels_avg)
    network.train(frame_lists, action_lists,  network_model_name, model_path)
    

if __name__ == '__main__':
    main()