#!/usr/bin/python

import tensorflow as tf
import numpy as np
import cv2

import sys
import os

import argparse
import time


from tensorpack import *
from tensorpack.dataflow import *

from tensorpack.utils.gpu import get_nr_gpu

from ACVPnet import *

class ACVPGenerator:
    def __init__(self, model_file, arch):
        avg = np.loadtxt("frame_pred/2017_12_11_15_39_05_avg_pixels.txt")
        config = PredictConfig(
            model=ACVPModel('cnn', avg, 0, 1, 12),
            session_init=get_model_loader(model_file),
            input_names=['input','action'],
            output_names=['A/cnn_prediction'])


        self.predictor = OfflinePredictor(config)
    

    def predict_frame(old_frames, action, arch):
        # TODO: Delete this function unless it has to return self.predictor()[0]
        return self.predictor(old_frames, action)