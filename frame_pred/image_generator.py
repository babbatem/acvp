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
    def __init__(self, model_file):
        config = 
        self.predictor = OfflinePredictor(config)
    

    def predict_frame(old_frames, action):
        # TODO: Delete this function unless it has to return self.predictor()[0]
        return self.predictor(old_frames, action)