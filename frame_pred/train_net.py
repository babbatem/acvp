#!/usr/bin/python

import tensorflow as tf
import numpy as np
import cv2

import sys
import os
import pprint
import argparse
import time
import random


from tensorpack import *
from tensorpack.dataflow import *
from tensorpack.callbacks import *
from tensorpack.utils.gpu import get_nr_gpu

from ACVPnet import *

class OurJSONWriter(JSONWriter):

    def _trigger_step(self):
	print("TRIGGERD")
	loss = self.get_tensors_maybe_in_tower(['complete_loss:0'])[0]#	tf.get_default_graph().get_tensor_by_name('complete_loss:0')
        print(loss)
	# will do this in trigger_epoch
	self.trainer.monitors.put_summary(loss)
	self._push()
	#if self.local_step % 2 == 0:
	#    self._push()


def main():
    parser = argparse.ArgumentParser(description="Train the ACVP image generation network.")
    parser.add_argument("--episode_dir", type=str, default="/data/people/mcorsaro/acvp_03epsil/", \
        help="Directory containing episode subdirectories. NOTE: It must be the absolute path.")
    parser.add_argument("--avg_pixel_file", type=str, default="2017_12_11_15_39_05_avg_pixels.txt", \
        help="Directory containing average pixel file for a given dataset." + \
        " If unspecified, a new file will be created. NOTE: It must be the absolute path.")
    # TODO: Optional
    parser.add_argument("--load", help="optional .checkpoint or .data model file")
    parser.add_argument("--phase", choices=["1", "2", "3"], default="1",
        help="types of models are cnn, rnn, and naff")
    parser.add_argument("--network", choices=["cnn", "rnn", "naff"], default="cnn",
        help="types of models are cnn, rnn, and naff")

    args = parser.parse_args()

    logger.set_logger_dir(timestamp() + '_' + str(random.randint(0, 10000000)) + "train_log")
    
    nr_gpu = max(get_nr_gpu(), 1)
    batch_size_cnn_naff = 32//nr_gpu if args.phase == "1" else 8//nr_gpu
    batch_size_rnn = 4//nr_gpu

    learning_rate = 1e-4 if args.phase == "1" else 1e-5
    num_epochs = 3 if args.phase == "1" else 2
    steps = 1 if args.phase == "1" else (3 if args.phase == "2" else 5)
    in_channel_size = 3 if args.network == "rnn" else 12

    items = readFilenamesAndActions(args.episode_dir, steps)
    avgs = calcAvgPixel(args.episode_dir) if args.avg_pixel_file == "" else \
        np.loadtxt(args.avg_pixel_file)
    if args.avg_pixel_file == "":
        # Save locally
        np.savetxt(timestamp() + "_avg_pixels.txt", avgs)
    print "Average pixel values:", avgs

    dataflow = AtariReplayDataflow(items, avgs, args.network, steps, shuffle=True, \
        batch_size=(batch_size_rnn if args.network == "rnn" else batch_size_cnn_naff))
    dataflow = PrefetchDataZMQ(dataflow, 4)
    dataflow = PrintData(dataflow)
    # TODO(Ben/Matt), when running:
    #Set TENSORPACK_PIPE_DIR=/ltmp/
    '''
    print "Now testing dataflow speed."
    TestDataSpeed(dataflow).start()
    '''
    config = TrainConfig(
        model=ACVPModel(args.network, avgs, learning_rate, steps, in_channel_size),
        dataflow=dataflow,
	callbacks=[ModelSaver(), TensorPrinter(['tower0/complete:0'])],
	#monitors=[TFEventWriter(), OurJSONWriter(), ScalarPrinter()],
	#steps_per_epoch=10000,
	max_epoch=num_epochs,
        session_init=SaverRestore(args.load) if args.load else None 
    )
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))

if __name__ == '__main__':
    main()
