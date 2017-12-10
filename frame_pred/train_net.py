#!/usr/bin/python

import tensorflow as tf
import numpy as np
import cv2

import sys
import os

import argparse
import time


from tensorpack import *

from tensorpack.utils.gpu import get_nr_gpu

from ACVPnet import *


def main():
    parser = argparse.ArgumentParser(description="Train the ACVP image generation network.")
    parser.add_argument("--episode_dir", type=str, default="/home/matt/acvp/frame_pred/fake_data", \
        help="Directory containing episode subdirectories. NOTE: It must be the absolute path.")
    parser.add_argument("--network", choices=["cnn", "rnn", "naff"], default="cnn",
        help="types of models are cnn, rnn, and naff")

    args = parser.parse_args()

    logger.auto_set_dir()
    
    batch_size_cnn_naff = 32
    batch_size_rnn = 4

    avgs = calcAvgPixel(args.episode_dir)
    print "Average pixel values:", avgs, "TODO(MATT): Save these in a file, load at runtime"
    items = readFilenamesAndActions(args.episode_dir)

    dataflow = AtariReplayDataflow(items, args.network, avgs, shuffle=True, \
        batch_size=(batch_size_rnn if args.network == "rnn" else batch_size_cnn_naff))
    # NOTE: Frame combination is handled in AtariReplayDataflow's read_item, these shouldn't be necessary
    # dataflow = MapDataComponent(lambda a: np.concat(a, axis=1), 0)
    # dataflow = PrefetchDataZMQ(dataflow, 4)
    # TODO(Ben/Matt), when running:
    #Set TENSORPACK_PIPE_DIR=/ltmp/
    config = TrainConfig(
        model=ACVPModel(args.network, avgs),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(0, 1e-4), (3, 1e-5), (5, 1e-5)])
            ScheduledHyperParamSetter('steps', [(0, 1), (3, 3), (5, 5)])
        ],
        max_epoch=7,
        session_init=None# if not args.load else SaverRestore(args.load)
    )
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))


if __name__ == '__main__':
    main()