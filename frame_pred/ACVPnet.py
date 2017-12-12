#!/usr/bin/python

import tensorflow as tf
from tensorpack import *
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import get_global_step_var
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope


import numpy as np
import cv2

import sys
import os

import argparse
import time

NUM_ACTIONS = 6

def timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

# Show an img for s seconds(?)
def show_img(img, s):
    cv2.imshow("MyImage", img)
    cv2.waitKey(s*1000)

class ACVPModel(ModelDesc):
    def __init__(self, network_type, avg, learning_rate, k, in_channel_size):
        super(ACVPModel, self).__init__()
        self.network_type = network_type
        self.avg = avg
        self.learning_rate = learning_rate
        self.k = k
        self.in_channel_size = in_channel_size

    def _get_inputs(self):
        # Images are either concatenated (12 channels in cnn/naff, 3 in rnn)
        return [InputDesc(tf.float32, (None, 210, 160, self.in_channel_size), 'input'),
                # Give the next 5 images and actions that yield them as labels/actions in all phases
                InputDesc(tf.float32, (None, 208, 160, 15), 'label'),
                InputDesc(tf.int32, (None, 5), 'action')]

    @auto_reuse_variable_scope
    def next_frame_cnn(self, image, action):
        encoder_out = (LinearWrap(image)
            .Conv2D('conv0', out_channel=64, kernel_shape=8, stride=2)
            .Conv2D('conv1', out_channel=128, kernel_shape=6, stride=2)
            .Conv2D('conv2', out_channel=128, kernel_shape=6, stride=2)
            .Conv2D('conv3', out_channel=128, kernel_shape=4, stride=2, padding="VALID")())
        h = (LinearWrap(encoder_out)
            .FullyConnected('fc0', 2048, nl=tf.nn.relu)
            .FullyConnected('fc1', 2048, nl=tf.identity)())
        encoder_and_actions = tf.tensordot(h, FullyConnected('fca', tf.one_hot(action, NUM_ACTIONS), \
            2048, nl=tf.identity), [[1], [0]])
        dec_in_w = int(encoder_out.shape[1])
        dec_in_h = int(encoder_out.shape[2])
        print "Decoder input sizes", dec_in_w, dec_in_h
        decoder_in = (LinearWrap(encoder_and_actions)
            .FullyConnected('fc3', 2048, nl=tf.identity)
            .FullyConnected('fc4', dec_in_w * dec_in_h * int(encoder_out.shape[3]),
                nl=tf.nn.relu)())
        decoder_in_4d = tf.reshape(decoder_in, \
            [-1, dec_in_w, dec_in_h, int(encoder_out.shape[3])])
        decoder_out = (LinearWrap(decoder_in_4d)
            .Deconv2D('deconv1', 128, 4, stride=2, padding="VALID")
            .Deconv2D('deconv2', 128, 6, stride=2)
            .Deconv2D('deconv3', 128, 6, stride=2)
            .Deconv2D('deconv4', 3, 8, stride=2, nl=tf.identity)())
        return decoder_out

    @auto_reuse_variable_scope
    def next_frame_naff(self, image):
        l = (LinearWrap(image)
            .Conv2D('conv0', out_channel=64, kernel_shape=8)
            .Conv2D('conv1', out_channel=128, kernel_shape=6)
            .Conv2D('conv2', out_channel=128, kernel_shape=6)
            .Conv2D('conv3', out_channel=128, kernel_shape=4)
            .FullyConnected('fc0', 2048, nl=tf.nn.relu)
            .FullyConnected('fc1', 2048, nl=tf.nn.relu)
            .FullyConnected('fc1', 2048, nl=tf.nn.relu)
            .Deconv2D('deconv1', 128, 4)
            .Deconv2D('deconv2', 128, 6)
            .Deconv2D('deconv3', 128, 6)
            .Deconv2D('deconv4', 3, 8, nl=tf.identity)())
        return l

    @auto_reuse_variable_scope
    def next_frame_rnn(self, image, action):
        l = (LinearWrap(image)
            .Conv2D('conv0', out_channel=64, kernel_shape=8)
            .Conv2D('conv1', out_channel=128, kernel_shape=6)
            .Conv2D('conv2', out_channel=128, kernel_shape=6)
            .Conv2D('conv3', out_channel=128, kernel_shape=4)
            .FullyConnected('fc0', 2048, nl=tf.nn.relu)())
            # When the recurrent encoding network is trained on 1-step prediction objective, the network is unrolled
            # through 20 steps and predicts the last 10 frames by taking ground-truth images as input
        #l = 
        l = FullyConnected('fc1', l, 2048, nl=tf.identity)
        l = tf.tensordot(l, FullyConnected('fca', tf.one_hot(action, NUM_ACTIONS), 2048, nl=tf.identity))
        l = (LinearWrap(l)
            .FullyConnected('fc3', 2048, nl=tf.identity)
            .FullyConnected('fc4', 2048, nl=tf.nn.relu)
            .Deconv2D('deconv1', 128, 4)
            .Deconv2D('deconv2', 128, 6)
            .Deconv2D('deconv3', 128, 6)
            .Deconv2D('deconv4', 3, 8, nl=tf.identity)())
        return l

    def _build_graph(self, inputs):
        #k = tf.get_variable('steps', initializer=1, trainable=False)
        
        losses = []
        with argscope([Conv2D, Deconv2D], nl=tf.nn.relu, use_bias=True, stride=2):
            image, label, action = inputs
            with tf.variable_scope('A'):
                last_img = image
                for step in range(self.k):
                    nextf = self.next_frame_cnn(last_img, action[:, step]) if self.network_type == "cnn" \
                        else (self.next_frame_naff(last_img) if self.network_type == "naff" \
                        else self.next_frame_rnn(last_img, action[:, step]))
                    losses.append(tf.squared_difference(nextf, label[:, :, :, 3*step:3*step+3]))
                    # Combine the next 3 frames (which could be a combination of real and predicted)
                    # with the predicted image
                    last_img = tf.concat(last_img[:, :, :, 3:], nextf, axis=3) if self.network_type == "cnn" or \
                        self.network_type == "naff" else nextf

        '''
        viz = 255*l + self.avg
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('vizsum', viz, max_outputs=30)
        '''

        loss = tf.add_n(losses, name='total_loss')
        loss /= (2*k)
        add_moving_summary(loss)
        return loss

    def _get_optimizer(self):
        learning_rate = tf.get_variable('learning_rate', initializer=self.learning_rate, trainable=False)
        step = get_global_step_var() # get_global_step()
        learning_rate = tf.cond(step % 10000, lambda: learning_rate.assign(learning_rate * .9), lambda: learning_rate)
        return tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=tf.Constant(0.9), epsilon=0.01)

class AtariReplayDataflow(RNGDataFlow):
    def __init__(self, items, averages, model_type, shuffle=True, batch_size=32):
        def chunk(n, lst):
            n = min(n, len(lst)-1)
            return [lst[i:i+n] for i in range(len(lst) - n+1)]
        self.shuffle = shuffle
        self.items = chunk(batch_size, items)
        self.model_type = model_type
        self.avg = averages

    def size(self):
        # Return size of dataset
        return len(self.items)

    def get_data(self):
        def read_item(item):
            data_list = []
            for batch_entry in item:
                # If rnn, just use the most recent frame as input
                frame_file_list = batch_entry[0][-1:] if self.model_type == "rnn" else batch_entry[0]
                frame_list = [cv2.imread(file) for file in frame_file_list]
                assert((len(frame_list) == 1) if self.model_type == "rnn" else (len(frame_list) == 4))
                frames = preprocessImages(frame_list[0], self.avg) if self.model_type == "rnn" else \
                    preprocessImages(np.array(frame_list), self.avg).transpose(1, 2, 0, 3).reshape(210, 160, 12)

                actions = batch_entry[1]
                assert(len(actions) == 5)

                next_frame_file_list = batch_entry[2]
                next_frame_list = [cv2.resize((cv2.imread(file)), (160, 208)) for file in next_frame_file_list]
                assert(len(next_frame_list) == 5)
                next_frames = np.array(next_frame_list).transpose(1, 2, 0, 3).reshape(208, 160, 15)
                
                # 4 frames, combined across channels, 5 actions, and 5 frames (combined across channels) yielded
                # from taking those actions
                data_list.append((frames, actions, next_frames))

            return data_list

        idxs = np.arange(self.size())
        if self.shuffle:
            self.rng.shuffle(idxs)
        for idx in idxs:
            yield read_item(self.items[idx])

# Given np array of images and np vector of length 3 containing average pixel values
def preprocessImages(images, avgs):
    # subtract mean, divide by 255
    return (images - avgs)/255

def calcAvgPixel(head_directory):
    # Given list filenames, calculate average pixel values for each of the 3 channels
    pixel_sums = np.zeros((3))
    num_images = 0
    for directory, subdirs, files in os.walk(head_directory):
        if "actions.txt" not in files:
            continue
        files.remove("actions.txt")
        enough = False
        for file in files:
            image = cv2.imread(directory + '/' + file)
            pixel_sums += image.sum(axis=(0, 1))
            num_images += 1
            if num_images % 10000 == 0:
                print "Calculated average pixel values for", num_images, "images"
            if num_images >= 500000:
                enough = True
                break
        if enough:
            break

    avg_pixels = pixel_sums / float(num_images*160*210)
    return avg_pixels

def fileNum2FileName(num):
    return "frame_step_" + num.zfill(8) + ".jpg"

def readFilenamesAndActions(head_directory):
    # Returns list of tuples of the form ([list_of_4_frame_filenames], [list_of_5_actions], [list_of_5_frame_filenames])
    frames_actions_nextframes = []
    img_tot = 0
    for directory, subdirs, files in os.walk(head_directory):
        files.sort()
        if "actions.txt" not in files:
            print "Skipping subdirectory", directory, "because it does not contain an actions.txt"
            continue
        files.remove("actions.txt")
        actionfile = open(directory + "/actions.txt", 'r')
        action_lines = actionfile.read().split('\n')
        while '' in action_lines: action_lines.remove('')
        action_tuples = [line.split() for line in action_lines]
        # In the CNN case, we need m=4 frames and an action to predict a 5th frame.
        # Remove action that yields the 4th frame so the first action yields the 5th frame
        if len(files) < 9:
            print "Skipping episode", directory, "because it doesn't contain enough data."
            continue
        skip_ep = False
        enough_data = False
        for i in range(len(action_tuples)):
            if i > 2 and i < len(action_tuples) - 5:
                frames = [fileNum2FileName(action_tuples[i-3][1]), fileNum2FileName(action_tuples[i-2][1]), \
                    fileNum2FileName(action_tuples[i-1][1]), fileNum2FileName(action_tuples[i][1])]
                actions = [int(action_tuples[i+1][0]), int(action_tuples[i+2][0]), int(action_tuples[i+3][0]), \
                    int(action_tuples[i+4][0]), int(action_tuples[i+5][0])]
                next_frames = [fileNum2FileName(action_tuples[i+1][1]), fileNum2FileName(action_tuples[i+2][1]), \
                    fileNum2FileName(action_tuples[i+3][1]), fileNum2FileName(action_tuples[i+4][1]), \
                    fileNum2FileName(action_tuples[i+5][1])]
                for j in range(9):
                    if fileNum2FileName(action_tuples[i-3+j][1]) not in files:
                        print "Found action", i-j+3, "but could not find", corresponding_image, \
                            ": Skipping remainder of episode", directory
                        skip_ep = True
                if skip_ep:
                    break
                frames_actions_nextframes.append(([os.path.abspath(directory + '/' + f) for f in frames], actions, \
                    [os.path.abspath(directory + '/' + f) for f in next_frames]))
                img_tot += 1
                if img_tot % 10000 == 0:
                    print "Read in filenames and actions for", img_tot, "training examples"
                if img_tot >= 1000:
                    enough_data = True
                    break
        if skip_ep:
            continue
        if enough_data:
            break
    print "Read in a total of", img_tot, "training examples."
    return frames_actions_nextframes
