#!/usr/bin/python

import tensorflow as tf
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

def conv(input, kernel, depth, in_channels, stride, h_pad, v_pad):
    # UNKNOWN: Initialization range?
    conv_filters = tf.Variable(tf.truncated_normal([kernel, kernel, in_channels, depth], stddev=0.1))
    conv_filters_bias = tf.Variable(tf.truncated_normal([depth], stddev=0.1))
    # UNKNOWN: PADDING - horizontal different from vertical in TensorFlow/Pack?
    pad = ""
    if h_pad == 0 and v_pad == 0:
        pad = "VALID"
    else:
        pad = "SAME"
    conv = tf.nn.conv2d(input, conv_filters, [1, stride, stride, 1], pad)
    return tf.nn.bias_add(conv, conv_filters_bias)

def deconv(inpt, kernel, depth, out_channels, stride, h_pad, v_pad):
    # UNKNOWN: Initialization
    deconv_filters = tf.Variable(tf.truncated_normal([kernel, kernel, out_channels, depth], stddev=0.1))
    # TODO(Aaron): Bias, but batch size is unknown. Same with bias in forward4d
    # deconv_filters_bias = tf.Variable(tf.truncated_normal([depth], stddev=0.1))
    input_minus_b = inpt#tf.subtract(inpt, deconv_filters_bias)

    # output shape parameter
    # UNKNOWN: PADDING
    pad = ""
    if h_pad == 0 and v_pad == 0:
        pad = "VALID"
    else:
        pad = "SAME"
    # TODO(Aaron): I haven't had time to determine what output shape of deconvolution should be.
    # First dimension will probably be batch size/unknown, not sure how to handle that
    # Last dimension will probably be out_channels..
    output_shape = []
    deconv = tf.nn.conv2d_transpose(input_minus_b, deconv_filters, output_shape, strides=[1, stride, stride, 1], padding=pad)
    # https://stackoverflow.com/questions/43113984/output-shape-of-tf-nn-conv2d-transpose-is-entirely-undefined-even-though-only-ba
    # tf.reshape()
    return deconv

def forward4d(inpt, hidden_size, init):
    W = tf.Variable(tf.random_uniform([int(inpt.get_shape()[3]), hidden_size], minval=-1*init, maxval=init))
    td = tf.tensordot(inpt, W, [[3], [0]])
    return td
    # TODO(Aaron): Though they didn't explicitly mention bias, it should be added.. also, this doesn't work because batch size is unknown
    # b = tf.Variable(tf.random_uniform([-1, int(inpt.get_shape()[1]), int(inpt.get_shape()[2]), hidden_size], minval=-1*init, maxval=init))
    # return tf.add(td, b)

def forward(inpt, shape, init):
    W = tf.Variable(tf.random_uniform(shape, minval=-1*init, maxval=init))
    b = tf.Variable(tf.random_uniform([shape[1]], minval=-1*init, maxval=init))
    return tf.add(tf.matmul(inpt, W), b)

class ACVPnet:
    def __init__(self, model_type, pixel_avgs):
        self.session = tf.Session()
        self.model_type = model_type
        self.pixel_avgs = pixel_avgs

        self.networkPlaceholders()
        self.networkEncoder()
        self.decoder_in = None

        # TODO(Aaron): Get these working and uncomment decoder and loss
        if model_type == "cnn":
            self.networkTransformationCNN()
        elif model_type == "rnn":
            self.networkTransformationRNN()
        elif model_type == "naff":
            self.networkTransformationNAFF()
        else:
            sys.exit("Unknown model type provided.")

        # self.networkDecoder()
        # self.networkLoss()

    def loadModel(self, filename, pixel_avgs):
        # TODO(Aaron, Matt, Ben): After the model is saved correctly, test loading. Make a new script that does this,
        # and write a function to predict the next frame given a frame history (4 or 1, depending on network) and action
        
        pass
        # Load the model given model filename and np vector of length 3 containing average pixel values

    def train(self, filename_lists, action_lists, model_filename, model_path):
        """
        filename_lists: list containing lists of filenames, where each list represents a single episode
        action_lists: list containing lists of actions. If the corresponding list of filenames contains
        [frame_step_00000001.jpg, 2.jpg, 3.jpg, ...], the action list for this episode begins with the action
        taken in frame 4 that results in frame 5

        model_filename and model_path are implemented in the commented-out model_saving - see below
        """

        # TODO(Aaron): Save the model parameters, and possibly checkpoint after each epoch. Should be able to uncomment
        # this and self.saver calls below, but I'm not sure
        # self.saver = tf.train.Saver()
        
        # Three training phases, as described in paper:
        learning_rates = [1e-4, 1e-5, 1e-5]
        # iterations as stated in paper = [1.5e6, 1e6, 1e6] - Divide by ~5e5 examples, 3 epochs, 2 epochs, 2 epochs
        epochs = [3, 2, 2]
        # CNN and Naff batch sizes
        batch_size_cnn = [32, 8, 8]
        batch_size_rnn = [4, 4, 4]

        '''
        # TODO(Aaron): Note about steps (aka k) and loss:
        Steps are used in loss function. In phase 1, compute loss over 1 predicted frame.
        
        In phase 2, use each 210x160x12 frame history to predict next frame (as usual). Then use that frame and
        history[1:] (210x160x9) to predict subsequent frame 2. Then use 2 predicted frames and history[2:] to predict
        frame 3. Calculate and sum loss over all 3 images.
        
        Phase 3 is similar, though loss is calculated over 5 predicted frames rather than 3.
        
        Note that this isn't necessarily possible at the end of a batch unless we pass in additional data, and
        certainly isn't possible at the end of an episode.

        '''
        steps = [1, 3, 5]

        # Training loop
        self.session.run(tf.global_variables_initializer())
        for phase in range(3):
            learning_rate = learning_rates[phase]
            # Used to determine learning rate, as it's multiplied by 0.9 every time 10,000 images are processed
            num_images_processed = 0
            # TODO(Aaron): Pass k through to loss calculation
            k = steps[phase]
            for epoch in range(epochs[phase]):
                # Iterate over all episodes
                for i in range(len(filename_lists)):
                    # Images for this episode
                    avg_images_episode = preprocessImages(filenamesToImages(filename_lists[i]), self.pixel_avgs)
                    if avg_images_episode.shape[0] < 5:
                        print "This episode has less than 5 images, skipping.", filename_lists[i]
                        continue
                    '''
                    TODO(Aaron): if self.model_type == "cnn" or self.model_type == "naff", concatenate 4 images into
                    210x160x12 frame histories, which is done below.
                    else: rnn, so just pass 210x160x3 images - add another branch that skips this concatenation.
                    all you probably need to do is use avg_images_episode[:-1] as train_input
                    '''
                    # Read in images for the current episode
                    train_input = []
                    for index in range(avg_images_episode.shape[0]-4):
                        # Slice of 4 210x160x3 images
                        frame_hist = avg_images_episode[index:index+4]
                        # Turn into 210x160x12 image
                        hist_combined_in_channels = frame_hist.transpose(1, 2, 0, 3).reshape(210, 160, 12)
                        train_input.append(hist_combined_in_channels)

                    # These should be the same for CNN and RNN
                    train_label = avg_images_episode[4:]
                    train_action = action_lists[i]
                    if len(train_input) != len(train_action) or train_label.shape[0] != len(train_input):
                        print "Number of training examples don't match up:", len(train_input), "input images,", \
                            train_label.shape[0], "output images,", len(train_action), "actions. Skipping."
                        continue
                    
                    # TODO(Aaron, Matt): Split this data into batch_size chunks
                    # TODO(Aaron): Pass in variable learning rate somehow
                    # num_images_processed += _
                    # multiply learning rate by 0.9 every 1e5 iterations
                    self.LR = learning_rates[phase] * np.power(0.9, np.floor(num_images_processed/1e5))
                    
                    FD = {self.frames: train_input, self.next_frame: train_label, self.actions: train_action}
                    return_vals = [self.h]
                    # TODO(Aaron): Trying to access self.enc_ac at runtime produces an error, but self.h does not
                    # return_vals = [self.enc_ac]
                    # InvalidArgumentError Matrix size-incompatible: In[0]: [972,2048], In[1]: [9,2048]
                    [thing] = self.session.run(return_vals, feed_dict=FD)
                    print thing.shape

                # Save a copy of the model (checkpoint) after each epoch
                # save_path = saver.save(session, model_path + "/" + str(phase) + '_' + str(epoch) + '_' + model_filename)
        # Save at the end of training
        # save_path = saver.save(session, model_path + "/final_" + model_filename)
        self.session.close()

    def networkPlaceholders(self):
        # network model

        # batch_size vector of actions, which will be converted to one-hot when needed
        self.actions = tf.placeholder(tf.int32, shape=[None])
        # Batch size by image size by channel size (3-color*4 images for CNN, 3-color for RNN)
        # Note cv2 image shape: 
        self.frames = tf.placeholder(tf.float32, shape=[None, 210, 160, None])
        self.next_frame = tf.placeholder(tf.float32, shape=[None, 210, 160, 3])

    # Used in next step since channel_size varies between model types
    def conv1ChannelSize(self, channel_size):
        return conv(self.frames, 8, 64, channel_size, 2, 0, 1)

    def networkEncoder(self):
        # encoder
        c1 = None
        if self.model_type == "cnn" or self.model_type == "naff":
            c1 = self.conv1ChannelSize(12)
        else:
            # 3 channel images for RNN
            c1 = self.conv1ChannelSize(3)
        r1 = tf.nn.relu(c1)
        c2 = conv(r1, 6, 128, 64, 2, 1, 1)
        r2 = tf.nn.relu(c2)
        c3 = conv(r2, 6, 128, 128, 2, 1, 1)
        r3 = tf.nn.relu(c3)
        c4 = conv(r3, 4, 128, 128, 2, 0, 0)
        self.enc_out = tf.nn.relu(c4)

    def networkTransformationCNN(self):

        # Batch size by NUM_ACTIONS
        onehot_actions = tf.one_hot(self.actions, NUM_ACTIONS)
        a = forward(onehot_actions, [NUM_ACTIONS, 2048], 0.1)
        print "Act", self.actions.get_shape(), onehot_actions.get_shape(), a.get_shape()

        ff1 = forward4d(self.enc_out, 2048, 1.0)
        relu1 = tf.nn.relu(ff1)
        # TODO: Selfs here are just for testing
        self.h = forward4d(relu1, 2048, 1.0)
        # TODO(Aaron): This works in the model definition, but produces a shape error at runtime. See self.train
        self.enc_ac = tf.tensordot(self.h, a, [[3], [0]])

        ff2 = forward4d(self.enc_ac, 2048, 1.0)
        ff3 = forward4d(ff2, 2048, 1.0)
        self.decoder_in = tf.nn.relu(ff3)
        
        print "Decoder shape", self.decoder_in.get_shape()

    def networkTransformationRNN(self):
        # TODO(RNN)
        # Clip gradients at [-0.1, 0.1]
        # When the recurrent encoding network is trained on 1-step prediction objective, the network is unrolled
        # through 20 steps and predicts the last 10 frames by taking ground-truth images as input

        # Very similar to networkTransformationCNN, LSTM between relu1 and h. See figure 9.
        pass

    def networkTransformationNAFF(self):
        ff1 = forward4d(self.enc_out, 2048, 1.0)
        relu1 = tf.nn.relu(ff1)
        ff2 = forward4d(relu1, 2048, 1.0)
        relu2 = tf.nn.relu(ff2)
        ff3 = forward4d(relu2, 2048, 1.0)
        self.decoder_in = tf.nn.relu(ff3)

    def networkDecoder(self):
        # decoder
        dc1 = deconv(self.decoder_in, 4, 2048, 128, 2, 0, 0)
        r1 = tf.nn.relu(dc1)
        print "R1", r1.get_shape()
        dc2 = deconv(r1, 6, _, 128, 2, 1, 1)
        r2 = tf.nn.relu(dc2)
        dc3 = deconv(r2, 6, _, 128, 2, 1, 1)
        r3 = tf.nn.relu(dc3)
        self.output_image = deconv(r3, 8, _, 3, 2, 0, 1)

    def networkLoss(self):

        """
        Paper notes:
        Given T frame-action pairs,
        loss: average squared error over K-step predictions
        1/(2K) sum over i, t, k (from 1 to K)
        "unrolled" through K time-steps
        RMSProp [34, 10] is used with momentum of 0.9, (squared) gradient momentum of 0.95,
        and min squared gradient of 0.01
        """
        
        # TODO(Aaron): Pass variable K here. 1, 3, or 5 depending on phase - see self.train
        K = None
        last_predicition = self.output_image

        # TODO(Aaron): As described in comments of self.train, we need to use the network to predict K future images,
        # and calculate loss on each of them.
        # TODO(Aaron): Future actions and labels could be in the next batch. See notes in self.train
        future_action = None
        future_prediction = None
        # TODO(Aaron): Calculate loss as described in self.train. Requires generating K predictions through the network
        err_sum = 0
        for k in range(K):
            prediction_k = None # Function of future_action and frame history/old predictions - run through the network
            squared_pixel_error = np.power(tf.norm(prediction_k - future_prediction), 2)
            err_sum += squared_pixel_error
        self.loss = err_sum / (2*K)

        # TODO(Aaron): Pass variable learning rate, maybe through feed_dict?
        learning_rate = None
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=tf.Constant(0.9), epsilon=0.01)
        self.train_op = optimizer.minimize(self.loss)

# Given np array of images and np vector of length 3 containing average pixel values
def preprocessImages(images, avgs):
    # subtract mean, divide by 255
    return (images - avgs)/255

def calcAvgPixel(frame_lists):
    # Given list of lists of filenames, calculate average pixel values for each of the 3 channels
    pixel_sums = np.zeros((3))
    num_pixels = 0
    for frame_list in frame_lists:
        images = filenamesToImages(frame_list)
        pixel_sums += images.sum(axis=(0, 1, 2))
        num_pixels += images.shape[0] * images.shape[1] * images.shape[2]
    avg_pixels = pixel_sums / float(num_pixels)
    print "Average pixel values across all 3 channels:", avg_pixels
    return avg_pixels

# Given list of filenames, return np array of images
def filenamesToImages(filenames):
    images = []
    for imgfile in filenames:
        img = cv2.imread(imgfile)
        if img is None:
            print "Unable to read", imgfile
            sys.exit()
        images.append(img)
    return np.array(images)

def readFilenamesAndActions(directories):
    # One list is added for each episode
    frame_lists = []
    action_lists = []
    img_tot = 0
    for head_episode_dir in directories:
        for directory, subdirs, files in os.walk(head_episode_dir):
            if directory != head_episode_dir:
                files.sort()
                if "actions.txt" not in files:
                    print "Skipping subdirectory", directory, "because it does not contain an actions.txt"
                    continue
                files.remove("actions.txt")
                actionfile = open(directory + "/actions.txt", 'r')
                action_lines = actionfile.read().split('\n')
                while '' in action_lines: action_lines.remove('')
                action_tuples = [line.split() for line in action_lines]
                if action_tuples[0][1] != '4':
                    print "First action taken does not yield frame 4. Skipping episode", directory
                    continue
                # In the CNN case, we need m=4 frames and an action to predict a 5th frame.
                # Remove action that yields the 4th frame so the first action yields the 5th frame
                action_tuples.pop(0)
                if len(files) < 5:
                    print "Skipping episode", directory, "because it doesn't contain enough data."
                    continue
                if files[:4] != ["frame_step_00000001.jpg", "frame_step_00000002.jpg", "frame_step_00000003.jpg", \
                    "frame_step_00000004.jpg"]:
                    print "First four frames not found. Skipping episode", directory
                    continue
                skip_ep = False
                for tup in action_tuples:
                    corresponding_image = "frame_step_" + tup[1].zfill(8) + ".jpg"
                    if corresponding_image not in files:
                        print "Found action", tup[1], "but could not find", corresponding_image, \
                            ": Skipping episode", directory
                        skip_ep = True
                        break
                if skip_ep:
                    continue
                absfiles = [os.path.abspath(directory + '/' + f) for f in files]
                only_actions = [int(tup[0]) for tup in action_tuples]
                # Ensure that each image has a corresponding action, except the first 4
                if len(absfiles) != len(only_actions) + 4:
                    print "The number of images and number of actions don't match up. Skipping episode", \
                        directory
                    continue
                frame_lists.append(absfiles)
                action_lists.append(only_actions)
                img_tot += len(absfiles)
    print "Read in a total of", img_tot, "images."
    if len(frame_lists) != len(action_lists):
        sys.exit("Somehow, the number of episodes are not equal between frames and actions.")
    return (frame_lists, action_lists)
