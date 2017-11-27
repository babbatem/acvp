#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np

import os
import sys
import re
import time
import random
import argparse
import subprocess
import multiprocessing
import threading
from collections import deque
import cv2
import gym
from gym import spaces

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import *
from tensorpack.utils.concurrency import *
import tensorflow as tf

from DQNModel import Model as DQNModel
from acvpCommon import Evaluator, eval_model_multithread, play_n_episodes, play_save_n_episodes
from atari_wrapper import FrameStack, MapState, FireResetEnv
from expreplay import ExpReplay
from atari import AtariPlayer

BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
#IMAGE_SIZE = (210, 160, 3)
FRAME_HISTORY = 4
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 1e6
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10  # each epoch is 100k played frames
EVAL_EPISODE = 50

NUM_ACTIONS = None
ROM_FILE = None
METHOD = None


# class AtariPlayerFull(AtariPlayer):

#     def __init__(self, env, **kwargs):
#         super(AtariPlayerFull,self).__init__(env, **kwargs)

#     def _current_state(self):
#         """
#         :returns: a gray-scale (h, w) uint8 image
#         """
#         ret = self._grab_raw_image()
#         # max-pooled over the last screen
#         ret = np.maximum(ret, self.last_raw_screen)
#         if self.viz:
#             if isinstance(self.viz, float):
#                 cv2.imshow(self.windowname, ret)
#                 time.sleep(self.viz)

#         cv2.imshow(self.windowname, ret)
#         ret = ret.astype('float32')
#         # 0.299,0.587.0.114. same as rgb2y in torch/image
#         ret_color = ret
#         ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
#         return ret.astype('uint8')#, ret_color.astype('uint8') # to save some memory

#     def _step(self, act):
#         oldlives = self.ale.lives()
#         r = 0
#         for k in range(self.frame_skip):
#             if k == self.frame_skip - 1:
#                 self.last_raw_screen = self._grab_raw_image()
#             r += self.ale.act(self.actions[act])
#             newlives = self.ale.lives()
#             if self.ale.game_over() or \
#                     (self.live_lost_as_eoe and newlives < oldlives):
#                 break

#         self.current_episode_score.feed(r)
#         trueIsOver = isOver = self.ale.game_over()
#         if self.live_lost_as_eoe:
#             isOver = isOver or newlives < oldlives

#         info = {'score': self.current_episode_score.sum, 'gameOver': trueIsOver}
#         return self._current_state(), r, isOver, info


# class MapState(gym.ObservationWrapper):
#     def __init__(self, env, map_func):
#         gym.ObservationWrapper.__init__(self, env)
#         self._func = map_func


#     def _observation(self, obs):
#         return self._func(obs)


# class FrameStack(gym.Wrapper):
#     def __init__(self, env, k):
#         """Buffer observations and stack across channels (last axis)."""
#         gym.Wrapper.__init__(self, env)
#         self.k = k
#         self.frames = deque([], maxlen=k)
#         shp = env.observation_space.shape
#         chan = 1 if len(shp) == 2 else shp[2]
#         self._base_dim = len(shp)
#         self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], chan * k))

#     def _reset(self):
#         """Clear buffer and re-fill by duplicating the first observation."""
#         ob = self.env.reset()
#         for _ in range(self.k - 1):
#             self.frames.append(np.zeros_like(ob))
#         self.frames.append(ob)
#         return self._observation()

#     def _step(self, action):
#         ob, reward, done, info = self.env.step(action)
#         self.frames.append(ob)
#         return self._observation(), reward, done, info

#     def _observation(self):
#         assert len(self.frames) == self.k
#         if self._base_dim == 2:
#             return np.stack(self.frames, axis=-1)
#         else:
#             return np.concatenate(self.frames, axis=2)

# def save_one_episode(env, func, render=False):
#     def predict(s):
#         """
#         Map from observation to action, with 0.001 greedy.
#         """
#         act = func(s[None, :, :, :])[0][0].argmax()
#         if random.random() < 0.001:
#             spc = env.action_space
#             act = spc.sample()
#         return act


#     ob = env.reset()
#     ob_rgb = env.env.env.env._grab_raw_image()
#     sum_r = 0
#     while True:
#         act = predict(ob)
#         ob, r, isOver, info = env.step(act)
#         #ob_rgb = env.env.env.env.last_raw_screen
#         #assert(ob_rgb)
#         #print(ob_rgb)
#         #cv2.imshow('test', ob_rgb)
#         if render:
#             env.render()
#         sum_r += r
#         if isOver:
#             return sum_r

# def play_one_episode(env, func, render=True):
#     def predict(s):
#         """
#         Map from observation to action, with 0.001 greedy.
#         """
#         act = func(s[None, :, :, :])[0][0].argmax()
#         if random.random() < 0.001:
#             spc = env.action_space
#             act = spc.sample()
#         return act

#     ob = env.reset()
#     sum_r = 0
#     while True:
#         act = predict(ob)
#         ob, r, isOver, info = env.step(act)
#         if render:
#             env.env.env.env.env.render()
#         sum_r += r
#         if isOver:
#             return sum_r

# def play_save_n_episodes(player, predfunc, nr, render=False):
#     logger.info("Start Playing, and saving! ... ")
#     for k in range(nr):
#         score = save_one_episode(player, predfunc, render=render)
#         print("{}/{}, score={}".format(k, nr, score))


# def play_n_episodes(player, predfunc, nr, render=True):
#     logger.info("Start Playing ... ")
#     for k in range(nr):
#         score = play_one_episode(player, predfunc, render=render)
#         print("{}/{}, score={}".format(k, nr, score))

def get_player(viz=False, train=False, save=False):
    play = AtariPlayer(ROM_FILE, frame_skip=ACTION_REPEAT, viz=viz,
                      live_lost_as_eoe=train, max_num_frames=30000)
    
    if save:
        dir = '/data/people/babbatem/' + str(time.time()) # change username as needed
#        dir = '/Users/abba/projects/acvp/acvp/frames'
        os.makedirs(dir)
        play.save_dir = dir
        play.save_flag = True
        play.action_file = open(dir + '/actions.txt', 'w')

    env = FireResetEnv(play)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    if not train:
        # in training, history is taken care of in expreplay buffer
        env = FrameStack(env, FRAME_HISTORY)
    return env

class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv2D, nl=PReLU.symbolic_function, use_bias=True), \
                argscope(LeakyReLU, alpha=0.01):
            l = (LinearWrap(image)
                 # Nature architecture
                 .Conv2D('conv0', out_channel=32, kernel_shape=8, stride=4)
                 .Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                 .Conv2D('conv2', out_channel=64, kernel_shape=3)

                 # architecture used for the figure in the README, slower but takes fewer iterations to converge
                 # .Conv2D('conv0', out_channel=32, kernel_shape=5)
                 # .MaxPooling('pool0', 2)
                 # .Conv2D('conv1', out_channel=32, kernel_shape=5)
                 # .MaxPooling('pool1', 2)
                 # .Conv2D('conv2', out_channel=64, kernel_shape=4)
                 # .MaxPooling('pool2', 2)
                 # .Conv2D('conv3', out_channel=64, kernel_shape=3)

                 .FullyConnected('fc0', 512, nl=LeakyReLU)())
        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, self.num_actions, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')


def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(train=True),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )

    return TrainConfig(
        data=QueueInput(expreplay),
        model=Model(),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=10000 // UPDATE_FREQ),    # update target network every 10k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (10, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
                interp='linear'),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['Qvalue'], get_player),
                every_k_epochs=10),
            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train','acvp'], default='train')
    parser.add_argument('--rom', help='atari rom', required=True)
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ROM_FILE = args.rom
    METHOD = args.algo
    # set num_actions
    NUM_ACTIONS = AtariPlayer(ROM_FILE).action_space.n
    logger.info("ROM: {}, Num Actions: {}".format(ROM_FILE, NUM_ACTIONS))

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue']))
        if args.task == 'play':
            play_n_episodes(get_player(viz=0.01), pred, 100)
        if args.task == 'acvp':
	        play_save_n_episodes(get_player(viz=0,save=True), pred, 100)
        elif args.task == 'eval':
            eval_model_multithread(pred, EVAL_EPISODE, get_player)
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'DQN-{}'.format(
                os.path.basename(ROM_FILE).split('.')[0])))
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
