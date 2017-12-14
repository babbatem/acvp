#!/usr/bin/env python

import os
import random
import time
import multiprocessing
import cv2
import threading
import six
from tqdm import tqdm
from six.moves import queue
import numpy as np

from collections import deque
from tensorpack.utils.concurrency import StoppableThread, ShareSessionThread
from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs
from frame_pred.image_generator import ACVPGenerator


def play_one_episode(env, func, render=False):
    def predict(s):
        """
        Map from observation to action, with 0.001 greedy.
        """
        act = func(s[None, :, :, :])[0][0].argmax()
        if random.random() < 0.001:
            spc = env.action_space
            act = spc.sample()
        return act

    spc = env.action_space
    ob = env.reset()
    sum_r = 0
    while True:
        act = predict(ob)
        # act = spc.sample()
        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        if isOver:
            return sum_r

def acvplay(env, func, acvp, pred_steps, arch, render=False):
    def predict(s):
        """
        Map from observation to action, with 0.001 greedy.
        """
        act = func(s[None, :, :, :])[0][0].argmax()
        if random.random() < 0.001:
            spc = env.action_space
            act = spc.sample()
        return act

    ob = env.reset()
    spc = env.action_space
    frame_0 = env.env.env.env._grab_raw_image()
    sum_r = 0
    pred_buffer = deque([],maxlen=4)
    ob_buffer = deque([],maxlen=4)
    for i in np.arange(4):
        pred_buffer.append(frame_0)

    pred_state = np.concatenate(pred_buffer, axis=2)

    k = 0
    while True:
        act = predict(ob)
        ob, r, isOver, info = env.step(act)
        # print 'real ob shape', ob.shape

        if (k % pred_steps == 0):
            # print('real_frame')
            last_four = info['last_four']
            pred_state = last_four
            # print(pred_state.shape)
        else: 
            # print('pred')
            for i in range(4):
                prediction = acvp([pred_state], np.array([[act]]))
                grey_predict = cv2.cvtColor(prediction[0][0,:210,:,:], cv2.COLOR_RGB2GRAY)
                # print 'grey_shape:', grey_predict.shape
                ob_buffer.append(cv2.resize(grey_predict, (84,84)))
                pred_buffer.append(prediction[0][0,:210,:,:])
                pred_state = np.concatenate(pred_buffer,axis=2)
                # print(pred_state.shape)
                # cv2.imwrite('pred_frames/test_img.jpg', prediction[0][0,:210,:,:])
            ob = np.array(ob_buffer)
            ob = ob.transpose(1,2,0)
            # print 'pred ob shape', ob.shape

        if render:
            env.render()
        
        sum_r += r
        k = k + 4
        if isOver:
            return sum_r
        
def rand_play(env,func,pred_steps,render=False):
    def predict(s):
        """
        Map from observation to action, with 0.001 greedy.
        """
        act = func(s[None, :, :, :])[0][0].argmax()
        if random.random() < 0.001:
            spc = env.action_space
            act = spc.sample()
        return act

    spc = env.action_space
    ob = env.reset()
    sum_r = 0
    k = 0
    while True:
        if (k % pred_steps == 0):
            # every pred_steps frames, take a real action
            act = predict(ob)
        else:
            # otherwise, act random
            act = spc.sample()

        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        k += 4
        if isOver:
            return sum_r

def play_n_episodes(player, predfunc, nr, render=False):
    logger.info("Start Playing ... ")
    score = np.zeros(nr)
    for k in range(nr):
	    score[k] = play_one_episode(player, predfunc, render=render)
	    print("{}/{}, score={}".format(k, nr, score))
    np.savetxt('/Users/abba/projects/acvp/acvp/em_scores.out', score)

def play_save_n_episodes(player, predfunc, nr, render=False):
	logger.info("Start Playing, and saving! ... ")
	score = np.zeros(nr)
	for k in range(nr):
		dir = '/data/people/babbatem/dataset33/' + 'ep' + str(k).zfill(3)
		# dir = '/Users/abba/projects/acvp/acvp/frames/' + 'ep' + str(k).zfill(3)
		os.makedirs(dir)
		player.env.env.env.save_dir = dir
		player.env.env.env.action_file = open(dir + '/actions.txt', 'w')
		player.env.env.env.step_count = 0
		score = play_one_episode(player, predfunc, render=render)
		print("{}/{}, score={}".format(k, nr, score))

def plot_episodes(player, predfunc, nr, arch, render=False):
    logger.info("Generating data for plots")
    blind_steps = np.arange(8, 100, 8)
    blind_steps = np.insert(blind_steps, 0, np.array([4]))
    # network = ACVPGenerator('frame_pred/2017_12_12_16_05_15_9727706train_log/model-499993',arch)
    # network = ACVPGenerator('frame_pred/2017_12_12_17_41_27_5032412train_log/model-499985',arch)
    network = ACVPGenerator('2017_12_13_12_53_11_7692214train_log/model-542062',arch)

    # network = ACVPGenerator('frame_pred/model-499993',arch)
    scores = np.zeros(blind_steps.size*nr).reshape((blind_steps.size,nr))
    k = 0
    for i in blind_steps:
        for j in range(nr):
            scores[k][j] = acvplay(player, predfunc, network.predictor, i, arch, render=render)
            # scores[k][j] = rand_play(player, predfunc, i, render=render)
            print("predictive steps {}/{}, repetition {}/{}, score={}".format(k, blind_steps.size, j, nr, scores))
        k += 1
    np.savetxt('scores/LG_phase1.out', scores)


def eval_with_funcs(predictors, nr_eval, get_player_fn):
    """
    Args:
        predictors ([PredictorBase])
    """
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(train=False)
                while not self.stopped():
                    try:
                        score = play_one_episode(player, self.func)
                        # print("Score, ", score)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()

    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        r = q.get()
        stat.feed(r)
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        r = q.get()
        stat.feed(r)

    if stat.count > 0:
        return (stat.average, stat.max)
    return (0, 0)


def eval_model_multithread(pred, nr_eval, get_player_fn):
    """
    Args:
        pred (OfflinePredictor): state -> Qvalue
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    with pred.sess.as_default():
        mean, max = eval_with_funcs([pred] * NR_PROC, nr_eval, get_player_fn)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))


class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean)
        self.trainer.monitors.put_scalar('max_score', max)
