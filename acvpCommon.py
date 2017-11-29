#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import os
import random
import time
import multiprocessing
from tqdm import tqdm
from six.moves import queue

from tensorpack.utils.concurrency import StoppableThread, ShareSessionThread
from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs
# import from Matt as acvp_net


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

    ob = env.reset()
    sum_r = 0
    while True:
        act = predict(ob)
        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        if isOver:
            return sum_r

def save_one_episode(env, func, k, render=False):
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
    sum_r = 0
    while True:
        act = predict(ob)
        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        if isOver:
            # env.env.env.env.step_count += 1000
            return sum_r

# def acvplay(env, func, acvp, pred_steps, arch, render=False):
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
#     buffer = []
#     buffer.append(env.env.env.env._grab_raw_image())
#     k = 0
#     while True:
#         act = predict(ob)
#         ob, r, isOver, info = env.step(act)
#         if arch == 'cnn':
#             buffer.append(env.env.env.env._grab_raw_image())
#             if len(buffer) >= 4:
#                 buffer.pop(0)
#         if (k % pred_steps == 0):
#             if arch == cnn:
#                 ob = acvp(buffer,action)
#             else:
#                 ob = acvp(env.env.env.env._grab_raw_image())
#         if render:
#             env.render()
#         sum_r += r
#         k = k + 1
#         if isOver:
#             return sum_r
        

def play_n_episodes(player, predfunc, nr, render=False):
    logger.info("Start Playing ... ")
    for k in range(nr):
        score = play_one_episode(player, predfunc, render=render)
        print("{}/{}, score={}".format(k, nr, score))

def play_save_n_episodes(player, predfunc, nr, render=False):
    logger.info("Start Playing, and saving! ... ")
    for k in range(nr):
        dir = '/data/people/babbatem/frames/' + 'ep' + str(k)
 #       dir = '/Users/abba/projects/acvp/acvp/frames/' + 'ep' + str(k)
        os.makedirs(dir)
        player.env.env.env.save_dir = dir
        player.env.env.env.action_file = open(dir + '/actions.txt', 'w')
#        player.env.env.env.step_count = 0
        score = save_one_episode(player, predfunc, k, render=render)
        print("{}/{}, score={}".format(k, nr, score))

def plot_episodes(players, predfunc, nr, arch, render=False):
    logger.info("Generating data for plots")
    blind_steps = np.arange(1, 100, 7)
    blind_steps = np.insert(blind_steps, 1, np.array([2,3,4,5,6,7]))
    # network = acvp_net(arch)
    for i in blind_steps:
        for j in range(nr):
            score = acvplay(player, predfunc, network, i, arch, render=render)
            # write to file
            print("predictive steps {}/{}, repetition {}/{}, score={}".format(i, blind_steps.size, j, nr, score))

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
