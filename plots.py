#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt 
from math import sqrt

x = np.arange(0,100)
rand_scores = np.loadtxt('scores/rand_scores.out')
mean_score = np.mean(rand_scores)
std_error = np.std(rand_scores)/sqrt(rand_scores.size)
rand = mean_score*np.ones(x.size)

em_scores = np.loadtxt('scores/em_scores.out')
em_error = np.std(em_scores)/sqrt(em_scores.size)
plt.plot(x, np.ones(x.size)*np.mean(em_scores),color='black', label='Emulator')
plt.fill_between(x, np.mean(em_scores)*np.ones(x.size) - em_error, np.mean(em_scores)*np.ones(x.size) + em_error,
	alpha = 0.5, edgecolor='#000000', facecolor='#000000')

plt.plot(x, np.ones(x.size)*rand, 'k', color='#CC4F1B', label='Random')
plt.fill_between(x, rand- std_error, rand+std_error,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')


rand_actions = np.loadtxt('scores/rand_actions.out')
x = np.arange(8, 100, 8)
x = np.insert(x, 0 , np.array([4]))


rand_mean = np.mean(rand_actions, axis=1)
rand_std = np.std(rand_actions, axis=1)/sqrt(14)

plt.plot(x, rand_mean, color= 'blue', label='Real action every k frames')
plt.fill_between(x, rand_mean + rand_std, rand_mean - rand_std, alpha = 0.5, edgecolor='blue', facecolor='blue')



# l = plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0.)
l = plt.legend()
plt.xlabel('Frames since last emulator frame (k)')
plt.ylabel('Mean Score from 30 Episodes')
plt.show()