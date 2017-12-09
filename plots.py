#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt 


x = np.arange(0,30)
y = np.load('scores.out')
plt.plot(x,y)