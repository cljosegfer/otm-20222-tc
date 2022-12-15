#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:06:36 2022

@author: jose
"""

from pareto import rea
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

log = []
# num = 10
nums = np.arange(10, 50 + 1)
for num in tqdm(nums):
    experimento = rea()
    export, border, sol = experimento.pareto(num)
    # export, border, sol = experimento.pareto_naive(num)
    log.append([sol, len(border)])
log = np.array(log)

# plot
plt.figure()
plt.plot(nums, log[:, 0])
plt.plot(nums, log[:, 1], label = 'pareto')
plt.legend()
plt.xticks([])
