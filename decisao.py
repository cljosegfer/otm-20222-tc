#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:07:01 2022

@author: jose
"""

from pareto import rea
import numpy as np
import matplotlib.pyplot as plt

num = 100
experimento = rea()
export, border = experimento.pareto(num)

# decision
tau = 1.25
penalizado = np.copy(border)
penalizado[:, 1] *= tau # penalizando custo de falha total

soma = np.sum(penalizado, axis = 1)
plt.plot(soma)
decisao = export.iloc[np.argmin(soma)].values
_, counts = np.unique(decisao, return_counts = True)

# alpha, gama
alpha = counts[0] / len(decisao)
if 2 in _:
    gama = counts[1] / len(decisao)
