#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:59:04 2022

@author: jose
"""

import pandas as pd
import numpy as np
from silh import silh
import matplotlib.pyplot as plt

def M(x):
    # _, count = np.unique(x, return_counts = True)
    # return np.sum(count * mpdb['m'].values)
    # ms = [m_rule[manutencao] for manutencao in x]
    # return np.sum(ms)
    return np.sum(x-1)

def weibull(t, eta, beta):
    return 1 - np.exp(-(t / eta) ** beta)

def prob(t_0, k, eta, beta, priori, deltat = 5):
    # priori = weibull(t_0, eta, beta) # this is constant
    return (weibull(t_0 + k * deltat, eta, beta) - priori) / (1 - priori)

def F(x):
    # k = [k_rule[manutencao] for manutencao in x]
    k = [(-x_i + 5) / 2 for x_i in x]
    p = [prob(t_0i, k_i, eta_i, beta_i, priori_i) for (t_0i, k_i, eta_i, beta_i, priori_i) in zip(t_0, k, eta, beta, priori)]
    return np.sum(p * f)

# read
equipdb = pd.read_csv('tc/EquipDB.csv', header = None, 
                      names = ['id', 't_0', 'cluster', 'f'])
mpdb = pd.read_csv('tc/MPDB.csv', header = None, 
                   names = ['id', 'k', 'm'])
clusterdb = pd.read_csv('tc/ClusterDB.csv', header = None, 
                        names = ['id', 'eta', 'beta'])

# dict
# m_rule = dict(enumerate(mpdb['m'], start = 1))
eta_rule = dict(enumerate(clusterdb['eta'], start = 1))
beta_rule = dict(enumerate(clusterdb['beta'], start = 1))
# k_rule = dict(enumerate(mpdb['k'], start = 1))

# custo constante
t_0 = equipdb['t_0'].values

cluster = equipdb['cluster'].values
eta = [eta_rule[i] for i in cluster]
beta = [beta_rule[i] for i in cluster]

priori = [weibull(t_0i, eta_i, beta_i) for (t_0i, eta_i, beta_i) in zip(t_0, eta, beta)]

f = equipdb['f'].values

# # test eval
# X = []

# # random
# x = np.random.choice(len(mpdb), len(equipdb)) + 1
# print('M: {} | F: {}'.format(M(x), F(x)))
# X.append(x)

# # min M
# x = np.ones(shape = 500)
# print('M: {} | F: {}'.format(M(x), F(x)))
# X.append(x)

# # min F
# x = np.ones(shape = 500) * 3
# print('M: {} | F: {}'.format(M(x), F(x)))
# X.append(x)

# # export
# X = pd.DataFrame(X).astype('int32').to_csv('tc/xhat.csv', header = False, index = False)

# solucao intuitiva eh fazer manutencao nos aparelhos caros e antigos
# logo, uma ideia eh ordenar o f*priori e fazer manutencao gradual
esperado = priori * f
# silh(np.copy(esperado), 'E(pf)', 'F esperado iniciado')

importancia = np.argsort(esperado)

# modulo deslizante 1d
# alpha = 0.5

# N = int(len(equipdb) * alpha)
# x = np.hstack((np.ones(shape = N), np.ones(shape = len(equipdb) - N) * 3))
# x = x[importancia.argsort()]

# # verifica
# log = []
# for i in importancia:
#     log.append([i, x[i]])
# log = np.array(log)

num = 100
alphas = np.linspace(start = 0, stop = 1, num = num)
log = []
for alpha in alphas:
    N = int(len(equipdb) * alpha)
    x = np.hstack((np.ones(shape = N), np.ones(shape = len(equipdb) - N) * 3))
    x = x[importancia.argsort()]
    
    log.append(F(x))

# plot
plt.figure()
plt.plot(alphas, log)
plt.xlabel('alpha')
plt.ylabel('F(x)')
plt.show()
