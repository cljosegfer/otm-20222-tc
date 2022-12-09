#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:28:21 2022

@author: jose
"""

import pandas as pd
import numpy as np
from silh import silh
import matplotlib.pyplot as plt
from tqdm import tqdm

def M(x):
    return np.sum(x-1)

def weibull(t, eta, beta):
    return 1 - np.exp(-(t / eta) ** beta)

def prob(t_0, k, eta, beta, priori, deltat = 5):
    return (weibull(t_0 + k * deltat, eta, beta) - priori) / (1 - priori)

def F(x):
    k = [(-x_i + 5) / 2 for x_i in x]
    p = [prob(t_0i, k_i, eta_i, beta_i, priori_i) for (t_0i, k_i, eta_i, beta_i, priori_i) in zip(t_0, k, eta, beta, priori)]
    return np.sum(p * f)

def hv(m, f):
    return (1000 - m)*(1745.4898 - f) / (1745.4898 - 1048.1788) / 1000
    
def hv_x(x):
    return (1000 - M(x))*(1745.4898 - F(x)) / (1745.4898 - 1048.1788) / 1000

# read
equipdb = pd.read_csv('tc/EquipDB.csv', header = None, 
                      names = ['id', 't_0', 'cluster', 'f'])
mpdb = pd.read_csv('tc/MPDB.csv', header = None, 
                   names = ['id', 'k', 'm'])
clusterdb = pd.read_csv('tc/ClusterDB.csv', header = None, 
                        names = ['id', 'eta', 'beta'])

# dict
eta_rule = dict(enumerate(clusterdb['eta'], start = 1))
beta_rule = dict(enumerate(clusterdb['beta'], start = 1))

# custo constante
t_0 = equipdb['t_0'].values

cluster = equipdb['cluster'].values
eta = [eta_rule[i] for i in cluster]
beta = [beta_rule[i] for i in cluster]

priori = [weibull(t_0i, eta_i, beta_i) for (t_0i, eta_i, beta_i) in zip(t_0, eta, beta)]

f = equipdb['f'].values

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

# num = 100
# alphas = np.linspace(start = 0, stop = 1, num = num)
# log = []
# logloglog = []
# for alpha in alphas:
#     N = int(len(equipdb) * alpha)
#     x = np.hstack((np.ones(shape = N), np.ones(shape = len(equipdb) - N) * 3))
#     x = x[importancia.argsort()]
    
#     log.append(F(x))
#     logloglog.append([x, alpha, M(x), F(x)])

# # plot
# plt.figure()
# plt.plot(alphas, log)
# plt.xlabel('alpha')
# plt.ylabel('F(x)')
# # plt.show()
# plt.savefig('fig/intuicao1d.png', dpi = 150)

# hv = [hv(report[-2], report[-1]) for report in logloglog]
# alpha = logloglog[np.argmax(hv)][1]
# print(alpha)

# plt.figure()
# plt.plot(alphas, hv)
# plt.xlabel('alpha')
# plt.ylabel('HV')
# plt.show()

# modulo deslizante 2d
# N = len(equipdb)

# alpha = 0.3
# gama = 0.4

# nenhuma = int(0.3 * N)
# intermediaria = int(0.4 * N)
# detalhada = 500 - nenhuma - intermediaria

# x = np.hstack((np.ones(shape = nenhuma), 
#                 np.ones(shape = intermediaria) * 2, 
#                 np.ones(shape = detalhada) * 3))
# x = x[importancia.argsort()]

N = len(equipdb)
num = 100
alphas = np.linspace(start = 0, stop = 1, num = num)
gamas = np.linspace(start = 0, stop = 1, num = num)
log = []
logloglog = []
for alpha in tqdm(alphas):
    loglog = []
    for gama in gamas:
        if alpha + gama > 1:
            loglog.append(1e3)
            continue
        
        nenhuma = int(alpha * N)
        intermediaria = int(gama * N)
        detalhada = 500 - nenhuma - intermediaria
        
        x = np.hstack((np.ones(shape = nenhuma), 
                        np.ones(shape = intermediaria) * 2, 
                        np.ones(shape = detalhada) * 3))
        x = x[importancia.argsort()]
        logloglog.append([x, alpha, gama, M(x), F(x)])
        
        loglog.append(F(x))
    log.append(loglog)
log = np.array(log).T

hv = [hv(report[-2], report[-1]) for report in logloglog]
param = logloglog[np.argmax(hv)][1:3]
print(param)

plt.figure()
plt.plot(hv)
plt.ylabel('HV')
plt.show()

# plot
plt.figure()
extent = [gamas[0], gamas[-1], alphas[-1], alphas[0]]
plt.imshow(log, cmap = 'gray', extent = extent)
plt.xlabel('gama')
plt.ylabel('alpha')
# plt.show()
plt.savefig('fig/intuicao2d.png', dpi = 150)