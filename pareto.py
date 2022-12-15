#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:19:54 2022

@author: jose
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from paretoset import paretoset
import time
import matplotlib.pyplot as plt

class rea():
    def __init__(self):
        # database
        self.equipdb = pd.read_csv('tc/EquipDB.csv', header = None, 
                                   names = ['id', 't_0', 'cluster', 'f'])
        self.mpdb = pd.read_csv('tc/MPDB.csv', header = None, 
                                names = ['id', 'k', 'm'])
        self.clusterdb = pd.read_csv('tc/ClusterDB.csv', header = None, 
                                     names = ['id', 'eta', 'beta'])
        
        # dict
        self.eta_rule = dict(enumerate(self.clusterdb['eta'], start = 1))
        self.beta_rule = dict(enumerate(self.clusterdb['beta'], start = 1))
        
        # constant cost
        self.t_0 = self.equipdb['t_0'].values

        cluster = self.equipdb['cluster'].values
        self.eta = [self.eta_rule[i] for i in cluster]
        self.beta = [self.beta_rule[i] for i in cluster]

        self.priori = [self._weibull(t_0i, eta_i, beta_i) for (t_0i, eta_i, beta_i) in zip(self.t_0, self.eta, self.beta)]

        self.f = self.equipdb['f'].values
        
        # custo esperado
        self.esperado = self.F(x = np.ones(shape = 500), sparse = True)
        self.importancia = np.argsort(self.esperado)
        
    def _weibull(self, t, eta, beta):
        return 1 - np.exp(-(t / eta) ** beta)
    
    def _prob(self, t_0, k, eta, beta, priori, deltat = 5):
        return (self._weibull(t_0 + k * deltat, eta, beta) - priori) / (1 - priori)
    
    def F(self, x, sparse = False):
        k = [(-x_i + 5) / 2 for x_i in x]
        p = [self._prob(t_0i, k_i, eta_i, beta_i, priori_i) for (t_0i, k_i, eta_i, beta_i, priori_i) in zip(self.t_0, k, self.eta, self.beta, self.priori)]
        if sparse:
            return p * self.f
        return np.sum(p * self.f)
    
    def M(self, x):
        return np.sum(x-1)
    
    def _filter(self, hv):
        start = time.time()
        mask = paretoset(hv, sense = ['min', 'min'])
        end = time.time()
        return mask, end - start
    
    def _hist(self, X):
        counts, bins = np.histogram(X)
        
        plt.figure()
        plt.hist(bins[:-1], bins, weights = counts)
        plt.xticks([1, 2, 3])
    
    def pareto(self, num = 100, report = True):
        N = len(self.equipdb)
        start = 0
        stop = 1
        
        log = []
        X = []
        
        alphas = np.linspace(start = start, stop = stop, num = num)
        gamas = np.linspace(start = start, stop = stop, num = num)
        for alpha in tqdm(alphas):
            for gama in gamas:
        # for gama in tqdm(gamas):
        #     for alpha in alphas:
                if alpha + gama > 1:
                    continue
                
                nenhuma = int(alpha * N)
                intermediaria = int(gama * N)
                detalhada = 500 - nenhuma - intermediaria
                
                x = np.hstack((np.ones(shape = nenhuma), 
                                np.ones(shape = intermediaria) * 2, 
                                np.ones(shape = detalhada) * 3))
                x = x[self.importancia.argsort()]
                X.append(x)
                log.append([self.M(x), self.F(x)])
        hv = np.array([[report[-2], report[-1]] for report in log])
        mask, deltat = self._filter(hv)
        border = hv[mask]
        
        # self._hist(np.array(X))
        # self._hist(np.array(X)[mask])

        # export
        export = pd.DataFrame(X).iloc[mask].astype('int32')
        export.to_csv('tc/xhat.csv', header = False, index = False)
        
        if report:
            print('num: {} | sol: {} | filter: {} | time: {}'.format(num, len(X), len(export), deltat))
        
        # return export, border, len(X)
        return export, border

    def pareto_naive(self, num = 100, report = True):
        N = len(self.equipdb)
        start = 0
        stop = 1
        
        log = []
        X = []
        
        alphas = np.linspace(start = start, stop = stop, num = num)
        for alpha in tqdm(alphas):
            nenhuma = int(alpha * N)
            detalhada = 500 - nenhuma
            
            x = np.hstack((np.ones(shape = nenhuma), 
                            np.ones(shape = detalhada) * 3))
            x = x[self.importancia.argsort()]
            X.append(x)
            log.append([self.M(x), self.F(x)])
        hv = np.array([[report[-2], report[-1]] for report in log])
        mask, deltat = self._filter(hv)
        border = hv[mask]

        # export
        export = pd.DataFrame(X).iloc[mask].astype('int32')
        export.to_csv('tc/xhat.csv', header = False, index = False)
        
        if report:
            print('num: {} | sol: {} | filter: {} | time: {}'.format(num, len(X), len(export), deltat))
        
        # return export, border, len(X)
        return export, border
