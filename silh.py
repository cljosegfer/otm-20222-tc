#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:33:46 2022

@author: jose
"""

import matplotlib.pyplot as plt
import numpy as np

def silh(custo, label = None, title = None):
    plt.figure()
    y_lower = 10
    ith_cluster_silhouette_values = custo
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 
                               0, ith_cluster_silhouette_values, 
                               alpha = 0.7)
    
    plt.yticks([])
    plt.xlabel(label)
    plt.title(title)
    plt.show()
