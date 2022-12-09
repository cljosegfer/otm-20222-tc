#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:37:37 2022

@author: jose
"""

from pareto import rea

num = 150
experimento = rea()
export, border = experimento.pareto(num)
