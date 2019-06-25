#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 05:36:08 2019

@author: chenchen
"""


duplicates, n_candidates = LSH(scaled_data, b=3, r=64, d=0.0003)
duplicates_t, n_candidates_t = LSH_t(scaled_data, b=3, r=64, d=0.0003)
print('We detected {} candidates.'.format(n_candidates))
print('We detected {} candidates.'.format(n_candidates_t))