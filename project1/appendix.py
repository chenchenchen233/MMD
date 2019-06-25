#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 03:40:19 2019

@author: chenchen
"""

b=8
r=32
num_vectors= r*b #number of hash functions
N,D=np.shape(scaled_data) #N=10000, D=8(features)
random_vectors=np.random.randn(D,r) 
test=np.dot(scaled_data,random_vectors)>=0


class LSH:
    def _init_(self, data):
        self.data=data
    def _generate_random_vectors(self, num_vector, dim):
        return np.random.randn(dim, num_vector)
    def gettable(self, num_vector):
        dim = self.data.shape[2]
        random_vectors = self._generate_random_vectors(dim,num_vector)
        powers_of_two = 1 << np.arange(num_vector -1, -1, -1)
        
        table={}
        
        ## boolen 
        bin_index_bits = (np.dot(self.data, random_vectors) >= 0)
        ## datasize*r , each rows is the certain value
        bin_indices = np.dot(bin_index_bits, power_of_two)
        
        for data_index, bin_index in enumerate(bin_indices):
            if bin_index not in table:
                table[bin_index]=[]
            table[bin_index].append(data_index)