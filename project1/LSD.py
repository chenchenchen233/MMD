#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:07:30 2019

@author: chenchen
"""

import gzip
import tarfile

import numpy as np
import pandas as pd
import time
import itertools

from sklearn import preprocessing
from collections import defaultdict

import matplotlib.pyplot as plt

tar = tarfile.open('/Users/chenchen/Documents/MSNE/Semester4/mining massive data/project/project1/millionsongsubset_full.tar.gz', 'r')
members = tar.getmembers()

##data preprocessing
tar.extract(members[5])
summary = pd.HDFStore(members[5].name)
songs = summary['/analysis/songs']
songs.head()
subset = songs[['duration', 'end_of_fade_in', 'key', 'loudness',
                'mode', 'start_of_fade_out', 'tempo', 'time_signature',]]

data_matrix = subset.values

scaled_data = preprocessing.scale(data_matrix)

## cosine distance
def cosine_distance(X, i, j):
    """Compute cosine distance between two rows of a data matrix.
    
    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix.
    i : int
        Index of the first row.
    j : int
        Index of the second row.
        
    Returns
    -------
    d : float
        Cosine distance between the two rows of the data matrix.
        
    """
    d = None
    
    ### YOUR CODE HERE ###
    d=np.sum(X[i]*X[j])/np.sqrt(np.sum(X[i]**2)*np.sum(X[j]**2))
    
    return d


            
            
def getsketch(X, vectors):
    
    N,D= X.shape
   # SigMatrix = np.zeros(N,num_vectors)
#     print("X dim:", x.shape)
#     print("vectors:", vectors.shape)
    Sig = np.dot(X, vectors)
    Sig[Sig >= 0] = 1
    Sig[Sig < 0] = -1
#     print(Sig.shape)
   # SigMatrix = (np.dot(X, vectors) >= 0)

    return Sig

def find_candidates(sigmatrix, begin, end):
    KeyList= dict()
    candidates = set()
    for i in range(sigmatrix.shape[0]):
        blocki = sigmatrix[i, begin:end]
        hashkey = hash(tuple(blocki)) ## tuple is hashable
        
        if hashkey not in KeyList:
            KeyList[hashkey] = [i] 
        else: KeyList[hashkey].append(i)
        
        
    ### find possible candidates
    for val in KeyList.values():
        if len(val) > 1:
            for pair in itertools.combinations(val, 2):
                candidates.add(pair)
                
    return candidates

def LSH(X, b=8, r=32, d=0.3):
    """Find candidate duplicate pairs using LSH and refine using exact cosine distance.
    
    Parameters
    ----------
    X : np.array shape [N, D]
        Data matrix.
    b : int
        Number of bands.
    r : int
        Number of rows per band.
    d : float
        Distance treshold for reporting duplicates.
    
    Returns
    -------
    duplicates : {(ID1, ID2, d_{12}), ..., (IDX, IDY, d_{xy})}
        A set of tuples indicating the detected duplicates.
        Each tuple should have 3 elements:
            * ID of the first song
            * ID of the second song
            * The cosine distance between them
    
    n_candidates : int
        Number of detected candidate pairs.
        
    """
    np.random.seed(158)
    n_candidates = 0
    duplicates = set()
    candidates = set()

    ### YOUR CODE HERE ###
    N,D = X.shape
    num_vectors = r*b #number of hash functions
   
    random_vectors=np.random.randn(D, num_vectors)
    SigMatrix = getsketch(X, random_vectors)
    for i in range(b):
        candi= find_candidates(SigMatrix, i*r, (i+1)*r)
        candidates.update(candi)
    n_candidates = len(candidates)
    ##find duplicates which distance < d
    for candidate in candidates:
        Candlist = list(candidate)
        dist = cosine_distance(X, Candlist[0], Candlist[1])
        if dist < d:
            duplicates.add((Candlist[0], Candlist[1], dist))
    
    
    
    return duplicates, n_candidates
