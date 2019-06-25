#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 05:27:05 2019

@author: chenchen
"""

import itertools

def generate_random_vectors(M,dim):
    """Generates Random Vectors which are used to create Sketch Matrix
    
    Parameters
    ----------
    M : int
        Number of Random Vectors
    dim : int
          Dimension of the Random Vector
        
    Returns
    -------
    rand_vectors : NDArray of Size [M,dim]
                   Array of Random Vectors
        
    """
    rand_vectors = np.random.randn(M,dim)
    return rand_vectors

def get_sketch_matrix(X,rand_vectors):
    """Compute the Sketch Matrix for the Data
    
    Parameters
    ----------
    X : sp.spmatrix, shape [N, D]
        Sparse data matrix.
        
    rand_vectors : NDArray of Size [M,dim]
                   Array of Random Vectors
        
    Returns
    -------
    sketch : NDArray, Shape [M,X.rows]
             Sketch matrix of the Sparse Matrix X
        
    """
    sketch = np.zeros((rand_vectors.shape[0], X.shape[0]),dtype = int)
    
    #Iterate for each hash function
    for i in range(rand_vectors.shape[0]):
        sketch[i] = np.sign(X.dot(rand_vectors[i]))
    
    return sketch

def find_potential_duplicates(sketch,startIndex,endIndex):
    """Compute the Potential Duplicate Pairs using the Sketch Matrix
    
    Parameters
    ----------
    sketch : NDArray, Shape [M,X.rows]
             Sketch matrix of the Sparse Matrix X
        
    startIndex : int
                 Start Index of the Band.
    endIndex : int
               End Index Index of the Band.
        
    Returns
    -------
    candidates : Set , len(candidates) = Possible Potential Pairs
                 The potential Candidate Pairs in a particular Band
        
    """
    candidates = set()
    dictList = dict()
    for i in range(sketch.shape[1]):
        v1 = sketch[startIndex:endIndex,i]
        hashKey = hash(tuple(v1))
        
        if hashKey not in dictList:
            dictList[hashKey] = [i]
        else:
            dictList[hashKey].append(i)
            
    
    #Find Candidate Pairs
    for key,val in dictList.items():
        if(len(val) >= 2):
            for pair in itertools.combinations(val,2):
                candidates.add(pair)
    
    return candidates

def LSH_t(X, b=8, r=32, d=0.3):
    """Find candidate duplicate pairs using LSH and refine using exact cosine distance.
    
    Parameters
    ----------
    X : sp.spmatrix, shape [N, D]
        Sparse data matrix.
    b : int
        Number of bands.
    r : int
        Number of rows per band.
    d : float
        Distance threshold for reporting duplicates.
    
    Returns
    -------
    duplicates : {(ID1, ID2, d_{12}), ..., (IDX, IDY, d_{xy})}
        A set of tuples indicating the detected duplicates.
        Each tuple should have 3 elements:
            * ID of the first review
            * ID of the second review
            * The cosine distance between them
    
    n_candidates : int
        Number of detected candidate pairs.
        
    """
    np.random.seed(158)
    n_candidates = 0
    duplicates = set()
    candidateList = set()
    
    ### YOUR CODE HERE ###
    
    #Generate Random Vectors For Hashing
    M = b*r
    rand_vectors = generate_random_vectors(M,X.shape[1])
    
    #Compute Signature/Sketch Matrix
    sketch = get_sketch_matrix(X,rand_vectors)
    
    for i in range(b):
        pot_candidate = find_potential_duplicates(sketch,i*r,i*r+r)
        candidateList.update(pot_candidate)
    
    n_candidates = len(candidateList)
    
    for candidate in candidateList:
        tempList = list(candidate)
        dist = cosine_distance(X,tempList[0], tempList[1])   
        
        if(dist <= d):
            duplicates.add((tempList[0], tempList[1],dist))
    
    return duplicates, n_candidates