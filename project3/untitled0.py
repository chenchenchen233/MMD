#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:48:37 2019

@author: chenchen
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from scipy import spatial

data = np.load("task03_data.npy", allow_pickle=True)
reviews_1star = [[x.lower() for x in s] for s in data.item()["reviews_1star"]]
reviews_5star = [[x.lower() for x in s] for s in data.item()["reviews_5star"]]
vocabulary = [x for s in reviews_1star + reviews_5star for x in s]
vocabulary, counts = zip(*Counter(vocabulary).most_common(500))
VOCABULARY_SIZE = len(vocabulary)
EMBEDDING_DIM = 100
"""
Implement
---------
word_to_ind: dict
    The keys are words (str) and the value is the corresponding position in the vocabulary
ind_to_word: dict
    The keys are indices (int) and the value is the corresponding word from the vocabulary
ind_to_freq: dict
    The keys are indices (int) and the value is the corresponding count in the vocabulary
"""

### YOUR CODE HERE ###

ind_to_word=dict(zip(range(500),vocabulary))
word_to_ind=dict(zip(vocabulary,range(500)))
ind_to_freq=dict(zip(range(500),counts))
print('Word \"%s\" is at position %d appearing %d times' % 
      (ind_to_word[word_to_ind['the']], word_to_ind['the'], ind_to_freq[word_to_ind['the']]))

def get_window(sentence, window_size):
    sentence = [x for x in sentence if x in vocabulary]
    pairs = []

    """
    Iterate over all the sentences
    Take all the words from (i - window_size) to (i + window_size) and save them to pairs
    
    Parameters
    ----------
    sentence: list
        A list of sentences, each sentence containing a list of words of str type
    window_size: int
        A positive scalar
        
    Returns
    -------
    pairs: list
        A list of tuple (word index, word index from its context) of int type
    """

    ### YOUR CODE HERE ###
    for i, x in enumerate(sentence):
        for j in range(i-window_size,i+window_size+1):
            if j >= 0 and j < len(sentence):
                if j != i:
                    pairs.append([word_to_ind[sentence[i]],word_to_ind[sentence[j]]])

    return pairs

data = []
for x in reviews_1star + reviews_5star:
    data += get_window(x, window_size=3)
data = np.array(data)

print('First 5 pairs:', data[:5].tolist())
print('Total number of pairs:', data.shape[0])

probabilities = [1 - np.sqrt(1e-3 / ind_to_freq[x]) for x in data[:,0]]
probabilities /= np.sum(probabilities)



class Embedding():
    def __init__(self, N, D, seed=None):
        """
        Parameters
        ----------
        N: int
            Number of unique words in the vocabulary
        D: int
            Dimension of the word vector embedding
        seed: int
            Sets the random seed, if omitted weights will be random
        """

        self.N = N
        self.D = D
        
        self.init_weights(seed)
    
    def init_weights(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        """
        We initialize weight matrices U and V of dimension (D, N) and (N, D) respectively
        """
        self.U = np.random.normal(0, np.sqrt(2 / self.D / self.N), (self.D, self.N))
        self.V = np.random.normal(0, np.sqrt(2 / self.D / self.N), (self.N, self.D))

    def one_hot(self, x, N):
        """
        Given a vector returns a matrix with rows corresponding to one-hot encoding
        
        Parameters
        ----------
        x: array
            M-dimensional vector containing integers from [0, N]
        N: int
            Number of posible classes
        
        Returns
        -------
        one_hot: array
            (N, M) matrix where each column is N-dimensional one-hot encoding of elements from x 
        """

        ### YOUR CODE HERE ###
        M=x.shape[0]
        one_hot=np.zeros([N,M])
        one_hot[x,range(M)] = 1  
       
        assert one_hot.shape == (N, x.shape[0])
        return one_hot

    def loss(self, y, prob):
        """
        Parameters
        ----------
        y: array
            (N, M) matrix of M samples where columns are one-hot vectors for true values
        prob: array
            (N, M) column of M samples where columns are probabily vectors after softmax

        Returns
        -------
        loss: int
            Cross-entropy loss calculated as: 1 / M * sum_i(sum_j(y_ij * log(prob_ij)))
        """

        ### YOUR CODE HERE ###
        N,M=y.shape
        loss=-1.0/M*sum(y*np.log(prob))
        
        return loss
    
    def softmax(self, x, axis):
        """
        Parameters
        ----------
        x: array
            A non-empty matrix of any dimension
        axis: int
            Dimension on which softmax is performed
            
        Returns
        -------
        y: array
            Matrix of same dimension as x with softmax applied to 'axis' dimension
        """
        
        ### YOUR CODE HERE ###
        x=np.exp(x-np.max(x))
        sum_x=np.sum(x,axis=axis)
        sum_x=np.expand_dims(sum_x,axis)
        y=x/sum_x
        
        return y
    
    def step(self, x, y, learning_rate=1e-3):
        """
        Performs forward and backward propagation and updates weights
        
        Parameters
        ----------
        x: array
            M-dimensional mini-batched vector containing input word indices of int type
        y: array
            Output words, same dimension and type as 'x'
        learning_rate: float
            A positive scalar determining the update rate
            
        Returns
        -------
        loss: float
            Cross-entropy loss
        d_U: array
            Partial derivative of loss w.r.t. U
        d_V: array
            Partial derivative of loss w.r.t. V
        """
        
        # Input transformation
        """
        Input is represented with M-dimensional vectors
        We convert them to (N, M) matrices such that columns are one-hot 
        representations of the input
        """
        y_ori = y
        x = self.one_hot(x, self.N)
        y = self.one_hot(y, self.N)

        
        # Forward propagation
        """
        Returns
        -------
        embedding: array
            (D, M) matrix where columns are word embedding from U matrix
        logits: array
            (N, M) matrix where columns are output logits
        prob: array
            (N, M) matrix where columns are output probabilities
        """
        
        ### YOUR CODE HERE ###
        
        U=self.U
        V=self.V
        
        embedding = np.dot(U,x)
        logits=np.dot(V,embedding)
        prob=self.softmax(logits, 0)
        

        
        assert embedding.shape == (self.D, x.shape[1])
        assert logits.shape == (self.N, x.shape[1])
        assert prob.shape == (self.N, x.shape[1])
    
    
        # Loss calculation
        """
        Returns
        -------
        loss: int
            Cross-entropy loss using true values and probabilities
        """
        
        ### YOUR CODE HERE ###
        loss=self.loss(y,prob)
        
        # Backward propagation
        """
        Returns
        -------
        d_U: array
            (N, D) matrix of partial derivatives of loss w.r.t. U
        d_V: array
            (D, N) matrix of partial derivatives of loss w.r.t. V
        """
        
        ### YOUR CODE HERE ###
        N,M = x.shape
        
        mask=np.zeros([N,M])
        mask[y_ori,range(M)]=-1
        scores=np.exp(logits)
        dscores=prob-np.multiply(scores,mask)
        d_V=np.multiply(dscores,y)*embedding.T
        d_U=self.V*np.multiply(dscores,y)*x.T
        
        
       
        
        assert d_V.shape == (self.N, self.D)
        assert d_U.shape == (self.D, self.N)
        
        
        

        
        # Update the parameters
        """
        Updates the weights with gradient descent such that W_new = W - alpha * dL/dW, 
        where alpha is the learning rate and dL/dW is the partial derivative of loss w.r.t. 
        the weights W
        """
        self.V -= (d_V*learning_rate)
        self.U -= (d_U*learning_rate)
        ### YOUR CODE HERE ###

        return loss, d_U, d_V
    
    
def get_loss(model, old, variable, epsilon, x, y, i, j):
    delta = np.zeros_like(old)
    delta[i, j] = epsilon

    model.init_weights(seed=132) # reset weights
    setattr(model, variable, old + delta) # change one weight by a small amount
    loss, _, _ = model.step(x, y) # get loss

    return loss

def gradient_check_for_weight(model, variable, i, j, k, l):
    x, y = np.array([i]), np.array([j]) # set input and output
    
    old = getattr(model, variable)
    
    model.init_weights(seed=132) # reset weights
    _, d_U, d_V = model.step(x, y) # get gradients with backprop
    grad = { 'U': d_U, 'V': d_V }
    
    eps = 1e-4
    loss_positive = get_loss(model, old, variable, eps, x, y, k, l) # loss for positive change on one weight
    loss_negative = get_loss(model, old, variable, -eps, x, y, k, l) # loss for negative change on one weight
    
    true_gradient = (loss_positive - loss_negative) / 2 / eps # calculate true derivative wrt one weight

    assert abs(true_gradient - grad[variable][k, l]) < 1e-5 # require that the difference is small

def gradient_check():
    N, D = VOCABULARY_SIZE, EMBEDDING_DIM
    model = Embedding(N, D)

    # check for V
    for _ in range(20):
        i, j, k = [np.random.randint(0, d) for d in [N, N, D]] # get random indices for input and weights
        gradient_check_for_weight(model, 'V', i, j, i, k)

    # check for U
    for _ in range(20):
        i, j, k = [np.random.randint(0, d) for d in [N, N, D]]
        gradient_check_for_weight(model, 'U', i, j, k, i)

    print('Gradients checked - all good!')

gradient_check()

 d=np.array([[1,2,3],]*3)
 b=np.sqrt(np.sum(a**2,axis=1))
 
 a=np.array([[1,2,3],[2,3,4]])
 c=np.array([b]*N)
 d=np.tile(b,(N,1))
SENTENCES_SIZE=100
ind_per=np.random.permutation(SENTENCES_SIZE)
ind_train=np.sort(ind_per[range(int(0.6*SENTENCES_SIZE)),])
ind_test=np.sort(ind_per[range(int(0.6*SENTENCES_SIZE),int(SENTENCES_SIZE)),])