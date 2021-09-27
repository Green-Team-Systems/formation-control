import numpy as np
import json
import os
import tempfile
import pprint
import time
import copy

from numpy.core.fromnumeric import shape

from utils.dataclasses import PosVec3, Quaternion



def nbr(adj: np.array, i: int):

    return np.nonzero(adj[:,i])
def CBAA_swarm(adj: np.array, pm: np.array, qm: np.array ) -> np.array :
    '''
    CBAA implementation for swarm assignment
    Source: MIT ACL Lab / aclswarm

    Inputs:
    -adj: n x n adjacency matrix
    -pm : 3 x n desired formation points
    -qm : 3 x n matrix of current vehicle positions

    assign 1 x n permutation vector
    Maps idx of qm to idx of pm (formpt)
    '''
    n = np.shape(pm)[1] #number of agents
    id = np.identity(n)
    #G = adj + np.identity(n)
    G = np.array([[adj[i][j] + id[i][j]  for j in range(len(adj[0]))] for i in range(len(adj))])
    print(type(G))


    ##Alignment

    # Compute the R, t, aligned_ps for agents
    Rs = []*n
    ts = []*n
    aligned_ps = []*n
    
    for i in list(range(1,n)):
        neighbors = G[:,i]
        ps = pm[:,i]
        qs = qm[:,i]


        

#--------------------TEST------------------

n = 4
Adj = np.ones(n) - np.identity(n)
x_i = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
p_i = np.array([[-2, 4, 2, 4],[0, 0, 0, 0]])


output = CBAA_swarm(Adj, p_i, x_i )
