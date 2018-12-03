#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:26:15 2018

@author: dorrington
"""

import numpy as np
import scipy

"""This function takes in an array of states and returns 2 arrays,
L: the lifetime of a state
S: the state it was in"""

def reg_lens(state_arr):
    sa=np.asarray(state_arr)
    n=len(sa)

    if n==0:
        return (None,None)
    
    else:
        #we create a truth array for if a state is about to change:
        x=np.array(sa[1:]!=sa[:-1])
        #we create an array containing those entries of sa where x is true
        #and append the final entry:
        y=np.append(np.where(x),n-1)
        #we create an array of persistent state lengths:
        L=np.diff(np.append(-1,y))
        #and an array of those state values:
        S=sa[y]
        return (L,S)

    
def make_samples(transmat,length):
    
    mat=np.asarray(transmat)
    n,m=mat.shape
    if n!=m:
        print("transmat must be square")
        return None
    samples=np.zeros(length,dtype=int)
    samples[0]=int(np.random.choice(np.arange(0,n)))
    
    for i in range(1,length):
        samples[i]=np.random.choice(np.arange(0,n),p=mat[samples[i-1]])
        
    return samples
        
def regime_simulator(K,D,length,transmat,means,covars):
    if means.shape!=(K,D):
        print("means are the wrong shape")
        return None
    if covars.shape!=(K,D,D):
        print("covars are the wrong shape")
        return None
    if transmat.shape!=(K,K):
        print("transmat is the wrong shape")
        return None
    series=np.zeros([D,length])
    
    states=make_samples(transmat,length)
    
    for i,state in enumerate(states):
        series[:,i]=np.random.multivariate_normal(means[state],covars[state])
    
    return (series,states)

def state_lagged_entropy(state,trange,state_vec):
    
    states=np.unique(state_vec)
    if state not in states:
        print("Invalid state index")
        return 0
    
    #the climatological distribution of states and their entropy
    P_c=[np.sum(state_vec==k) for k in states]
    P_c=P_c/sum(P_c)
    H_c=np.sum(P_c*np.log2(1/P_c))
    
    information=np.zeros(len(trange))
    for i,t in enumerate(trange):
        #the time lagged distribution of states and their entropy
        P_t=[np.sum(((state_vec[t:][(state_vec==state)[:-t]])==k)) for k in states]
        P_t=P_t/sum(P_t)
        P_t=np.array([1e-10 if P_it==0 else P_it for P_it in P_t]) #handles divide by zero errors
        H_t=np.sum(P_t*np.log2(1/P_t))
        
        information[i]= H_c - H_t
 
        
    return information
    
#This function generates a transition matrix from a state vector, and any combination of states in the form [[a1,a2],[b,1,b2,b3],...]
    
def transmat(states,state_combinations=None,exclude_diag=False):

    if state_combinations is None:
        state_combinations=np.unique(states)
    K=len(state_combinations)
    trans=np.zeros([K,K])
    
    for i,state1 in enumerate(state_combinations):
        for j,state2 in enumerate(state_combinations):
		
            trans[i,j]=sum((np.isin(states[1:],state2))&(np.isin(states[:-1],state1)))/sum(np.isin(states[:-1],state1))
    if exclude_diag:
        trans -= np.diag(trans)*np.eye(K)
        trans /= trans.sum(axis=1)[:,None]
    return trans
     
def matrix_entropy(matrix):
    eigs,left_eig_vecs=scipy.linalg.eig(matrix,left=True,right=False)
    i=np.argwhere(abs(eigs)==abs(eigs).max())[0][0]
    stat_vec=left_eig_vecs[:,i]
    entropy_mat=np.multiply(matrix,np.log2(matrix));
    entropy_mat=np.where(np.isnan(entropy_mat),0,entropy_mat)
    entropy= np.sum(np.matmul(stat_vec,entropy_mat))
    return abs(entropy)
