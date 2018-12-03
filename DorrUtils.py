#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:01:52 2018

@author: dorrington
"""
import numpy as np
 
"""this function implements the Freedman-Diaconis rule
to calculate an optimal bin number for histograming data"""
 
def fd_bins(arr):
    ia=np.asarray(arr)
    iqr = np.subtract(*np.percentile(ia, [75, 25]))
    n=len(ia)
    h=2*iqr*n**(-1.0/3.0)
    ans=np.round((ia.max() - ia.min())/h)
    return int(ans)
    
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def make_covar_ellipse(cov,centre,std_devs):
    from matplotlib.patches import Ellipse
    x,y=centre
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * 1 * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),width=w, height=h,angle=theta, color='black')
    return(ell)

def geteofs():
    from sklearn.decomposition import PCA
    filename="//data/dorrington/dphil/yr1/trm2/barotropic_model/results/older_results/determ.out"
    data=np.fromfile(filename).reshape([6,200000])
    pca=PCA(n_components=6).fit(data.T)
    return pca.components_
    #first 2 eofs are 0.9338055550667642 of the variance
    
#takes a DxN matrix and DxM matrix and returns the sum of the euclidean distances between the N and M vectors 
def sum_of_distances(vec1,vec2=None):
	if vec2 is None:
		return 2*np.sum((np.sum(vec1**2,0)-np.dot(vec1.T,vec1)))
	else:
		return np.sum((np.sum(vec1**2,0)+np.sum(vec2**2,0)-2*np.dot(vec1.T,vec2)))

#returns the autocorrelation of a 1D array
def acf(x, length):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])