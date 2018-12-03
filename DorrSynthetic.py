#This function creates random data with the same temporal autocorrelation
#as the data passed in
import numpy as np
from numpy import e
from numpy.random import rand
from numpy.fft import fft, ifft

#This function takes a data array of arbitrary shape [S] and returns an array 
#'synth_data' of shape [S,copies]. 'synth_data' is random data with the same
#correlation structure over the 't_ax' axis as the original, and same mean and
#standard deviation. Note that any nonlinear correlation between Fourier phases
#is lost.
def autocorrelated_like(data,t_ax=0,copies=1):
    
    #Makes sure 't_ax' provided is sensible
    if t_ax>=np.ndim(data):
        raise IndexError("t_ax is greater than rank of data")
        

    synth_data=np.zeros([*np.shape(data),copies])

    #get mean and std and remove from data
    mean=np.mean(data,axis=t_ax,keepdims=True)
    std=np.std(data,axis=t_ax,keepdims=True)
    data=data-mean
    data=data/std
    
    #fourier transform the data
    fr_trnsfrm=fft(data,axis=t_ax)
    
    #generate random phases
    rand_phase=rand(*np.shape(synth_data))
    
    #multiply the fourier transform by the phase, and reverse the transform
    #generating random data with the same autocorrelation along axis 't_ax'
    for i in range(copies):
        synth_data[...,i]=np.real(ifft(fr_trnsfrm*e**(1j*rand_phase[...,i]),axis=t_ax))

    #put the correct mean and std back in
    synth_data=synth_data*std[...,None]
    synth_data=synth_data+mean[...,None]
    return synth_data

#generates multimodal gaussian datasets of a 'shape' [D,T] with 'n_modes' modes,
#each equipped with a mean and variance. If no transmat is passed, a homogeneous
#transmat is assumed
def multimodal_gaussians(shape,means,variances,n_modes=1,trans_mat=None):
    
    from DorrRegime import make_samples
    
    if len(shape)!=2:
        raise ValueError("The 'shape' should be 2D")
    
    D,T=shape
    
    if np.shape(means)!=(n_modes,D):
        raise ValueError("'means' should be of shape (n_modes,D)")
    if np.shape(variances)!=(n_modes,D,D):
        raise ValueError("'variances' should be of shape (n_modes,D,D)")
      
    synth_data=np.zeros([D,T])
    if n_modes==1:
        synth_data=np.random.multivariate_normal(means[0],variances[0],T).T
        
    else:
        #define transmat if not defined
        if trans_mat is not None:
            if np.shape(trans_mat)!=(n_modes,n_modes):
                raise ValueError("'trans_mat' should be of shape (n_modes,n_modes)")
            if np.any(np.sum(trans_mat,axis=1)-1)>1e-6:
                raise ValueError("transition matrix rows must add to 1")
        else:
            print("Assuming a homogeneous transfer matrix")
            trans_mat=np.ones([n_modes,n_modes])/n_modes
        
        #generate states from transmat
        states=make_samples(trans_mat,T)
    
        #for each state, sample from gaussian
        for state in range(n_modes):
            N=sum(states==state)
            synth_data[:,states==state]=\
            np.random.multivariate_normal(means[state],variances[state],N).T        
    
    return (synth_data,states)
            