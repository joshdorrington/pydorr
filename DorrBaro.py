import numpy as np


##these are the basis funcs

def spec1(x,y,b):
    
    return np.sqrt(2.0)*np.cos(y/b)/b

def spec2(x,y,b):
    
    return 2.0*np.cos(x)*np.sin(y/b)/b

def spec3(x,y,b):
    
    return -2.0*np.sin(x)*np.sin(y/b)/b

def spec4(x,y,b):
    
    return np.sqrt(2.0)*np.cos(2.0*y/b)/b

def spec5(x,y,b):
    
    return 2.0*np.cos(x)*np.sin(2.0*y/b)/b

def spec6(x,y,b):
    
    return -2.0*np.sin(x)*np.sin(2.0*y/b)/b

def spec7(x,y,b):
    return 2.0*np.cos(2.0*x)*np.sin(y/b)/b

def spec8(x,y,b):
    return -2.0*np.sin(2.0*x)*np.sin(y/b)/b

def spec9(x,y,b):
    
    return 2.0*np.cos(2.0*x)*np.sin(2.0*y/b)/b

def spec10(x,y,b):
    
    return -2.0*np.sin(2.0*x)*np.sin(2.0*y/b)/b


"""This function is designed to take some spectral coefficients and return a meshed projection on which they can be plotted
It takes as input a NxD length array of coeffs, a D length array of corresponding functions, a 2x2 meshbounds that sets the 
boundary values of the gridand length 2 meshshape array that defines number of points on each mesh axis"""


def spec2mesh(spec_coeffs,*args,mesh_funcs=[spec1,spec2,spec3,spec4,spec5,spec6,spec7,spec8,spec9,spec10],meshbounds=[[0,2*np.pi],[0,0.5*np.pi]],meshshape=[50,50]):
    input_dims=np.shape(spec_coeffs)
    
    #make sure array is of right dimensions and is numpy array
    if len(input_dims)==1:
        spec_coeffs=np.reshape(spec_coeffs,[1,input_dims[0]])
    else:
        spec_coeffs=np.reshape(spec_coeffs,[input_dims[0],input_dims[1]])
    
    input_dims=np.shape(spec_coeffs)#new standardised shape
    
    if input_dims[1]!=len(mesh_funcs):
        raise ValueError("mismatch between number of function and coefficients")
    if np.shape(meshbounds)!=(2,2):
        raise ValueError("meshbounds must be shape (2,2)")
        
        
    #set up memory for grid data and define the x and y axes
    grid=np.zeros([input_dims[0],meshshape[0],meshshape[1]])
    x=np.linspace(meshbounds[0][0],meshbounds[0][1],meshshape[0])
    y=np.linspace(meshbounds[1][0],meshbounds[1][1],meshshape[1])
    
    xgrid=x[:,None].repeat(len(y),axis=1).T
    ygrid=y[:,None].repeat(len(x),axis=1)

    
    for row in range(0,input_dims[0]):
        for coeff in range(0,input_dims[1]):
            grid[row,:,:]+=spec_coeffs[row,coeff]*mesh_funcs[coeff](xgrid,ygrid,*args)
            
    return grid