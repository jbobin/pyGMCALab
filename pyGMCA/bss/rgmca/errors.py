# -*- coding: utf-8 -*-
"""
Functions for performing the errors made on the factorization:
- To reoder the factorization: function reOrder
- to compute the errors made on the mixing matrix: function errors

"""

import numpy as np
from munkres import munkres
from parameters import dS
#################################################################################################################################################################################################################################

def reOrder(A,S,Aori,Sori):
    ''' Reorder the factorization.
    Implementation similar to the one of J.Rapin
    Maximization of the SDR between A and Aori to reorder the factorization.
    (Implementation of J. Rapin with maximization of the correlation between S and Sori, but this
    can fail in the presence of outliers).
    Input:
    A:estimated matrix
    S:estimated sources
    Aori: initial mixing matrix
    Sori: initial sources
    Output: reordered S and A
    '''
    
    Ae= A.copy() #Estimated A
    Ar = Aori.copy() # A initial
    r=np.shape(Ae)[1]
    
    # set nan values to 0
    Ar[np.isnan(Ar)] = 0
   
    
    
    SDR_S = computeSDRmatrix(Ar, Ae)# compute SDR matrix
    costMatrix = -SDR_S #cost function to minimize
    hungarian = munkres.Munkres() 
    ind_list = hungarian.compute(costMatrix.tolist()) # Hungarian algorithm
    
    indices = np.zeros(r, dtype = int)
    for k in range(0, r):
        indices[k] = ind_list[k][1]
    
    # reorder the factorization
    A = A[:, indices]
    S = S[indices, :]
   
   # Change the sign of the signals if necessary
    for index in range(0,r):
        if np.sum(A[:,index]*Aori[:,index])<=-np.sum(A[:,index]*Aori[:,index]):
            S[index,:]=-S[index,:]
            A[:,index]=-A[:,index]
            
    return S,A
       
    
#################################################################################################################################################################################################################################


def computeSDRmatrix(X, Y):
    '''Compute the SDR matrix between X and Y
    Implementation from J. Rapin
    Input:
    X:  reference of column signals
    Y:  estimate of column signals
    Output : MSDR, matrix of the SDR such that MSDR_i,j is is the SDR between the i-th column of X with the j-th column of Y
    '''
    
    #normalize the reference
    X = X / np.linalg.norm(X,axis=0);
    
    #Dimensions
    n_x = X.shape[1]; 
    n_y = Y.shape[1];
    L = X.shape[0];
    
    #SDR matrix
    MSDR = np.zeros([n_x, n_y]);
    
    
    for n in range(0, n_x):
        #projection of then th  column of X on the estimate Y
        targets = X[:, n].reshape([L, 1]) * (X[:, n].T.dot(Y));
        diff = Y - targets; # difference between Y and the projection of the nth column of X projected on each column of Y
        
        
        norm_diff_2 = np.maximum(np.sum(diff * diff, 0), np.spacing(1)); #norm of diff, by column
        norm_targets_2 = np.maximum(
            np.sum(targets * targets, 0), np.spacing(1)); #norm of the target, by column
        MSDR[n, :] = -10 * np.log10(norm_diff_2 / norm_targets_2); #ratio between the projection and the difference
    
    return MSDR;
#################################################################################################################################################################################################################################
    
def errors(A,S,Aori,Sori):
    '''Compute the errors on the estimation of A.
    First reorder the factorization. Then compute Delta_A and the number of columns doing an angle smaller than 5 degrees.
    Input:
    A: estimated mixing matrix
    S: estimated sources
    Aori: initial mixing matrix
    Sori: initial sources
    Output: the number of columns doing an angle smaller than 5 degrees (normalized to one) and DeltaA
    '''
    #Reorder the factorization
    S,A=reOrder(A,S,Aori,Sori)
    
    #Compute DeltaA =  || pseudoInverse(A)* Aori - I ||_1 / (number of sources)^2
    DeltaA=np.sum(np.abs(np.linalg.pinv(A).dot(Aori)-np.eye(dS['n'])))/(dS['n']**2)
    
    #Compute the angles done between the columns of A and Aori (between 0 and 90):
    angle=np.arccos(np.abs(np.sum(A*Aori,axis=0)))*180./np.pi
    angle[np.isnan(angle)]=90
   
    #(Normalized) number of recovered columns
    angle5=np.sum(angle<5)/np.float(dS['n'])
    
    return angle5, DeltaA
        
    
    
     
    
