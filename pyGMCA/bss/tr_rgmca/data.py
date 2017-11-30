# -*- coding: utf-8 -*-
"""
Building of the components, whose parameters are set in the file paremeters.
"""
#Import files"
from parameters import*
import numpy as np
import scipy as sp
from scipy.fftpack import idct,dct
from scipy import stats

#################################################################################################################################################################################################################################
    
def sources():
    '''
    Creation of the sources, exactly sparse in DCT. 
    Output: a n*t matrix, sparse in DCT.
    '''
    S=np.zeros((dS['n'],dS['t']))   
    S=np.array(sp.stats.bernoulli.rvs(dS['pS'],size=( dS['n'],dS['t'])))
    S[S>0]=1
    S=S*np.random.randn( dS['n'],dS['t'])*dS['gS']
    S=idct(S, type=2, norm='ortho',axis=1)
    return S

    
#################################################################################################################################################################################################################################
def gaussianNoise():
    ''' 
    For the Gaussian noise. 
    Output: a m*t matrix, with i.i.d normal entries
    '''
    N=np.random.randn( dS['m'],dS['t'])*dS['gN']  
    return N
    
    
#################################################################################################################################################################################################################################    
def outliers()  : 
    '''Creation of the outliers matrix, O.
    A total of dS['nbCol'] columns of O are entirely corrupted, with Gaussian active entries.
    Output: a m*t matrix.
    '''
    O=np.zeros((dS['m'],dS['t']))
    while np.sum(O[0,:]>0)<dS['nbCol']:
           O[:,np.random.randint(0,dS['t'])]=1
    O[O>0]=1
    O=O* np.random.randn( dS['m'],dS['t'])
    return O
#################################################################################################################################################################################################################################    
def mixingMatrix():
    '''
    Create the mixing matrix.
    First a m*n matrix, with i.i.d. entries drawn from the normal law is generated.
    Then the columns of A are normalized.
    Output: a m by n matrix.
    '''
    
    A=np.random.randn(dS['m'], dS['n'])
    A/=np.linalg.norm(A, axis=0)
    
    return A     
    
