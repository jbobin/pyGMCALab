# -*- coding: utf-8 -*-
"""
Data Generation:
- the sources are created with the function sources
- the Gaussian noise is created with the function gaussianNoise
- the outliers are created with the function outliers
- the mixing matrix with mixingMatrix

"""

#import parametres#
import numpy as np
import scipy as sp
from scipy import special
from scipy import stats




#################################################################################################################################################################################################################################
    
#Sources generation#  
def sources(n=8,t=4096,pS=0.05,alphaS=2,ampliS=100):
     '''
      Source generation.
      The sources are drawn from a Bernoulli- (Generalized) Gaussian law. 
      Input:
        n: number of sources
        t: number of samples
        pS: activation parameter of the Bernoulli law
        alphaS: parameter of the Generalized Gaussian law (2 for Gaussian)
        ampliS: standard deviation of the Gaussian law
      Output: matrix size n by t
     '''
     S=np.array(sp.stats.bernoulli.rvs(pS,size=( n,t))) # Support from a Bernoulli law, with activation parameter pS
     S[S>0]=1
     
     S=S*generate2DGeneralizedGaussian(alphaS, n, t) # Amplitude follows a Generalized Gaussian law, with parameter gS
     
     return S*ampliS # Scale with the ampliS, standard deviation
     
    
#################################################################################################################################################################################################################################
#Gaussian noise#
def gaussianNoise(m=16,t=4096,ampliN=0.1):
   '''
      Gaussian noise generation.
      Input:
        m: number of observations
        t: number of samples
        ampliN: standard deviation of the Gaussian law
      Output: matrix size m by t
     '''
   N=np.random.randn(m, t)*ampliN
   
   return N    
    
   
#################################################################################################################################################################################################################################    
#Outliers #
def outliers(m=16,t=4096,nbCol=400,alphaO=2,ampliO=100):
    '''
    Outliers Generation
    Generation of outliers in general position. A total number of nbCol columns are corrupted/active,
    and their amplitude follows a Generalized Gaussian law.
    Input:
    m: number of observations
    t: number of samples
    nbCol: number of corrupted columns
    gO: parameter of the Generalized Gaussian law
    ampliO: standard deviation for the amplitude
    Output: matrix of size m by t.
    '''

    O=np.zeros((m,t))
    
    while np.sum(np.abs(O[0,:]>0))<nbCol:#Support of O#
           O[:,np.random.randint(0,t)]=1
           
    O=O* generate2DGeneralizedGaussian(alphaO,m, t)#Amplitude#

    return O*ampliO#Scale with the standard deviation
    
    

    
#################################################################################################################################################################################################################################    
#Mixing matrix 
def mixingMatrix(m=16,n=8):
    '''
    Generation of the mixing matrix, whose entries are drawn from a Gaussian law. The columns are normalized.
    Input:
    m: number of observations
    n: number of sources
    Output: matrix m by n
    '''
    A=np.random.randn(m, n)#Gaussian entries#
    A/=np.linalg.norm(A,axis=0)#Normalize the columns#

    return A     
    
#################################################################################################################################################################################################################################        
# Generalized Gaussian distribution
def generate2DGeneralizedGaussian(alpha, m,n):
    """
    from Matlab code BSSGUI by J. Petkov by Jakub Petkov
    adapted to Python by J. Rapin in order to have exact same simulated data 
    between Matlab and Python versions
    
    Generates random variables with generalized
    Gaussian distribution with parameter alpha > 0
                     and variance 1.
     The generator is only approximate,
      the generated r.v. are bounded by 1000.
    
     Method: numerical inversion of the distribution function
             at points uniformly distributed in [0,1];
    
    """
    
    r = 0.5 * np.random.random(m * n) + 0.5; # distribution is symmetric
    beta = np.sqrt(special.gamma(3.0 / alpha) / special.gamma(1.0 / alpha)); # it is enough to consider r>0.5
    
    y = r / beta;
    ymin = 1e-20 * np.ones(m * n);
    ymax = 1000 * np.ones(m * n);
    # for simplicity, generated r.v. are bounded by 1000.
    for iter in range(0, 33):
        cdf = 0.5 + 0.5 * special.gammainc(1.0 / alpha, (beta * y) ** alpha);
        
        indplus = np.nonzero(cdf > r);
        if len(indplus) > 0:
            ymax[indplus]=y[indplus];
        
        indminus = np.nonzero(cdf < r);
        if len(indminus) > 0:
            ymin[indminus] = y[indminus];
        
        y = 0.5 * (ymax + ymin);
    
    ind=np.nonzero(np.random.random(m * n) > 0.5);
    if len(ind) > 0:
       y[ind] = -y[ind];
    
    x = y.reshape([n,m]).T.copy();
    return x;
    
    
