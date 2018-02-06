
import sparse2d as sp2
import numpy as np

# pyStarlet

def forward(X,h = [0.0625,0.25,0.375,0.25,0.0625],J = 1):
    
    nX = np.shape(X)
    Lh = np.size(h)
    
    W = sp2.Starlet2D(nX[1],nX[2],nX[0],J,Lh).forward_omp(np.real(X),np.array(h))
    
    return W
    
    
def backward(W,h = [0.0625,0.25,0.375,0.25,0.0625]):
    
    nX = np.shape(W)
    Lh = np.size(h)
    
    rec = sp2.Starlet2D(nX[1],nX[2],nX[0],nX[3]-1,Lh).backward_omp(np.real(W))
    
    return rec
    
def forward1d(X,h = [0.0625,0.25,0.375,0.25,0.0625],J = 1):
    
    nX = np.shape(X)
    Lh = np.size(h)
    
    W = sp2.Starlet2D(nX[1],1,nX[0],J,Lh).forward1d_omp(np.real(X),np.array(h))
    
    normFact = np.array([ 0.72358037,  0.28547571,  0.17796025,  0.12221233,  0.08578265,
        0.06062301,  0.04293804,  0.0302886 ,  0.02141538,  0.01511499,
        0.01055062,  0.00739429,  0.00522072,  0.00372874,  0.00252544,
        0.00280893])
        
    for ii in range(np.shape(W)[2]-1): # Warning: the coarse scale is not normalized, tested
        W[:,:,ii] = W[:,:,ii]/normFact[ii]
        
    return W
    
def backward1d(W,h = [0.0625,0.25,0.375,0.25,0.0625]):
     
    normFact = np.array([ 0.72358037,  0.28547571,  0.17796025,  0.12221233,  0.08578265,
        0.06062301,  0.04293804,  0.0302886 ,  0.02141538,  0.01511499,
        0.01055062,  0.00739429,  0.00522072,  0.00372874,  0.00252544,
        0.00280893])
        
    for ii in range(np.shape(W)[2]-1): # Warning: the coarse scale is not denormalized, tested
        W[:,:,ii] = W[:,:,ii]*normFact[ii]
        
    rec = np.sum(W,axis=2)
    
    return rec