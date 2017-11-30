# -*- coding: utf-8 -*-


"""
File containing the methods relative to the metric errors. It has mainly been inspired by
the python code proposed by J. Rapin in the toolbox pyGMCALab, at http://www.cosmostat.org/software/gmcalab.

"""

import numpy as np
import scipy as sp
from munkres import munkres
from parameters import dS
from copy import copy




def  SD(O,S,A,Oori,Sori,Aori):
    '''
    Compute the metrics, according to Performance measurement in blind audio source separation, E.Vincent et al.
    Implementation similar to the one of the toolbox pyGMCALab, at http://www.cosmostat.org/software/gmcalab.

    Inputs: 
    - O: estimated outliers, m by t
    - S: estimated sources, n by t
    - A: estimated mixing matrix, m by n
    - Oori: reference outliers, m by t
    - Sori: initial sources, n by t
    - Aori: reference mixing matrix, m by n
    Outputs: a 8 by 1 matrix containing:
    - the minimal SDR, median SDR, minimal and median SIR, SNR and last SAR.
    Note the maximal value of the criteria are upper bounded by 60 dB
    '''
    criteria={}

    if np.sum(np.isnan(S))==0:
        Sr = Sori.T / np.linalg.norm(Sori.T, axis=0); # normalized sources
        Or=Oori.T/ np.linalg.norm(Oori.T, axis=0); #normalized outliers
        
        
        pS = Sr.dot(sp.linalg.lstsq(Sr, S.T)[0]); # Projection of S on Sr
        SN = np.hstack((Sr, Or));  
        pSN = SN.dot(sp.linalg.lstsq(SN, S.T)[0]); # Projection of S on (Sr U Or)
       
        
        eps = np.spacing(1);
        
        decomposition = {};
        #targets
        decomposition['target S'] = sp.sum(S.T * Sr, 0) * Sr; #Projection of S_i on Sori_i
    
        #interferences
        decomposition['interferences SS'] = pS - decomposition['target S'];
        #noise
        decomposition['noise Tot'] = pSN - pS;
    
    
        #artifacts
        decomposition['artifacts S'] = S.T - pSN ;
    
    
        
        if np.max(np.abs(S)):
            #SDR: source to distortion ratio
            num = decomposition['target S'];
            den = decomposition['interferences SS'] + (
                  decomposition['noise Tot']+ decomposition['artifacts S']);
            norm_num_2 = np.sum(num * num, 0);
            norm_den_2 = np.sum(den * den, 0);
            eps=max(np.max(np.abs(norm_num_2))/1e6, np.spacing(1))  # Upper bound, can be (should be ?) removed.
            mat=np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)
            mat=mat[mat!=1]
            if np.shape(mat)[0]>0:   
                    criteria['SDR Min'] = np.min(10 * np.log10(mat));
                    criteria['SDR Med'] = np.median(10 * np.log10(mat));
            else:
                     criteria['SDR Min'] =- np.inf;
                     criteria['SDR Med'] = -np.inf;      
    
            
            #SIR: source to interferences ratio
            num = decomposition['target S'];
            den = decomposition['interferences SS'];
            norm_num_2 = sum(num * num, 0);
            norm_den_2 = sum(den * den, 0);
            eps=max(np.max(np.abs(norm_num_2))/1e6, np.spacing(1))
            mat=np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)
            mat=mat[mat!=1]
            if np.shape(mat)[0]>0:   
                    criteria['SIR Min'] = np.min(10 * np.log10(mat));
                    criteria['SIR Med'] = np.median(10 * np.log10(mat));
            else:
                     criteria['SIR Min'] =- np.inf;
                     criteria['SIR Med'] = -np.inf;    
            
            #SNR: source to noise ratio (noise==outliers)
            if np.max(np.abs(O))>0: #only if there is noise
                num = decomposition['target S'] + decomposition['interferences SS'];
                den = decomposition['noise Tot'];
                norm_num_2 = sum(num * num, 0);
                norm_den_2 = sum(den * den, 0);
                eps=max(np.max(np.abs(norm_num_2))/1e6, np.spacing(1))
                mat=np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)
                mat=mat[mat!=1]
                if np.shape(mat)[0]>0:   
                    criteria['SNR Min'] = np.min(10 * np.log10(mat));
                    criteria['SNR Med'] = np.median(10 * np.log10(mat));
                else:
                     criteria['SNR Min'] =- np.inf;
                     criteria['SNR Med'] = -np.inf;          
    
            else:
                criteria['SNR Min'] = np.inf;
                criteria['SNR Med'] = np.inf;
                
            
            #SAR: sources to artifacts ratio
            if ((Or).shape[1] + Sr.shape[1] < Sr.shape[0]):
                num = decomposition['target S'] + (
                      decomposition['interferences SS'] + decomposition['noise Tot']);
                den = decomposition['artifacts S'];
                norm_num_2 = sum(num * num, 0);
                norm_den_2 = sum(den * den, 0);
                eps=max(np.max(np.abs(norm_num_2))/1e6, np.spacing(1))
                
                mat=np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)
                mat=mat[mat!=1]
                if np.shape(mat)[0]>0:   
                    criteria['SAR Min'] = np.min(10 * np.log10(mat));
                    criteria['SAR Med'] = np.median(10 * np.log10(mat));
                else:
                     criteria['SAR Min'] =- np.inf;
                     criteria['SAR Med'] = -np.inf;          
    
            else:
                criteria['SAR Min'] = np.inf;
                criteria['SAR Med'] = np.inf;
        else:

           criteria['SDR Med']=-np.inf
           criteria['SIR Med']=-np.inf
           criteria['SNR Med']=-np.inf
           criteria['SAR Med']=-np.inf          
    else:
          
           
           criteria['SDR Med']=-np.inf
           criteria['SIR Med']=-np.inf
           criteria['SNR Med']=-np.inf
           criteria['SAR Med']=-np.inf
           
    return np.array((criteria['SDR Min'],criteria['SDR Med'], criteria['SIR Min'], criteria['SIR Med'], criteria['SNR Min'], criteria['SNR Med'], criteria['SAR Min'], criteria['SAR Med'])).reshape((8,))
    
     
def errors(O,S,A,Oori,Sori,Aori):
    '''Compute the different metrics. 
    The sources and mixing matrix should be 'reordered' (permutation indeterminacy - see reOrder), 
    prior to computing the errors.
    See also SD for precision on SDR,SIR,SAR and SNR (here SOR).
    Inputs:
    - O: estimated outliers, size m by t
    - S: estimated sources, size n by t
    - A: estimated mixing matrix, size m by n
    - Oori: reference outliers, size m by t
    - Sori: reference sources, size n by t
    - Aori: reference mixing matrix, size m by n
    Outputs: 1 by 11 matrix containing the following metrics:
    - Minimal SDR, median SDR, minimal and median SIR, SNR, and SAR (see SD)
    - Square error on the outliers: -10 log10 (norm(O-Oori, Frobenius)/norm(Oori, Frobenius))
    - Delta A: norm(pinv(A).Aori-I_n,1)/(n*n)
    -Maximal spectral angle, in degree: max_i arccos(<A^i, Aori^i>)
    -Median spectral angle, in degree: med_i arccos(<A^i, Aori ^i>)
    '''
        
    return np.hstack((SD(O,S,A,Oori,Sori,Aori) , np.hstack((-10*np.log10(np.linalg.norm(O-Oori, 'fro')/(np.linalg.norm(Oori, 'fro')  +1e-16)),  np.hstack(( np.sum(np.abs(np.linalg.pinv(A).dot(Aori)-np.eye(dS['n'])))/(dS['n']**2)  , np.hstack(( np.max(np.arccos(np.abs(np.sum(A*Aori,axis=0)))*180/np.pi), np.median(np.arccos(np.abs(np.sum(A*Aori,axis=0)))*180/np.pi)))))))))



def reOrder(A,S,Aori,Sori):
    '''
    'Solve' the permutation indeterminancy, by looking for the permutation minimizing
    the error made on the mixing matrix. The signs of the sources and mixing matrix are also changed if necessary.
    Inputs:
    - A: estimated mixing matrix of size m by n
    - S: estimated sources, size n by t
    - Aori: reference mixing matrix, size m by n
    - Sori: reference sources, size n by t
    Outputs:
    - S: reordered sources
    - A: reordered mixing matrix
    '''
     # get column signals
#    import numpy as np
    Ae= copy(A)
    Ar = copy(Aori)
    Sr = copy((Sori).T)
    Se =  copy((S).T)
    r = Sr.shape[1]
    
    # set nan values to 0
    Ar[np.isnan(Ar)] = 0
    Sr[np.isnan(Sr)] = 0
    Se[np.isnan(Se)] = 0
    
    SDR_S = computeSDRmatrix(Ar, Ae)
    costMatrix = -SDR_S
    hungarian = munkres.Munkres()
    ind_list = hungarian.compute(costMatrix.tolist())
    
    indices = np.zeros(r, dtype = int)
    for k in range(0, r):
        indices[k] = ind_list[k][1]
    
    A = A[:, indices]
    S = S[indices, :]
   
    
    
    for index in range(0,dS['n']):
        if np.sum(A[:,index]*Aori[:,index])<=-np.sum(A[:,index]*Aori[:,index]):
            S[index,:]=-S[index,:]
            A[:,index]=-A[:,index]
            
    return S,A
       
    
def computeSDRmatrix(X, Y):
    '''
    Compute the SDR between every column of X and the ones of Y.
    Inputs:
    - X, a matrix with the q column signals of reference
    - Y, a matrix with the p estimated columns signals
    Output:
    - MSDR: a matrix of size q by p; such that MSDR_ij is the SDR between the i-th column of X with the j-th column of Y
    '''
   
    X /= np.linalg.norm(X, axis=0);
    
    n_x = X.shape[1];
    n_y = Y.shape[1];
    L = X.shape[0];
    MSDR = np.zeros([n_x, n_y]);
    
    for n in range( n_x):
        targets = X[:, n].reshape([L, 1]) * (X[:, n].T.dot(Y));
        diff = Y - targets;
        
        norm_diff_2 = np.maximum(np.sum(diff * diff, 0), np.spacing(1));
        norm_targets_2 = np.maximum(
            np.sum(targets * targets, 0), np.spacing(1));
        MSDR[n, :] = -10 * np.log10(norm_diff_2 / norm_targets_2);
    
    return MSDR;
    

