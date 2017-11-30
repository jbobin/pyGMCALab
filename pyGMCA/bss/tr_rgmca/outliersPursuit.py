# -*- coding: utf-8 -*-
"""
Implementation of the combination OP+GMCA.
First OP is performed on the observations to separate the low-rank term L from the sparse outliers O (see 'Robust PCA via Outlier Pursuit', H.Xu et al, IEEE TIT 2012) 
Then GMCA is performed on L to recover A and S.
The combination is performed for different values of the parameter lambda for OP, and the best estimation of A is returned.

"""

import numpy as np
from errors import reOrder
from parameters import dS
from copy import *
import XMCA

      
#################################################################################################################################################################################################################################
        
def softThres(X,thres):
    '''
    Soft thresholding operator
    Input:
    X: the matrix to be thresholded
    thres: threshold
    Output: soft-thresholding of X by thres
    '''
    y=np.maximum(np.abs(X)-thres*np.ones((np.shape(X)[0],1)),0)*np.sign(X)
    return y
#################################################################################################################################################################################################################################
def OP_GMCA(M, Aori, Sori, Aini): 
     '''Implementation of outliers pursuit combined with GMCA. The implementation of outliers pursuit, 
     performed for different parameters values, followed the one proposed in 'Robust PCA via Outlier Pursuit', H.Xu et al, IEEE TIT 2012.
     The notations taken are also the ones proposed in the aforementioned  paper.
     Inputs:
     - M: the observations, of size m by t. The outliers should be sparsely represented in this domain (use Otransf, see XMCA, if needed)
     - Aori: the reference mixing matrix, size m by n
     - Sori: the reference sources (in the direct domain) (are sparely represented in DCT, via Stransf, in XMCA)
     - Aini: the mixing matrix for the initialization of GMCA
     -Outputs:
     -Send: the estimated sources, size n by t
     -Aend: estimated mixing matrix, size m by n
     -Oend: estimated outliers, size m by t.
     '''
    
     m=np.shape(M)[0] #dimension m
     
     muOri=0.99*np.linalg.norm(M, 'fro')
     errorX=1e5
     flagExit=0
     delta=1e-5
     nu=0.9
     index=1
     muBar=delta*muOri
     prevRank=0
    
     while flagExit==0 and index<50: #Test for different parameter values
         
         Lambda=1./(np.sqrt(dS['t']))*index/5. #Threshold value
         
         mu=muOri
         
         t=1.
         t_=1.
         k=0
         kMax=15000
         convM=1
         L=np.zeros(np.shape(M))
         L_=np.zeros(np.shape(M))
         C=np.zeros(np.shape(M))
         C_=np.zeros(np.shape(M))
         try :
    
             while (k<kMax and convM>1e-5) or k<30: #While have not converge
                 
                 Yl=L+(t_-1.)/t*(L-L_)
                 Yc=C+(t_-1.)/t*(C-C_)
                 
                 Gl=Yl-0.5*(Yl+Yc-M)
                 Gc= Yc-0.5*(Yl+Yc-M)
                 
                 L_=L.copy()
                 U,s2,UT=np.linalg.svd(np.dot(Gl,Gl.T),full_matrices=False)
                 s=np.sqrt(s2)# eigenvalues of (M-O+Y*1./mu)
                 S=np.zeros((m,m))
                 S[:m, :m] =np.diag(s)
                 Sinv=S.copy()
                 Sinv[S>0]=1./S[S>0]          
                 V=np.dot(Sinv.T,np.dot(UT,Gl))
                 S=softThres(S,mu/2.)# Threshold of the eigenvalues
                 L=np.dot(U, np.dot(S, V)) 
                 
                 C_=C.copy()
                 C=Gc*np.maximum(0, 1.-Lambda*mu/2./np.linalg.norm(Gc, axis=0))
                 t_=t
                 t=(1.+np.sqrt(4*t_**2+1))/2.
                 mu=max(nu*mu, muBar)
                 k+=1
                 convM=np.linalg.norm(L-L_, 'fro')/(np.linalg.norm(L_, 'fro')+1e-16)+np.linalg.norm(C-C_, 'fro')/(np.linalg.norm(C_, 'fro')+1e-16)
             
             L=XMCA.Stransf(L)#DCT/ domain in which the sources are sparse
             
             if np.linalg.matrix_rank(L)>=dS['n'] and prevRank>0: #If the low rank term, L, has a rank larger or equal to n.
                     u,s,v=np.linalg.svd(np.dot(L,L.T))
                     Aini=u[:,0:dS['n']]
                     St,A=XMCA.AMCA(L,Aini,0) #GMCA
                     S=XMCA.Sback(St)
                     S,A=reOrder(A,S,Aori,Sori)  
                     deltaGMCA=np.sum(np.abs(np.linalg.pinv(A).dot(Aori)-np.eye(dS['n'])))/(dS['n']**2) #compute the corresponding error
                     if deltaGMCA<errorX :
                             #If there is an improvement, keep this results
                            errorX=deltaGMCA
                            Send=S.copy()
                            Aend=A.copy()
                            Oend=C.copy()
                            
                     elif deltaGMCA>1.5*errorX : # if the error is significantly larger than the best one, stop the algorithm
                            index+=100
                            

                     index+=1  
                    
        
             

             elif np.linalg.matrix_rank(L)==0: # If the parameters are not correctly chosen, make a big step
                 index+=5
             elif prevRank>0: # if we are not too far from the optimum
                 index+=1
             elif np.linalg.matrix_rank(L)>=dS['n'] and prevRank==0:#if we have made a too large step
                 index=max(1, index-2)
             else:
                 index+=1
             prevRank=np.linalg.matrix_rank(L)
             
         except np.linalg.LinAlgError:
            index+=1
     
   
   
     return Send,Aend,Oend
         

   

    