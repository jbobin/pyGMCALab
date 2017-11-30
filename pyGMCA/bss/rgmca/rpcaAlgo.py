# -*- coding: utf-8 -*-
"""
Implementation of the combination PCP+GMCA.
First PCP is performed on the observations to separate the low-rank term L from the sparse outliers O. The implementation
of PCP follows the suggestions presented in the paper 'Robust Principal Component Analysis?' of E. Candes et al. JACM 2011.
Then GMCA is performed on L to unmix the sources L=AS, 'Sparsity and morphological diversity in blind source separation' J.Bobin et al.
IEEE TIP, 2007.
The combination is performed for different values of the parameter lambda for PCP, and the best estimation of A is returned.

"""

import numpy as np
from errors import reOrder

from parameters import dS,aS,pS

from AGMCA import AMCA

#################################################################################################################################################################################################################################


    
def pcpGMCA(M,Aori,Sori,Aini) :
    '''Algorithm performing the combination PCP+GMCA
    The implementation of PCP follows the one proposed in the article 
    Robust Principal Component Analysis? of E. Candes et al.
    Several combinations PCP+GMCA for different values of the parameter Lambda are performed, 
    and the best estimates are returned.
    Input:
     M: observations
     Aori: initial mixing matrix
     Sori: initial sources
     Aini: initializating mixing matrix for GMCA
    Output: S the estimated sources and A, the estimated mixing matrix
    '''
    #Dimensions
    m=np.shape(M)[0]
    t=np.shape(M)[1]
    
    #Initialization of the variables
    L=M.copy()#Low rank term (corresponding to AS)
    O=0*M #outliers
    Y=np.zeros(np.shape(M))
    
    #Initialization of the parameters for the loop
    index=0
    errorX=20000
    
    #Loop with the varying parameter Lambda
    while index<20:
        
        Lambda=(np.float(index)/2.+0.5)/np.sqrt(t) #current value of lambda
        
        mu=(t*m/(4*np.sum(np.abs(M))))#value of mu, given in the aticle (can do better than that)
        
        #Parameters for pcp
        ite=0
        convO=10
        tot=1e16
        convTot=1e16
        convL=1
        

        try:
            #Perform pcp given the current value of lambda
            while (ite< aS['iteMaxPCP'] and (convL>1e-3 or convO>1e-3 or convTot>1e-6)) or ite<15:
                
                #Update variables for convergence
                tot_=tot #value of the cost function
                O_=O #outliers
                L_=L#low-rank
                
                L,S=updateL(mu,M,O,Y)# update of the low-rank term
                O=updateO(Lambda,mu,M,L,Y) # update of the outliers
                Y=Y+mu*(M-L-O) #update Y, the Lagrange multiplier  matrix
#                

#             
                tot=np.sum(S)+Lambda*np.sum(np.abs(O)) # update cost function
                convTot=np.abs((tot-tot_))/tot# convergence cost function
                convO=np.linalg.norm(O-O_)/np.linalg.norm(O_) #convergence outliers
                convL=np.linalg.norm(L-L_)/np.linalg.norm(L_)# convergence low-rank term
                ite+=1  
                
               
               
               
            #perform GMCA for the returned low-rank term L
            S,A=AMCA(L,Aini.copy(),amca=0)
            S,A=reOrder(A,S,Aori,Sori)  
            deltaGMCA=np.sum(np.abs(np.linalg.pinv(A).dot(Aori)-np.eye(dS['n'])))/(dS['n']**2) #compute the corresponding error
            
           
            if deltaGMCA<errorX:
                 #If there is an improvement, keep this results
                errorX=deltaGMCA
                Sfin=S.copy()
                Afin=A.copy()
                Ofin=O.copy()
                if pS['verbosePCP']:
                    print 'PCP+GMCA, current best error:', errorX, 'at index' , index, 'th index in ', ite, 'iterations'
           
                
            elif deltaGMCA>1.5*errorX: #Otherwise
                index+=1
        except np.linalg.LinAlgError:
            pass
        index+=1
    

    return Sfin,Afin,Ofin
    
#################################################################################################################################################################################################################################
   
def updateL(mu,M,O,Y) :
    '''Update of the low-rank term for PCP.
        The eigenvalues of M-O+Y/mu are thresolded by 1./mu.
        Further details can be found in Robust Principal Component Analysis? of Candes et al.
    Input:
        mu: inverse of the threshold
        M: observations
        O: outliers
        Y: Lagrange multiplier matrix
    Output: the low-rank term L, the eignevalues of L denoted by S
    '''
    
    m=np.shape(M)[0] #dimension m
    
    #Svd on (M-O+Y*1./mu).((M-O+Y*1./mu).T) to fasten
    U,s2,UT=np.linalg.svd((M-O+Y*1./mu).dot((M-O+Y*1./mu).T),full_matrices=True)
    s=np.sqrt(s2)# eigenvalues of (M-O+Y*1./mu)
    S=np.zeros(np.shape(M))
    S[:m, :m] =np.diag(s)
    Sinv=S.copy()
    Sinv[S>0]=1./S[S>0]          
    V=Sinv.T.dot(UT.dot(M-O+Y*1./mu)) #To perform the reconstruction of L
    S=softThres(S,1./(mu))# Threshold of the eigenvalue
    L=U.dot(S.dot(V)) # estimation of L
  
    return L,S
#################################################################################################################################################################################################################################
    
def updateO(Lambda,mu,M,L,Y) :
    '''Update of the sparse term for PCP.
    The outliers M-L+Y*1./mu are thresolded with parameter Lambda/mu
    Input: 
    Lambda: parameter Lambda
    mu: parameter mu
    M:observations
    L: current low-rank term
    Y: Lagrance mutliplier matrix
    Output: O, the sparse term (outliers)
    '''
    
    return softThres(M-L+Y*1./mu,Lambda*1./mu) 
      
#################################################################################################################################################################################################################################
        
def softThres(X,thres):
    '''
    Soft thresholding operator on each row of X
    Input:
    X: the matrix to be thresholded
    thres: threshold
    Output: soft-thresholding of X by thres, for each row
    '''
    y=np.maximum(np.abs(X)-thres*np.ones((np.shape(X)[0],1)),0)*np.sign(X) 
    return y    

        


    