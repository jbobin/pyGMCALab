# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import scipy as sp
from parameters import dS,aS
from scipy.fftpack import idct,dct
from errors import reOrder
from copy import copy

def Stransf(S):
    '''
    Forward transformation of the sources, in the domain in which they are sparse.
    In this example, it corresponds to the DCT. Stransf can be changed if the sources
    are sparsely represented in another domain.
    (cf Sback for the backward transformation)
    Input: the sources in the observation domain,  n by t matrix.
    Output:  n by t matrix, corresponding to the sparse coefficients of S, DCT transform of S (orthonormal transformation).
    '''
    return dct(S, type=2, norm='ortho',axis=1)
    
#####################################################################################

def Sback(St):
    '''
    Backward transformation of the sources coefficients (towards the observation domain).
    In this example, it corresponds to the inverse DCT.
    (cf Stransf for the forward transformation)
    Input: St, the sparse coefficients of S (dct representation), of size n by t.
    Output: the sources in the direct/observation domain, n by t matrix.
    '''

    return idct( St, type=2, norm='ortho', axis=1) 
    
#####################################################################################    

def Otransf(O):
    '''
    Forward transformation of the outliers, in the domain in which they are sparse.
    In this example, it corresponds to the identity matrix (the outliers are sparse in the observation domain).
    Otransf can be changed if the outliers are sparsely represented in another domain.
    (cf Oback for the backward transform)
    Input: the outliers in the observation domain,  m by t matrix.
    Output:  m by t matrix, corresponding to the sparse coefficients of O (i.e. O in this example).
    '''
    return O

#####################################################################################    

def Oback(Ot):
    '''
    Backward transformation of the outliers coefficients (towards the observation domain).
    In this example, it corresponds to the identity.
    (cf Otransf for the forward transformation).
    Input: Ot, the sparse coefficients of O (Ot=O in the proposed example), of size m by t.
    Output: the outliers in the direct/observation domain, m by t matrix.
    '''
    return Ot 
    
#####################################################################################

def madAxis(xin,axis='none'):
    ''' 
    Compute the median absolute deviation of a matrix, global or along each row.
    Inputs:
    - xin: the signal. If axis='none', xin can be an array or a list. Otherwise, should be a 2D matrix.
    - axis: if 'none', the mad is performed globally on xin. Otherwise, the mad of each row of xin is computed.
    Output:
    - if axis='none', returns a scalar/otherwise a column vector with the same number of rows than xin, each row containing the mad of the corresponding row of xin.
    '''
    if axis=='none':
        z = np.median(abs(xin - np.median(xin)))/0.6735

        return z
    else:
        z = np.median(abs(xin - np.median(xin,axis=1).reshape((np.shape(xin)[0],1))),axis=1)/0.6735
        return z.reshape((np.shape(xin)[0],1))    

#####################################################################################

def softThres(x,thres,typeThres):
    '''
    Hard or Soft Thresholding operator.
    Inputs:
    - x the signal, of size n1*n2 (can be a scalar)
    - thres, the thresholding values, of size n3*n4; with n3=n1 or n3=1; and n4=n2 or n4=1.
    - typeThres: should be 1 for the soft thresolding and 0 for hard thresholding
    Output:
    - soft/hard thresholded version of x. If thres is a scalar, every entry of x is thresholded by this same value.
    If thres is a column vector, then each row of x is thresholded by the corresponding value of thres. Reciprocally, if 
    thres is a row vector, the i th column of x is thresholded by thres_i. Last, if thres is a matrix, with dimension n1 by n2, each entry of
    x is thresholded by corresponding value in thres.
    '''
    
    return x*(np.array(np.abs(x)>thres,int))-np.sign(x)*thres*np.array(np.abs(x)>thres,int)*typeThres
    
#####################################################################################
    

def trRGMCA(X,Aini,Afix=0):
    '''
    tr-rGMCA algorithm: performs the estimations of A, S and O, based on the
    observations X, the number of sources n (dictionary dS['n']), and the difference of morphology
    between the outliers and the sources.
    This difference of morphology is taken into account with the transformations: Stransf, Otransf (and Sback, Oback).
    Inputs: 
    - X the observations (m by t matrix)
    - Aini the initialized mixing matrix (m by n)
    - boolean Afix: if true, then the oracle is performed (no estimation of A). Otherwise, A is jointly estimated.
    Outputs: S (the estimated sources, n by t), A (estimated mixing matrix, m by n) and O (estimated outliers, m by t matrix).
    '''
    O=np.zeros((dS['m'],dS['t'])) # Estimated outliers
    S=np.zeros((dS['n'],dS['t'])) # Estimated sources
    A=Aini.copy() # Estimated mixing matrix
    Sini=np.linalg.pinv(A).dot(X) #First estimated sources- only used to reorder the sources
    Siniws=Stransf(Sini)# Expansion coefficients of Sini
    
    #Expansion Coefficients of the components in the dictionary in which S is sparse
    Ows=Stransf(O)  
    Sws=Stransf(S)
    Xws=Stransf(X)
    
    #Expansion Coefficients of the components in the dictionary in which O is sparse
    Owo=Otransf(O)
    Swo=Otransf(S)
    Xwo=Otransf(X)
    
    
    #########################################################
       #      Warm-up Phase or Oracle with A known       #
    #########################################################

    
    indexTotW=aS['Rew'] #Number of reweighting loops

    I=0 # Index of the outer loop (A-S// O-S)
    convA=10 # Criterion for the stability of A
    if Afix==0: #If A is estimated, set the maximal number of outer loops to 25.
        loopTot=25
    else:
        loopTot=1# If A is fixed, only O and S are estimated (and the number of outer loops is set to 1)
    while (I<loopTot and (convA>2)) or (I<3 and Afix==0): #Stop the algorithm if the number of outer loops is too large (should not be the case), A is stable, and at least 3 outer loops have been done.
      I+=1
      #Copy of A for convergence
      Aold=A.copy()
      
      #Weights for the reweightings
      WO=np.ones(np.shape(Owo))
      WS=np.ones(np.shape(Sws))
      
     
     
      iteMax=1000 # The maximal number of iterations for the loop estimating S and O
      indexW=0#Set the index of the reweighting procedure to 0 
     
      ########################  Estimation of A #################################
         
      if Afix==0: #If A is estimated
          
          Sws,A=AMCA((Xws-Ows),A,0) #run GMCA on Xws - Ows, for estimating A and S

          (Sws,A)=reOrder(A,Sws,Aini,Siniws)#permutations
          convA=np.max(np.arccos(np.abs(np.sum(Aold*A,axis=0))/(np.linalg.norm(Aold,axis=0)*np.linalg.norm(A,axis=0)))*180/np.pi) #Maximal angle deviation from one estimation of A to another. (Should be smaller than 2 degrees to stop the algorithm).
          
          #Compute the corresponding outliers
          S=Sback(Sws)
          Swo=Otransf(S)
          madO=madAxis(Otransf(X-A.dot(S)-O))
          normO=np.linalg.norm(Xwo-A.dot(Swo),axis=0)
          thr_O=np.sqrt(2)*aS['kOMax']*madO*sp.special.gamma(dS['m']/2.+0.5)/sp.special.gamma(dS['m']/2.)
          if np.max(normO)>thr_O:
                  Owo=(Xwo-A.dot(Swo))*(np.maximum(1.-thr_O*WO/normO,0))
          else:
              O*=0
              Owo*=0

          O=Oback(Owo)
          Ows=Stransf(O)
          
      L=np.linalg.norm(A.T.dot(A), 2)#Maximal eigenvalue of A.T.A
      
    
      ########################  EstimationS of O and S #################################

      if  I==1: #If this is the first loop, do not reweight 
         indexTotW=1
      else:
         indexTot=aS['Rew']
      if Afix==1: #If A is fixed, set the number of reweighting loops to 5
         indexTotW=5
      else:
          indexTotW=1
          
      convW=1 #criterion for the convergence of the joint O-S estimation with reweighting (based on the stability of S)
   
      ########### Reweighting ############
      while indexW<indexTotW and (convW>1e-2 or indexW<2 ):  # Stop the reweighting if S is stable of the number of reweighting steps is reached
          S_W=S.copy() # To check the stability of S
          
          if  indexW==0:   #Do not reweight the output of GMCA

              WS=np.ones((np.shape(Sws)))
              WO=np.ones((np.shape(Owo)))
          else: #Weights based on the current thresholds and components
              
              WS=np.maximum(madS*aS['kSMax'],0)/(madS*aS['kSMax']+np.abs(Sws))
              WO=np.maximum(thr_O,0)/(thr_O+np.linalg.norm(Owo,axis=0))

          index=0
          convO=1
          convS=1
          ####### Estimations O and S ########
          while index< iteMax and ((convO+convS)>1e-5  ): # Stop the algorithm when O and S are stable (should not reach iteMax)
                if index>iteMax-2 and (convA<=1e-4 or Afix==1):
                    print 'Convergence not reached O-S'
                if (convO+convS)>1e-4:  #Fix the parameters for convergence once the components are stable enough
                        madS=madAxis(A.T.dot(Stransf(X-A.dot(S)-O)), axis=1) #projected noise level on the sources
                        madO=madAxis(Otransf(X-A.dot(S)-O)) # noise level for the outliers
                  
                  
                  
                #Copies of the components for convergence check
                O_=Owo.copy()
                S_=S.copy()
                
                
                ###Estimation of S, with LASSO and FISTA implementation###
                # Thresholds of the sources: madS*WS*aS['kSMax']
                convS_sub=1
                indexS=0
                t_=1
                y=S.copy()
                while convS_sub>1e-6 and indexS<5000 :
                        S_old=S.copy()
                        S=Sback(softThres(Stransf(y+1./L*A.T.dot(X-A.dot(y)-O)),madS*aS['kSMax']*1./L*WS,1))
                        t=(1.+np.sqrt(1+4*(t_)**2))/2.
                        y=S+(t_-1.)/(t)*(S-S_old)
                        t_=t
                        convS_sub=np.linalg.norm(S_old-S)/np.linalg.norm(S)
                        indexS+=1
                       
                            
                        
                Swo=Otransf(S)
                Sws=Stransf(S)
                
                         
                
                ### Estimation of O ###
                normO=np.linalg.norm(Xwo-A.dot(Swo),axis=0)
                thr_O=np.sqrt(2)*aS['kOMax']*madO*sp.special.gamma(dS['m']/2.+0.5)/sp.special.gamma(dS['m']/2.)
                if np.max(normO)>thr_O:
                       Owo=(Xwo-A.dot(Swo))*(np.maximum(1.-thr_O*WO/normO,0))
                else:
                    Owo*=0

                O=Oback(Owo)
                Ows=Stransf(O)
               
                ### Convergence criteria O-S###
                convO=np.linalg.norm(Owo-O_)/np.linalg.norm(Owo+1e-16)
                convS=np.linalg.norm(S-S_)/np.linalg.norm(S+1e-16)
                
                index+=1
            
          indexW+=1
          ######Convergence reweigthing######
          convW=np.linalg.norm(S-S_W)/np.linalg.norm(S)


          ###End warm-up###

    #########################################################
                    #      Refinement       #
    #########################################################    

    if Afix==0:
        
        S,A,O=palm_trRGMCA(X, O, S,A )
        
        
    return S,A,O    

#####################################################################################


def palm_trRGMCA(X, Oini, Sini,Aini ):
    '''PALM implementation of tr-rGMCA. It can be used alone (if the sources Sini are null)
    see trRGMCA
    Inputs:
    - X: the observations, size m by t.
    - Oini: the outliers, size m by t.
    - Sini: sources, size n by t. Sini should be null to perform the complete process.
    - Aini: mixing matrix, size m by n.
    Outputs:
    - S: the estimated sources, size n by t.
    - A: estimated mixing matrix, size m by n.
    - O: estimated outliers, size m by t.
    '''
    
    O=Oini.copy()
    S=Sini.copy()
    A=Aini.copy()
    
    #Coefficients of the components in the dictionary in which S is sparse
    Sws=Stransf(S)
    
    #Coefficients of the components in the dictionary in which O is sparse
    Owo=Otransf(O)
    Swo=Otransf(S)
    Xwo=Otransf(X)
    
    if np.linalg.norm(S)>0: #If the sources are not null: refinement.
        indexW=1
    else:
        indexW=0
    

    madS=madAxis(A.T.dot(Stransf(X-A.dot(S)-O)), axis=1)
    madO=madAxis(Otransf(X-A.dot(S)-O))
    thr_O=np.sqrt(2)*aS['kOMax']*madO*sp.special.gamma(dS['m']/2.+0.5)/sp.special.gamma(dS['m']/2.)

    if  indexW==0:
        iteRew=3 # If the estimation starts from scratch: reweighting
    else:
        iteRew=2 #do not perform several loops if refinement
   
    convW=1 #convergence of S for reweighting 
    
    
    ########### Reweighting Procedure #########    
    
    while (indexW<iteRew and convW>1e-2  ) :  
        
        SW=S.copy()#copy of S for reweighting convergence
        if indexW==0 :#if the sources were null, do not update the weights
            WS=np.ones((np.shape(Sws)))
            WO=np.ones((np.shape(Owo)))
        else:
#          
           WS=np.maximum(madS*aS['kSMax'],0)/(madS*aS['kSMax']+np.abs(Sws))
           WO=np.maximum(thr_O,0)/(thr_O+np.linalg.norm(Owo,axis=0))

        indexW+=1
        
        convApalm=1#Convergence of A during the PALM
        iPalm=0 #iterations Palm
        iteMax=100000 #Maximal number of iterations for PALM
        convOS=1 #Convergence of O and S during PALM


        ########### Palm #########    

        while ((convOS>1e-5 or convApalm>1e-4)and iPalm<iteMax) or iPalm<5: #Stop the algorithm when O, S and A are stable
            iPalm+=1
            #Copies of the variables for checking the convergence
            A_=A.copy()
            S_=S.copy()
            O_=O.copy()
            
            if convOS>1e-4 and iPalm<50000: #Update the parameters whenever we are not stable. The parameters are then fixed for convergence
                madS=madAxis(A.T.dot(Stransf(X-A.dot(S)-O)), axis=1)
                madO=madAxis(Otransf(X-A.dot(S)-O))

#                        
            thr_O=np.sqrt(2)*aS['kOMax']*madO*sp.special.gamma(dS['m']/2.+0.5)/sp.special.gamma(dS['m']/2.)
           
            #####Update of S ####
            L=np.linalg.norm((A.T.dot(A)),2)
            S=Sback(softThres(Stransf(S+1./L*A.T.dot(X-A.dot(S)-O)),madS*aS['kSMax']*1./L*WS,1))
            Swo=Otransf(S)
            Sws=Stransf(S)

            ###Update of O ####
            normO=np.linalg.norm(Xwo-A.dot(Swo)+1e-32,axis=0)
            Owo=(Xwo-A.dot(Swo))*(np.maximum(1.-thr_O*WO/normO,0))
            O=Oback(Owo)
            
            
            ###Update of A####
            L=np.real(np.max(np.linalg.eigvals(S.dot(S.T))))
            A=A+1./L*(X-O-A.dot(S)).dot(S.T)
            A=A/np.maximum(1, np.linalg.norm(A, axis=0))

             
            ### Metrics for convergence ### 
            convApalm=np.sum(np.abs(np.linalg.pinv(A).dot(A_)-np.eye(dS['n'])))/(dS['n']**2)
            convOS=np.linalg.norm(S-S_, 'fro')/np.linalg.norm(S_, 'fro')+ np.linalg.norm(O-O_, 'fro')/np.linalg.norm(O_, 'fro')
        
            

        if iPalm>iteMax-10:
            print 'PALM has not converge'
        
        ### Metrics for convergence for the reweighting #########
        convW=np.linalg.norm(SW-S)/np.linalg.norm(S)


    return S, A, O
    
#####################################################################################
def AMCA(Xini,Aini, aMCA):
    '''Perform GMCA or AMCA (sparse BSS) on the observations Xini, J.Bobin et al., Sparsity and Adaptivity for the Blind Separation of Partially Correlated Sources.
        
    Inputs:
    - Xini: the observations, m by t matrix. Xini corresponds to the observations expressed in a dictionary in which the sources to be estimated are sparse.
    - Aini: the initial mixing matrix, m by n matrix
    - aMCA: if aMCA=1, then AMCA is performed; if aMCA=0, GMCA is performed. 
    Outputs:
    - S, the estimated sources, matrix of size n by t
    - A, the estimated mixing matrix, size m by t
    '''

    X=copy(Xini)
    A=copy(Aini)    
    S=np.zeros((dS['n'],dS['t']))
    
    #Initialize the weights for AMCA, with the largest entries.
    W = 1./np.linalg.norm(X+1e-10,axis=0,ord=1)**2;    
    W/=np.max(W)
    
    if aMCA==0:
        W=np.ones((dS['t'])) # Set the weights to 1 if GMCA is performed
        
    kend = aS['kSMax'] # Final thresholds of the sources: kend * sigma (k-mad)
  
    
    nmax=np.float(aS['iteMaxXMCA'])#Number of loops



    kIni=10. #Starting value for the threshold, k-mad
    dk=(kend-kIni)/nmax#Decrease of the k, for the kmad
    perc = 1./nmax
 
 
 
    ##### Start of GMCA/AMCA####
    it=0
    while it<nmax:#Stop when the maximal number of iterations is reached
        it += 1
        
    
    
        #######   Estimation of the sources ###
        sigA = np.sum(A*A,axis=0)
        indS = np.where(sigA > 0)[0]
        if np.size(indS) > 0: 
            Ra = np.dot(A[:,indS].T,A[:,indS])  
            Ua,Sa,Va = np.linalg.svd(Ra)
            cd_Ra = np.min(Sa)/np.max(Sa)

            ###Least squares estimate####
            if cd_Ra > 1e-5: #If A has a moderate condition number, performs the least squares with pseudo-inverse
                iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
                piA = np.dot(iRa,A[:,indS].T)    
                S[indS,:] = np.dot(piA,X)
 
            else:   #If A has a large condition number, update S with an 'incomplete' gradient descent                
                La = np.max(Sa)
                for it_A in range(250):
                    S[indS,:] = S[indS,:] + 1/La*np.dot(A[:,indS].T,X - np.dot(A[:,indS],S[indS,:]))
              
            Stemp = S[indS,:] 
            
            ###thresholding####
            for r in range(np.size(indS)):
                St = Stemp[r,:]
                indNZ=np.where(abs(St) > ((kIni+it*dk   )*madAxis(St)))[0]#only consider the entries larger than k-mad
                if len(indNZ)<dS['n']:
                    indNZ=np.where(abs(St)>=np.percentile(np.abs(St), 100.*(1-np.float(dS['n'])/dS['t'])))[0]
                Kval = np.min([np.floor(perc*(it)*len(indNZ)),dS['t']-1.])
                I = (abs(St[indNZ])).argsort()[::-1]
                Kval = np.int(min(max(Kval,dS['n']),len(I)-1))
                thrd=abs(St[indNZ[I[Kval]]])# threshold based on the percentile of entries larger than k-mad
                St[abs(St)<thrd]=0
                indNZ = np.where(abs(St) > thrd)[0]
                St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ]) #l1 thresholding
                Stemp[r,:] = St 
                
            S[indS,:] = Stemp
            Sref=copy(S)

                
        ####### Weights Update ####  
        if aMCA==1  and it>1: #Weights update for AMCA
                      
                        alpha=0.1**((it-1.)/(nmax-1.))/2.#p- of the lp norm of the weights
                        Ns = np.sqrt(np.sum(Sref*Sref,axis=1))
                        IndS = np.where(Ns > 0)[0] 
                        if len(IndS)>0:
                            Sref[IndS,:] = np.dot(np.diag(1./Ns[IndS]),Sref[IndS,:]) #normalized sources
                            W = np.power(np.sum(np.power(abs(Sref[IndS,:]),alpha),axis=0),1./alpha)
                            ind = np.where(W > 0)[0]
                            jind = np.where(W == 0)[0]
                            W[ind] = 1./W[ind];   
                            W/=np.max(W[ind])
                            if len(jind) > 0:
                                W[jind] = 1
                                
                            W/=np.max(W)#Weights



        #### Update of A #########
        Ns = np.sqrt(np.sum(S*S,axis=1))
        indA = np.where(Ns > 0)[0]
        if len(indA) > 0:
            Sr = copy(S)*W # weighted sources
            Rs = np.dot(S[indA,:],Sr[indA,:].T)
            Us,Ss,Vs = np.linalg.svd(Rs)
            cd_Rs = np.min(Ss)/np.max(Ss)
            if cd_Rs > 1e-4: # if the sources have a fair condition number, use the pseudo-inverse
                piS = np.dot(Sr[indA,:].T,np.linalg.inv(Rs));
                A[:,indA] = np.dot(X,piS)
                A = np.dot(A,np.diag(1./(1e-24 + np.sqrt(np.sum(A*A,axis=0)))));
            else:#if the condition number of the projected sources is too large, do an 'interrupted' gradient descent
                Ls = np.max(Ss)
                indexSub=0
                while indexSub<250:
                    A[:,indA] = A[:,indA] + 1/Ls*np.dot(X - np.dot(A[:,indA],S[indA,:]),Sr[indA,:].T)
                    A[:,indA] = np.dot(A[:,indA],np.diag(1./(1e-24 + np.sqrt(np.sum(A[:,indA]*A[:,indA],axis=0)))));
                    indexSub+=1
      
    return S,A
    
def MCA(Xini):
    ''' MCA algorithm, J.L. Starck et al, Redundant Multiscale Transforms and their application for Morphological Component Analysis.
    Performs the separation between AS and O (Xini=AS+O), sparsely represented in two different dictionaries.
    For this purpose, the components are transformed with Stransf/Sback, Otransf/Oback.
    Input:
    -Xini: the observations, m by t matrix (direct domain).
    Outputs:
    - AS: matrix of size m by t, sparsely represented with Stransf.
    - O: second component, matrix of size m by t, sparsely represented with Otransf. 
    '''
    ### NB: this implementation can be improved - see for instance J.Bobin et al, Morphological Component Analysis: An Adaptive Thresholding Strategy.
    #The setting of the parameters are handled with the l1 and decreasing thresholding and then l0.
    
 
    AS=np.zeros(np.shape(Xini)) #First component
    O=np.zeros(np.shape(Xini))#Second Component
    
    for j in range(dS['m']): # MCA is performed on every channel
        Oloc=np.zeros(( 1,dS['t'])) #row of O, size 1 by t
        ASloc=np.zeros((1, dS['t']))#row of AS, size 1 by t
        X=Xini[j,:].reshape((1,dS['t']))# one observation, size 1  by t
        
        index=0
        Max=100
        while  index<Max :#Stop after Max iterations
            index+=1

            thrO=aS['kOMax']*madAxis(Otransf(X-ASloc-Oloc)) #Only consider the entries of Oloc larger than k-mad
            thrS=aS['kSMax']*madAxis(Stransf(X-ASloc-Oloc)) # Only consider the entries of ASloc larger than k-mad
            
            Oloc=Otransf(X-ASloc)#Oloc before thresholding
            if np.linalg.norm(Oloc[np.abs(Oloc)>thrO])>0:
                thrO=np.percentile(np.abs(Oloc[np.abs(Oloc)>thrO]), 100*(1.-index/np.float(Max))) #Consider an increasing number of entries, larger than k-mad
                Oloc=Oback(softThres(Oloc, thrO, 1))#l1 thresholding
            ASloc=Stransf(X-Oloc)#ASloc before thresholding
            if np.linalg.norm(ASloc[np.abs(ASloc)>thrS])>0:
                thrS=np.percentile(np.abs(ASloc[np.abs(ASloc)>thrS]), 100*(1.-index/np.float(Max)))#Threshold update
                ASloc=Sback(softThres(ASloc, thrS, 1))#l1

        ###Refine with the l0
        conv=1           
        index=0
        Max=100
        while  index<Max or conv>1e-6:#Stop after Max iterations or convergence
            index+=1
            ASloc_=ASloc.copy()
            Oloc_=Oloc.copy()

            #Final Threshold, k-mad
            thrO=aS['kOMax']*madAxis(Otransf(X-ASloc-Oloc))
            thrS=aS['kSMax']*madAxis(Stransf(X-ASloc-Oloc))
            
            
            Oloc=Otransf(X-ASloc)
            Oloc=Oback(softThres(Oloc, thrO, 0)) #Hard-threshoding
            
            ASloc=Stransf(X-Oloc)
            ASloc=Sback(softThres(ASloc, thrS, 0))#Soft Thresholding

            conv=np.linalg.norm(ASloc-ASloc_)/np.linalg.norm(ASloc)+np.linalg.norm(Oloc-Oloc_)/np.linalg.norm(Oloc) #convergence metric
           
        AS[j,:]=ASloc[0,:]
        O[j,:]=Oloc[0,:]
        
    return AS, O
    
    
