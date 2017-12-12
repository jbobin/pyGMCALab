# -*- coding: utf-8 -*-
"""
Computes the sparse BSS algorithms:
- the modified version of AMCA, initially presented in 'Sparsity and adaptivity for the blind separation of partially correlated sources', J.Bobin et al.
which can also implement GMCA presented in 'Sparsity and morphological diversity in blind source separation' , J. Bobin et al. TIP 2007.
The initial script for AMCA can be found in pyGMCALab http://www.cosmostat.org/software/gmcalab/
- rGMCA, presented in 'Robust Sparse Blind Source Separation', C. Chenot et al. SPL 2015
-rAMCA, presented in 'Unsupervised separation of sparse sources in the presence of outliers', C. Chenot and J. Bobin.
"""
#import#
import numpy as np
from parameters import aS, dS, pS
from copy import deepcopy, copy
from errors import reOrder
import scipy.special
import data

###Dictionary containing the value for the alpha parameter for the delta-density, according the number of sources###
WminD={}
WminD['2']=1.4
WminD['4']=2.1
WminD['6']=2.7
WminD['8']=3.3
WminD['10']=3.9
WminD['12']=4.5
WminD['14']=5
WminD['16']=5.6


#####################################################################################
def rAMCA(Xini,A,amca=1):
    '''Computes the rAMCA algorithm. It estimates A, S and O jointly. It estimates the outliers
    based on their norm and delta-density, and estimates A and S with AMCA wich penalizes the corrupted samples.
    Input:
    Xini: Observations, represented in a domain in which both the sources and the outliers are sparse.
    A: initializing mixing matrix
    amca: boolean. If it is set to one, AMCA is performed, otherwise, GMCA is performed in the inner loop to estimate A and S. (Should be one)
    Output:
    S: estimated sparse sources
    A: estimated mixing matrix
    O: estimated sparse outliers
    '''

    #Initialize the variables and parameters

    X=copy(Xini)
    A=copy(A)
    O=np.zeros(np.shape(Xini))


    itMAX=aS['iteMaxrAMCA']
    IndexTot=0
    convA=10 #stop when A has converged


    densityMin=WminD[str(dS['n'])]#Maximal value of the delta-density for not being considered as corrupted
    gammaValue=np.sqrt(2)*scipy.special.gamma((dS['m']+1.)/2.)/scipy.special.gamma((dS['m'])/2.) #For the l_2,1 norm estimation of the outliers

    perc=5 #percentage of outliers newly considered at every iteration (can be smaller to avoid false estimation of outliers)
    tperc=perc/100.*dS['t']


    suppOest=np.zeros((dS['t'],)) #Support of the estimated outliers
    new=np.zeros((dS['t'],)) # Newly estimated outliers



    #First estimation of A and S
    S,A=AMCA(X-O,A,amca)

    #Start of the main loop
    while (IndexTot<itMAX  and (np.sum(new)>0 or convA>5)): #Stop if A has converged and no outlier has been estimated


       #########################################
        #Update the support of the outliers
       #########################################
        #Update the delta density of the estimated sources
        deltaDen=np.linalg.norm(S,axis=0,ord=1.)/np.max(np.abs(S)+1e-6,axis=0)
        #update the l2 norm of the columns
        normX=np.linalg.norm(X-A.dot(S)-O,axis=0)

        if np.sum(deltaDen>densityMin)>0:

            deltaDen[deltaDen<densityMin]=0 #The sparse samples are not considered
            normX[deltaDen==0]=0
            minSuppX=np.percentile(normX[normX>0],max((np.sum(normX>0)-tperc)/np.sum(normX>0)*100,0)) #Lower value of the norm of the admissible corrupted samples

            if (np.sum(new)==0 and convA<5): #If it has nearly converged, decreases the value of minSuppX to avoid local minima and bias
                minSuppX=mad(X-A.dot(S)-O)*gammaValue


            new=np.array(normX>minSuppX,int) #Support of the samples corresponding to the largest entries in the residue and whose delta-density is large enough

            new[new*suppOest>0]=0 # Take only the new oultiers#
            suppOest+=np.array(new,int) #Add the new outliers
            suppOest[suppOest>0]=1 # support of the outliers#

            #Sanity Check to avoid the false estimation of outliers. It is not necessary if the initial matrix A has a small condition number (less than 50), and if the parameter perc is small enough.
            #As the outliers are broadly distributed, its m eigenvalues should be centered around a same value.
            #If a source has been wrongly estimated as being outliers, then one of the eigenvalue will be much more larger than the other one (the energy of the sources has been added up to the one of the outliers)
            #If the difference between the largest eigenvalue and the mean of the other eigenvalues is equal to five time the standard deviation of the m-1 smallest eigenvalues, then,
            # the samples contributing to this eigenvalue are not considered as being outliers.
            if np.sum(new)>max(0.005*dS['t'],dS['m']) : #if new outliers have been deteced#
                            Onew=(X/(np.linalg.norm(X,axis=0)))[:,(suppOest>0)] #corresponding columns in X, normalized to avoid scaling issue in the PCA (if the sources amplitude is much smaller than the outliers amplitude, they may not be detected otherwise)
                            u,s,v=np.linalg.svd(Onew,full_matrices=True)#Compute the svd, can do better than that
                            if (s[0]-np.mean(s[1::]))/(np.var(s[1::])**0.5)>5: #if an eigenvalue is anormaly large, a source is detected
                                Onew_=copy(Onew)
                                Os=s[0]*(u[:,0].reshape((dS['m'],1)).dot(v[0,:].reshape((1,np.size(v[0,:]))))) #outliers contribution corresponding to this eigenvalue

                                Onew=Onew_ - Os # outliers contribution on the m-1 remaining dimensions
                                suppOest[np.where(suppOest>0)[0][np.linalg.norm(Onew,axis=0)< np.linalg.norm(Os,axis=0)]]=0 #release of the samples whose projection on the m-1 remaining dimensions has a smaller energy than on the axis corresponding to the largest eigenvalue

        else:
            new=0*new#no outlier


        #########################################
        #Update the amplitudes of the outliers
       #########################################
        R=X
        beta=mad(X-A.dot(S)-O)*gammaValue #threshold value of the l_2,1 norm
        O[:,suppOest>0]=((R)[:,suppOest>0])* (np.maximum(1-beta/(np.linalg.norm((R)[:,suppOest>0],axis=0)),0))


        #########################################
        #Update S and A with AMCA
       #########################################
        A_=copy(A)#A before estimation
        A=np.random.randn(dS['m'],dS['n']) #randomize the initialization point
        A/=np.linalg.norm(A,axis=0)

        S,A=AMCA(X-O,A,amca)#Estimation of S, A

        S,A=reOrder(A,S,A_,S) # reorder the factorization  (to compute the convergence metric)
        angle=np.arccos(np.abs(np.sum(A_*A,axis=0)))*180/np.pi
        convA=np.max(angle) #A has converged if the maximal angle is less than 5


        IndexTot+=1
    if pS['verboserAMCA']:
        print 'rAMCA terminates in ', IndexTot, ' iterations'

    return S,A,O

######################################################

def rGMCA(Xini,Aini):
    '''
    Performs the rGMCA algorithm.It estimates jointly O,S and A. It further implements a weighting scheme which penalizes the
    entries corrupted by outliers.
    Input:
    Xini: the observations, represented in a domain in which the sources and outliers have a sparse reprentation.
    Output:
    S: the estimated sparse sources
    A: the estimatedmixing matrix
    O: the estimated sparse outliers
    '''

    #Initialization
    X=copy(Xini)
    A=copy(Aini)
    O=np.zeros((dS['m'],dS['t']))
    S=np.zeros((dS['n'],dS['t']))

    #########################################
        #Initialization
    #########################################
    #First outliers estimation
    Xabs=np.array(np.abs(X))
    sigmaN=np.median(Xabs)#can do better than that
    iniO=(np.percentile(Xabs[Xabs>sigmaN],50))#Threshold for the first estimation of the outliers

    O=softThres(X-A.dot(S),iniO)#First estimation of the outliers

    while np.linalg.cond(S)>1e4 or np.linalg.cond(A)>1e4 or np.linalg.norm(S)==0:#first estimation of the sources
        S=np.array(np.abs(np.linalg.pinv(A).dot(X-O)))
        ini=np.array(np.percentile(S,99,1)).reshape((dS['n'],1))#threshold for each source
        S=softThres(S,ini)


        while np.linalg.cond(S)>1e4 or np.linalg.cond(A)>1e4 or np.linalg.norm(S)==0:#estimation of the mixing matrix
            A=data.mixingMatrix(dS['m'],dS['n'])
            S=np.array(np.abs(np.linalg.pinv(A).dot(X-O)))
            tot=99
            ini=np.array(np.percentile(S,tot,1)).reshape((dS['n'],1))
            S=softThres(S,ini)

            A=(X-O).dot(np.linalg.pinv(S))
            A/=np.linalg.norm(A,axis=0)



    #The outliers will be thresholded by kO*mad(X-AS-O), where k decreases toward 3
    kOIni= max(iniO,10)/mad(X-A.dot(S)-O)*np.ones((dS['m'],1)) #initial kO

    #########################################
        #Joint estimation
    #########################################
    indexIte=0
    convA=10
    while (convA>1 and indexIte<aS['iteMaxrGMCA']) or indexIte<5:
        A_=A
        S_=S
        # Decrease k#
        kO=kOIni+(aS['kOMax']*np.ones((dS['m'],1))-kOIni)*np.float((indexIte))/(aS['iteMaxrGMCA']-1)

        W=1./(np.median(np.abs(S[np.abs(S)>0]))/10.+np.linalg.norm(O,axis=0,ord=1)) #Weight for the estimation of A

        S,A=AMCA(X-O,A,amca=0,WO=W)#Estimate A and S

        O=softThres(X-A.dot(S),kO*np.median(mad(X-A.dot(S)-O)))#update of O

        S,A=reOrder(A,S,A_,S_)#reorder the factorization to compute the convergence metric
        angle=np.arccos(np.abs(np.sum(A_*A,axis=0)))*180/np.pi
        convA= np.max(angle)
        indexIte+=1

    if pS['verboserGMCA']:
        print 'rGMCA terminates in ', indexIte, ' iterations'


    return S,A,O

######################################################


def AMCA(Xini,Aini, amca,WO=np.ones(dS['t'],)):
    ''' Performs the AMCA/GMCA algorithm: factorize the observations Xini into a product A*S, where S are sparse sources. If AMCA is performed,
    a weighting scheme is implemented to penalize the corrupted entries.
    If the boolean amca is set to one, AMCA is performed, otherwise GMCA is performed (no penalizating procedure).
    Input:
    Xini: the observations on which the factorization is performed (represented in a domain in which the sources are sparse)
    Aini: initial mixing matrix
    amca: boolean. If set to 1: AMCA is performed. Otherwise, GMCA is performed
    WO: weights for rGMCA (should be combined with amca=0)
    Output:
    S: the estimated sparse sources
    A: the mixing matrix, whose columns are normalized to one.
    '''

    #Initialization of the variables
    X=Xini.copy() # observations
    A=Aini.copy()  # mixing matrix
    S=np.zeros((dS['n'],dS['t'])) # Sources

    #Initial weights
    W = 1./np.linalg.norm(X,axis=0,ord=1)**2;#Penalize the largest samples at the first iteration
    W/=np.max(W)
    W1=np.ones((dS['t'],))# Set to identity. May not be used

    nmax=np.float(aS['iteMaxAMCA'])# Number of iterations of AMCA
    if amca==0:
        W=WO #Set to 1 the weights for GMCA and to WO for rGMCA
        nmax=np.float(aS['iteMaxGMCA'])# Number of iterations of GMCA


    #Only consider the samples of the sources whose magnitude is larger than k*mad(N), mad(N) corresponds to the estimate of the standard deviation of the Gaussian noise corrupting the sources.
    kend = aS['kSMax']    #Final threshold for the sources in kend*mad(N)
    kIni=10     # Starting value of k
    dk=(kend-kIni)/nmax # value of the decrease of k (linear)



    #Main loop
    Go_On = 1
    it = 0
    while Go_On==1:
        it += 1
        if it == nmax:
            Go_On = 0


    #########################################
        #Estimation of the sources
    #########################################

        sigA = np.sum(A*A,axis=0)
        indS = np.where(sigA > 0)[0]
        if np.size(indS) > 0: # if some columns are non-zero

            ### Sources before threshold: S= pseudoInverse(A).X ###
            Ra = np.dot(A[:,indS].T,A[:,indS])
            Ua,Sa,Va = np.linalg.svd(Ra) #Svd of A.T*A (to quicken the svd)
            cd_Ra = np.min(Sa)/np.max(Sa) #condition number of A.T.A

            if cd_Ra > 1e-5: #If the condition number of A.T.A is not to large
                iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
                piA = np.dot(iRa,A[:,indS].T)    #Compute the pseudo inverse
                S[indS,:] = np.dot(piA,X) # Sources before thresholding

            else:  #otherwise, do not completly compute the inversion
                La = np.max(Sa) # Descent step
                for it_A in range(0,250):
                    S[indS,:] = S[indS,:] + 1/La*np.dot(A[:,indS].T,X - np.dot(A[:,indS],S[indS,:]))

            ### Thresholding of each source independently ###
                #Only the entries largest than k*sigma are considered.
                # An increasing percentage of entries (largest than k*sigma) are kept
                # The value of the threshold is chosen according to this percentage.

            Stemp = S[indS,:]
            for r in range(np.size(indS)):

                St = Stemp[r,:] # Source before threshold
                indNZ=np.where(abs(St) > ((kIni+it*dk   )*mad(St)))[0] # only keep the entries largest than k*mad(St), where mad(St) should correspond to an estimate of the standard deviation of the Gaussian noise contribution on this source (as the source is sparse)

                if len(indNZ)<dS['n']:# To be sure that at least n entries per source are active
                    indNZ=np.where(abs(St)>=np.percentile(np.abs(St), 100.*(1-np.float(dS['n'])/dS['t'])))[0]

                Kval = np.min([np.floor(1./(nmax-1)*(it)*len(indNZ)),dS['t']-1.]) # the number of active entries in the source
                I = (abs(St[indNZ]*W1[indNZ])).argsort()[::-1] # sorts the entries of S*W1 (W1 is the identity for GMCA)
                Kval = np.int(min(max(Kval,dS['n']),len(I)-1)) # the Kval th largest value
                thrd=abs(St[indNZ[I[Kval]]]) # threshold for the sources with the Kval th largest entries (can be smaller than the Kval th if W1 is not the identity)

                #Soft-thresholding of the sources
                St[abs(St)<thrd]=0
                indNZ = np.where(abs(St) > thrd)[0]
                St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ])
                Stemp[r,:] = St

            S[indS,:] = Stemp #Estimated sources



    #########################################
        #Updates of the weights
    #########################################
               # The weights W are used for the estimation of A, to penalize the non-sparse entries#
               # The weights W1 are used for updating the threshold values (by trying to add non-correlated entries)

        Sref=copy(S)
        if amca==1  and it>1:
                        alpha = 0.1**((it-1.)/(nmax-1.))/2. #the weights are computed with the l_alpha norm, where alpha is decreasing
                        Ns = np.sqrt(np.sum(Sref*Sref,axis=1))
                        IndS = np.where(Ns > 0)[0]

                        if len(IndS)>0:
                            Sref[IndS,:] = np.dot(np.diag(1./Ns[IndS]),Sref[IndS,:]) #normalize the sources
                            W = np.power(np.sum(np.power(abs(Sref[IndS,:]),alpha),axis=0),1./alpha) # compute the l_alpha norm of the sources (columns)
                            ind = np.where(W > 0)[0]
                            jind = np.where(W == 0)[0]
                            W[ind] = 1./W[ind];   #inverse the norm
                            W/=np.max(W[ind])
                            if len(jind) > 0:
                                W[jind] = 1
                            W/=np.max(W) #set the maximal value to 1

                            #The following weights may not be necessary and/or we can find a smarter procedure.
                            # The weights of W1 help to introduce sparse samples in the estimation of the sources#
                            # The samples which are non-sparse are given a weight smaller than 1. That is why, by ordering W1*S when computing the threshold,
                            #we can add sparse entries by decreasing the threshold.
                            #If all the entries are non-sparse or if they are all sparse, this has no effect.
                            # The weights correspond to the ratio l1 over l2 for each sample.
                            IndS = np.where(np.sum(S*S,axis=0) > 0)[0]
                            W1=np.ones((dS['t'],))
                            W1[IndS] = np.power(np.sum(np.power(abs(X[:,IndS]),2),axis=0),1./2)/np.power(np.sum(np.power(abs(S[:,IndS]),1),axis=0),1./1)# ratio l2 over l1
                            ind = np.where(W1 > 0)[0]
                            jind = np.where(W1 == 0)[0]
                            W1[ind] = 1./W1[ind];   #inverse the ratio of the non-zeros samples
                            W1/=np.max(W1[ind])
                            if len(jind) > 0:
                                W1[jind] = np.max(W1[ind])
                                W1 /= np.max(W1)


    #########################################
        #Updates of the mixing matrix
    #########################################

        Ns = np.sqrt(np.sum(S*S,axis=1))
        indA = np.where(Ns > 0)[0]
        if len(indA) > 0:
            Sr = deepcopy(S)*W
            Rs = np.dot(S[indA,:],Sr[indA,:].T)
            Us,Ss,Vs = np.linalg.svd(Rs) #computes the svd on S.W.S.T to quicken the process
            cd_Rs = np.min(Ss)/np.max(Ss)# condition number
            if cd_Rs > 1e-10:# if the condition number is not too large, compute directly the pseudo inverse
                piS = np.dot(Sr[indA,:].T,np.linalg.inv(Rs));
                A[:,indA] = np.dot(X,piS)
                A = np.dot(A,np.diag(1./(1e-24 + np.sqrt(np.sum(A*A,axis=0)))));#normalize the columns
            else:
                #--- iterative update
                Ls = np.max(Ss)
                indexSub=0
                while indexSub<250:
                    A[:,indA] = A[:,indA] + 1/Ls*np.dot(X - np.dot(A[:,indA],S[indA,:]),Sr[indA,:].T)
                    A[:,indA] = np.dot(A[:,indA],np.diag(1./(1e-24 + np.sqrt(np.sum(A[:,indA]*A[:,indA],axis=0)))));
                    indexSub+=1

    return S,A
#

#
######################################################

def softThres(X,thres):
    '''
    Soft-thresolding operator
    Input:
    X: matrix to be thresholded
    thres: threshold
    Output: X, soft-thresholded by thres
    '''
    return np.maximum(np.abs(X)-thres,0)*np.sign(X)


#####################################################################################

def madAx(X):
    '''Compute the mad of each row of a matrix
    Implementation from J. Rapin
    Input:
    X: input matrix
    output: column vector outvals corresponding to the mad of each row of X
    '''

    dim=1
    sizes = np.array(X.shape);
    sizes[dim] =1;
    outvals = np.median(X, axis = dim).reshape(sizes);
    outvals = 1.4826 * np.median(np.abs(X - outvals), axis = dim).reshape(sizes);

    return outvals



#####################################################################################

def mad(xin = 0):
    '''
     Compute the median absolute deviation of a matrix
     Input:
     xin: inmput matrix
     Output: mad of x
     '''
    z = np.median(abs(xin - np.median(xin)))/0.6735

    return z
