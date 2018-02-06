#  Version
#    v2 - September,28 2017 - C. Kervazo and J.Bobin - CEA Saclay


import numpy as np
import scipy.linalg as lng
import copy as cp
import time
from copy import deepcopy
from pyGMCA.common.wavelets import pyStarlet as pys
from pyGMCA.common.prox import Prox_SparsePositive as psp

################# RANDOM PERMUTATION

def randperm(n=1,notIn=[],numberInd=-1):

    import numpy as np

    if numberInd == -1:
        numberInd = n

    X = np.random.randn(n)
    I = X.argsort()

    return I


################# DEFINES THE MEDIAN ABSOLUTE DEVIATION

def mad(xin = 0):

    import numpy as np

    z = np.median(abs(xin - np.median(xin)))/0.6735

    return z

################# AMCA Main function

def bGMCA(X,n,mints=3,nmax=100,L0=0,verb=0,Init=0,blocksize=2,optBlock=0,J=0,optPos=0):
    '''Usage:
        S,A,exception = bGMCA(X,n,maxts = 0,mints=3,nmax=100,L0=0,verb=0,Init=0,blocksize=2,optBlock=0,J=0,optPos=0)


      Inputs:
       X    : m x t array (input data, each row corresponds to a single observation)
       n     : scalar (number of sources to be estimated)
       mints : scalar (final value of the k-mad thresholding)
       nmax  : scalar (number of iterations)
       L0   : boolean (0: use of the L1 norm, 1: use of the L0 norm)
       verb   : boolean (if set to 1, in verbose mode)
       Init  : scalar (if set to 0: PCA-based initialization, if set to 1: random initialization)
       blocksize : int (size of the blocks)
       optBlock: int in {0,1,2,3} (way the block is constructed: 0 : Random creation of the block, 1 : Correlation, 2 : Loop over all the sources + take the sources that are the most correlated with the current one, 3 : We create the batchs sequentially)
       J : int (number of wavelet scales. If J = 0, no wavelet transform)
       optPos : int in {0,1,2,3} (option for the non-negativity: 0 no constraint, 1 : non-negativity on A and S, 2: non-negativity on S, 3: non-negativity on A)

      Outputs:
       S    : n x t array (estimated sources)
       A    : m x n array (estimated mixing matrix)
       exception : boolean (equals to 1 is a SVD did not converge during the iterations)

      Description:
        Computes the sources and the mixing matrix with bGMCA using blocks. Performs the initialization of A and S

      Example:
         S,A,exception = bGMCA(X,50,mints=3,nmax=10000,L0=0,verb=1,Init=1,blocksize=5,optBlock=0,J=2,optPos=0)'''

    nX = np.shape(X);
    m = nX[0];t = nX[1]
    Xw = cp.copy(X)

    #Initialization could largely be improved -> it should be data-dependent

    if verb:
        print("Initializing ...")

    if Init == 0:
        R = np.dot(Xw,Xw.T)
        D,V = lng.eig(R)
        A = V[:,0:n]

    if Init == 1:
        A = np.random.randn(m,n)

    A = np.dot(A,np.diag(1./np.sqrt(np.sum(A**2,axis=0))))

    if J > 0:
        S = np.zeros((n,t*J));
    else:
        S = np.zeros((n,t));


    if J > 0:
        S,A,exception,Sw = bGMCA_MainBlock(Xw,n,A,S,kend = mints,nmax=nmax,L0=L0,verb=verb,blocksize=blocksize,optBlock=optBlock,J=J,optPos=optPos);

        return S,A,exception,Sw

    else:
        S,A,exception = bGMCA_MainBlock(Xw,n,A,S,kend = mints,nmax=nmax,L0=L0,verb=verb,blocksize=blocksize,optBlock=optBlock,J=J,optPos=optPos);

        return S,A,exception

################# AMCA internal code (Weighting the sources only)

def bGMCA_MainBlock(X=0,n=0,A=0,S=0,kend=3,nmax=100,L0=1,verb=0,blocksize=2,tol=1e-12,optBlock=0,J=0,optPos = 0):
    '''Usage:
        S,A,exception,Sw = bGMCA_MainBlock(X=0,n=0,A=0,S=0,kend=3,nmax=100,L0=1,verb=0,blocksize=2,tol=1e-12,optBlock=0,J=0,optPos = 0)


      Inputs:
       X    : m x t array (input data, each row corresponds to a single observation)
       n     : scalar (number of sources to be estimated)
       A    : m x n array (initialization of the mixing matrix)
       S    : n x t array (initialization of the sources)
       kend : scalar (final value of the k-mad thresholding)
       nmax  : scalar (number of iterations)
       L0   : boolean (0: use of the L1 norm, 1: use of the L0 norm)
       verb   : boolean (if set to 1, in verbose mode)
       blocksize : int (size of the blocks)
       tol : float (stopping criterion)
       optBlock: int in {0,1,2,3} (way the block is constructed: 0 : Random creation of the block, 1 : Correlation, 2 : Loop over all the sources + take the sources that are the most correlated with the current one, 3 : We create the blocks sequentially)
       J : int (number of wavelet scales. If J = 0, no wavelet transform)
       optPos : int in {0,1,2,3} option for the non-negativity: 0 no constraint, 1 : non-negativity on A and S, 2: non-negativity on S, 3: non-negativity on A

      Outputs:
       S    : n x t array (estimated sources)
       A    : m x n array (estimated mixing matrix)
       exception : boolean (equals to 1 is a SVD did not converge during the iterations)
       Sw : n x (t x j) array (wavelet transform of the sources)

      Description:
        Computes the sources and the mixing matrix with bGMCA using blocks. Does not perform the initialization of A and S

      Example:
        S,A,exception,Sw = bGMCA_MainBlock(Xw,50,A,S,kend = 3,nmax=10000,L0=0,verb=1,blocksize=10,optBlock=0,J=2,optPos=0)'''

    if J > 0: # Use of wavelets
        Xdir = cp.deepcopy(X)
        t = np.shape(Xdir)[1]
        Xw = pys.forward1d(X,J=J)
        n_Xw = np.shape(Xw)
        X = Xw[:,:,0:J].reshape(n_Xw[0],n_Xw[1]*J)

    #--- Initialization

    powCor = 0.5
    n_S = np.shape(S)
    exception = 0
    comptPasGrad = 0


    perc = 1./nmax




    Aold = deepcopy(A)

    deltaTabAngle = np.zeros(nmax + 1)






    if optPos in [1,2,3]:
        for ii in range(0,n_S[0]): # Initialization to ensure that the given matrices A and S are non negative
            Sl = S[ii,:]
            Al = A[:,ii]

            if optPos in [1,2]:
                indMax = np.argmax(abs(Sl))
                if Sl[indMax] < 0:
                    Sl = -Sl
                    Al = -Al

                Sl[Sl < 0] = 0
                if optPos == 1:
                    Al[Al < 0] = 0

                    Al = Al/np.linalg.norm(Al)



            elif optPos == 3:
                indMax = np.argmax(abs(Al))
                if Al[indMax] < 0:
                    Sl = -Sl
                    Al = -Al

                Al[Al < 0] = 0

                Al = Al/(1e-24 + np.linalg.norm(Al))

            S[ii,:] = Sl
            A[:,ii] = Al

    # Stopping/iteration variables

    Go_On = 1
    it = 1



    if verb:
        print("Starting main loop ...")
        print(" ")
        print("  - Final k: ",kend)
        print("  - Maximum number of iterations: ",nmax)
        print("  - Batch size: ",blocksize)
        print("  - Using support-based threshold estimation")
        if L0:
            print("  - Using L0 norm rather than L1")
        print(" ")
        print(" ... processing ...")
        start_time = time.time()

    # Defines the residual

    Resi = cp.deepcopy(X)

    #---
    #--- Main loop
    #---


    while Go_On:
        it += 1

        if it == nmax:
            Go_On = 0

        #--- Estimate the sources (only when the corresponding column is non-zeros)

        sigA = np.sqrt(np.sum(A*A,axis=0))
        indS = np.where(sigA > 1e-24)[0]

        if optBlock == 0:# Random creation of the block
            if blocksize < n_S[0]:
                IndBatch = randperm(len(indS))  #---- mini-batch amongst available sources
            else:
                IndBatch = range(len(indS))

            if blocksize < len(indS):
                indS = indS[IndBatch[0:blocksize]]

        elif optBlock == 1:# Correlation
            if blocksize < n_S[0]:
                currSrcInd = it % len(indS)
                currSrcInd = indS[currSrcInd]
                currSrc = np.power(S[currSrcInd,:].T,powCor)
                matCor = np.dot(np.power(S,powCor),currSrc)
                matCor[currSrcInd] = -1
                IndBatch = np.argsort(matCor)
                IndBatch = np.append(IndBatch,currSrcInd)
                IndBatch = IndBatch[::-1]

            else:
                IndBatch = range(len(indS))

            if blocksize < len(indS):
                if len(indS) == n_S[0]:
                    indS = indS[IndBatch[0:blocksize]]


                else:
                    print('Attention, cas special')

                    indSTemp = [currSrcInd]
                    ii = 0
                    while len(indSTemp) < blocksize:
                        ii = ii + 1
                        if IndBatch[ii] in indS:
                            indSTemp = indSTemp + [IndBatch[ii]]




        elif optBlock == 2:# Loop over all the sources + take the sources that are the most correlated with the current one
            if blocksize < n_S[0]:
                currSrcInd = it % len(indS)
                currSrcInd = indS[currSrcInd]
                angTab = np.dot(A[:,currSrcInd].T,A)
                angTab[currSrcInd] = -2 #0 : correspond a l'angle le plus grand, 90 degres
                IndBatch = np.argsort(angTab)
                IndBatch = np.append(IndBatch,currSrcInd)
                IndBatch = IndBatch[::-1]

            else:
                IndBatch = range(len(indS))

            if blocksize < len(indS):
                if len(indS) == n_S[0]:
                    indS = indS[IndBatch[0:blocksize]]
                else:
                    print('Attention, cas special')
                    indSTemp = [currSrcInd]
                    ii = 0
                    while len(indSTemp) < blocksize:
                        ii = ii + 1
                        if IndBatch[ii] in indS:
                            indSTemp = indSTemp + [IndBatch[ii]]



        elif optBlock == 3: # We create the batchs deterministically
            indStt = blocksize*(it-1) % np.shape(S)[0]
            indEnd = blocksize*it % np.shape(S)[0]

            if indStt < indEnd:
                IndBatch = range(int(indStt),int(indEnd))
            else:
                ind1 = range(indStt,np.shape(S)[0])
                if(indEnd > 0):
                    ind2 = range(0,indEnd)
                    IndBatch = np.concatenate((ind1,ind2))
                else:
                    IndBatch = ind1

            if blocksize < len(indS):
                if len(indS) == n_S[0]:
                    indS = indS[IndBatch[0:blocksize]]


                else:
                    print('Attention, cas special')
                    indSTemp = [indStt]
                    ii = 0
                    while len(indSTemp) < blocksize:
                        ii = ii + 1
                        if IndBatch[ii] in indS:
                            indSTemp = indSTemp + [IndBatch[ii]]






        Resi = Resi + np.dot(A[:,indS],S[indS,:])   # Putting back the sources



        if np.size(indS) > 0:

            if len(indS) > 1:

                # Least_square estimate

                Ra = np.dot(A[:,indS].T,A[:,indS])
                excThisIt = 0
                try:
                    Ua,Sa,Va = np.linalg.svd(Ra)
                except np.linalg.linalg.LinAlgError:
                    print(indS)
                    print(np.sum(np.isnan(Ra)))
                    print(np.sum(np.isinf(Ra)))
                    print('ATTENTION PAS DE CONVERGENCE DE LA SVD')
                    exception += 1
                    excThisIt = 1


                if excThisIt == 0:
                    cd_Ra = np.min(Sa)/np.max(Sa)
                else:
                    cd_Ra = np.linalg.norm(Sa,ord=-2)/np.linalg.norm(Sa,ord=2)

                if (cd_Ra > 1e-12) and not(excThisIt): # if the condition number is large enough
                    iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
                    piA = np.dot(iRa,A[:,indS].T)
                    S[indS,:] = np.dot(piA,Resi)

                if (cd_Ra < 1e-12 or excThisIt == 1) : # otherwise a few gradient descent steps
                    comptPasGrad += 1
                    print('GRADIENT STEP INSTEAD OF PSEUDO-INVERSE NUMBER %s' %comptPasGrad)
                    if excThisIt == 0:
                        La = np.max(Sa)
                    else:
                        La = np.linalg.norm(Ra,ord=2)
                        print('WARNING SVD DID NOT CONVERGE AT THIS ITERATION')

                    for it_A in range(0,10):
                        S[indS,:] = S[indS,:] + 1/La*np.dot(A[:,indS].T,X - np.dot(A[:,indS],S[indS,:]))

            else:  # only a single element in the selection / the update is straightforward

                S[indS[0],:] = np.dot(1./np.linalg.norm(A[:,indS[0]])*A[:,indS[0]],Resi)

            # Thresholding

            Stemp = S[indS,:]


            for r in range(len(indS)): # Thresholding
                St = Stemp[r,:]
                indNZ = np.where(abs(St) > kend*mad(St))[0]
                thrd = mad(St[indNZ])

                Kval = np.min([np.floor(np.max([0.01,perc*it])*len(indNZ)),n_S[1]-1.]) # Includes an increasing percentage of available entries
                I = abs(St[indNZ]).argsort()[::-1]
                Kval = np.int(np.min([np.max([Kval,n]),len(I)-1.]))
                IndIX = np.int(indNZ[I[Kval]])
                thrd = abs(St[IndIX])








                if optPos in [1,2]:
                    St[St < thrd] = 0
                    indNZ = np.where(St > thrd)[0]

                    if L0 == 0:
                        St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ])


                else:
                    St[(abs(St) < thrd)] = 0
                    indNZ = np.where(abs(St) > thrd)[0]

                    if L0 == 0:
                        St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ])



            S[indS,:] = Stemp

            # --- Updating the mixing matrix

            if len(indS) > 1:

                Atemp = unmix_cgsolve(S[indS,:].T,Resi.T,maxiter=n)    #---- Using CG

                A[:,indS] = Atemp.T

                A[:,indS] = np.dot(A[:,indS],np.diag(1./(1e-24 + np.sqrt(np.sum(A[:,indS]**2,axis=0)))))  #--- Renormalization

            else:

                Atemp = np.dot(Resi,S[indS,:].T)
                A[:,indS] = Atemp/(1e-24 + np.linalg.norm(Atemp))

            if optPos in [1,3]:
                for ii in range(len(indS)):
                    Al = A[:,indS[ii]]
                    Al[Al < 0] = 0
                    A[:,indS[ii]] = Al



            Resi = Resi - np.dot(A[:,indS],S[indS,:]) #--- Removing the sources

            Delta = np.linalg.norm(A - Aold)/(1e-24 + np.linalg.norm(Aold))






            if (np.mod(it,500) == 1) & (verb == 1):
                print("It #",it," - Delta = ",Delta)
                if len(indS) == 1:
                    print("Deflation")
                elif len(indS) == n:
                    print("GMCA")
                else:
                    print("minibatch - ratio : ",len(indS)/n)

            if verb:
                DeltaAngle = np.sum(A*Aold,axis=0)
                deltaTabAngle[it] = np.min(DeltaAngle) # min car cos decroissant: on veut que le plus grand angle soit faible, donc le plus petit cos grand
                if np.mod(it,500) == 1:
                    print('Delta angle: %s' %(np.arccos(deltaTabAngle[it])))


            Aold = deepcopy(A)

    if verb:
        elapsed_time = time.time() - start_time
        print("Stopped after ",it," iterations, in ",elapsed_time," seconds")





    if J > 0:
        Sw = cp.deepcopy(S)
        SFinW = np.zeros((n,t,J+1))
        SFinW[:,:,0:J] = S.reshape(n,t,J)
        SFinW[:,:,J] = np.dot(np.linalg.pinv(A),Xw[:,:,J])
        S = pys.backward1d(SFinW)


    if J > 0:
        return S,A,exception,Sw
    else:
        return S,A,exception

################# END OF cGMCA

def J_alpha(Sin=0,alpha=0.5):


    S = deepcopy(Sin)
    nS = np.shape(S)
    S = np.dot(np.diag(1./(np.max(abs(S),axis=1)+1e-12)),S)

    Sa = np.power(abs(S),alpha)
    wa = np.sum(Sa,axis=0)+1e-24

    S1 = abs(S)

    Wa = np.tile(wa,(nS[0],1))*Sa

    for r in range(0,nS[0]):

        ind = np.where(Wa[r,:] > 1e-24)[0]
        jind = np.where(Wa[r,:] < 1e-24)[0]

        if len(ind) > 0:
            Wa[r,ind] = S1[r,ind]/(Wa[r,ind] + np.median(Wa[r,ind]));
            Wa[r,ind] /= np.max(Wa[r,ind])

        if len(jind) > 0:
            Wa[r,jind] = 1.

    return Wa,wa

################# Conjugate gradient for estimating A

def unmix_cgsolve(MixMat,X,tol=1e-6,maxiter=100,verbose=0):

    import numpy as np
    from copy import deepcopy as dp

    b = np.dot(MixMat.T,X)
    A = np.dot(MixMat.T,MixMat)
    S = 0.*dp(b)
    r = dp(b)
    d = dp(r)
    delta = np.sum(r*r)
    delta0 = np.sum(b*b)
    numiter = 0
    bestS = dp(S)
    bestres = np.sqrt(delta/delta0)

    while ((numiter < maxiter) & (delta > tol**2*delta0)):

        # Apply At A
        q = np.dot(A,d)
        alpha = delta/np.sum(d*q)

        S = S + alpha*d

        if np.mod(numiter+1,50) == 0:
            # r = b - Aux*x
            r = b - np.dot(A,S)
        else:
            r = r - alpha*q

        deltaold = dp(delta)
        delta = np.sum(r*r)
        beta = delta/deltaold
        d = r + beta*d
        numiter = numiter + 1

        if np.sqrt(delta/delta0) < bestres:
            bestS = S
            bestres = np.sqrt(delta/delta0)

    return bestS

def diffVect(vectRef,vectSec):
    Diff = set(vectRef) - set(vectSec)
    return list(Diff)



################ PALM internal code

def PALM_NMF_MainBlock(X=0,n=0,A=0,S=0,kend=3,nmax=100,L0=0,blocksize=2,tol=1e-12,optPos = 0,J=0):
    '''Usage:
        S,A = PALM_NMF_MainBlock(X=0,n=0,A=0,S=0,kend=3,nmax=100,L0=0,blocksize=2,tol=1e-12,optPos = 0,J=0)


      Inputs:
       X    : m x t array (input data, each row corresponds to a single observation)
       n     : scalar (number of sources to be estimated)
       A    : m x n array (initialization of the mixing matrix - can be performed by the bGMCA function)
       S    : n x t array (initialization of the sources - can be performed by the bGMCA function)
       kend : scalar (final value of the k-mad thresholding)
       nmax  : scalar (number of iterations)
       L0   : boolean (0: use of the L1 norm, 1: use of the L0 norm)
       blocksize : int (size of the blocks)
       tol : float (stopping criterion)
       optPos : int in {0,1,2,3} option for the non-negativity: 0 no constraint, 1 : non-negativity on A and S, 2: non-negativity on S, 3: non-negativity on A
       J : int (number of wavelet scales. If J = 0, no wavelet transform)

      Outputs:
       S    : n x t array (estimated sources)
       A    : m x n array (estimated mixing matrix)

      Description:
        Computes the sources and the mixing matrix with a PALM algorithm using blocks. Allows to enforce the non-negativity in the direct domain but not to use sparsity in the wavelet domain at the same time. Does not perform the initialization of A and S

      Example:
        S,A = PALM_NMF_MainBlock(X,n=50,A,S,kend=3,nmax=100000,L0=0,blocksize=10,tol=1e-12,optPos = 0,J=0)'''

    # Stopping/iteration variables
    #optPos : 1 : tout, 2 : S, 3 : A

    if J > 0: # Wavelet transform
        t = np.shape(X)[1]
        Xw = pys.forward1d(X,J=J)
        n_Xw = np.shape(Xw)
        X = Xw[:,:,0:J].reshape(n_Xw[0],n_Xw[1]*J)



    # Initializations
    Go_On = 1
    it = 1
    Aold = deepcopy(A)
    deltaTabAngle = np.zeros(nmax+1)
    n_X = np.shape(X)
    thrdTab = -np.ones((n,n_X[1]))






    # Initialization to ensure that the given matrices A and S are non negative
    n_S = np.shape(S)
    for ii in range(0,n_S[0]):
        Sl = S[ii,:]
        Al = A[:,ii]

        if optPos in [1,2]:
            indMax = np.argmax(abs(Sl))
            if Sl[indMax] < 0:
                Sl = -Sl
                Al = -Al

            Sl[Sl < 0] = 0
            if optPos == 1:
                Al[Al < 0] = 0

                if np.linalg.norm(Al) > 1:
                    Al = Al/np.linalg.norm(Al)



        elif optPos == 3:
            indMax = np.argmax(abs(Al))
            if Al[indMax] < 0:
                Sl = -Sl
                Al = -Al

            Al[Al < 0] = 0
            if np.linalg.norm(Al) > 1:
                Al = Al/np.linalg.norm(Al)

        S[ii,:] = Sl
        A[:,ii] = Al

    #---
    #--- Main loop
    #---

    while Go_On:

        it += 1

        if it == nmax: # If stopping criterion met
            Go_On = 0


         # We create the blocks deterministically
        indStt = blocksize*(it-1) % np.shape(S)[0]
        indEnd = blocksize*it % np.shape(S)[0]

        if indStt < indEnd:
            indS = range(int(indStt),int(indEnd))
        else:
            ind1 = range(indStt,np.shape(S)[0])
            if(indEnd > 0):
                ind2 = range(0,indEnd)
                indS = np.concatenate((ind1,ind2))
            else:
                indS = ind1






        if np.size(indS) > 0:

            # Computation of a gradient step for S
            Stemp = S[indS,:].copy()
            Atemp = A[:,indS].copy()

            specNormAtemp = np.linalg.norm(np.dot(Atemp.T,Atemp),ord=2)



            deltaF = np.dot(Atemp.T,np.dot(A,S)-X)
            Stemp = Stemp - 1/specNormAtemp*deltaF


                # Thresholding
            for r in range(0,np.shape(Stemp)[0]):

                gradSt = Stemp[r,:]


                if thrdTab[indS[r],0] < 0:# We fix the threshold at the first iteration and keep the same for the following ones.
                    thrd = mad(gradSt) # We have a single value for the threshold per line
                    thrd = kend*thrd

                    thrd = thrd*np.ones((1,len(gradSt)))[0,:]
                    thrdTab[indS[r],:] = thrd

                thrd = thrdTab[indS[r],:]/specNormAtemp



                if optPos in [1,2]: # if the non-negativity is enforced on the sources
                    gradSt[gradSt < thrd] = 0
                    indNZ = np.where(gradSt > thrd)[0]


                    if L0 == 0:
                        gradSt[indNZ] = gradSt[indNZ] - thrd[indNZ]*np.sign(gradSt[indNZ]) # We only have the positive part => we don't need to use the sign


                else:
                    gradSt[(abs(gradSt) < thrd)] = 0
                    indNZ = np.where(abs(gradSt) > thrd)[0]


                    if L0 == 0:
                        gradSt[indNZ] = gradSt[indNZ] - thrd[indNZ]*np.sign(gradSt[indNZ])



                Stemp[r,:] = gradSt.copy()

            A[:,indS] = Atemp.copy()
            S[indS,:] = Stemp.copy()




            # Computation of the gradient step for A

            Stemp = S[indS,:].copy()
            Atemp = A[:,indS].copy()

            specNormStemp = np.linalg.norm(np.dot(Stemp,Stemp.T), ord=2)
            Atemp = Atemp - 1/specNormStemp*np.dot((np.dot(A,S)-X),Stemp.T)


            if optPos in [1,3]: # If non-negativity is enforced on the mixing matrix
                for r in range(0,Atemp.shape[1]): # Le fait de multiplier par D ne change rien, vu que ses coeffs sont positifs.
                    Ac = Atemp[:,r]
                    indMax = np.argmax(abs(Ac))
                    if Ac[indMax] < 0:
                        print('Warning, max < 0')
                    Ac[Ac < 0] = 0
                    Atemp[:,r] = Ac

            # Projection of the column of gradA on the L1 ball

            for r in range(0,Atemp.shape[1]):
                if(np.linalg.norm(Atemp[:,r]) > 1):
                    Atemp[:,r] = Atemp[:,r]/np.linalg.norm(Atemp[:,r])# Normalisation validee


            A[:,indS] = Atemp.copy()
            S[indS,:] = Stemp.copy()




            # Computation of the stopping criterion
            DeltaAngle = np.sum(A*Aold,axis=0)
            deltaTabAngle[it] = np.min(DeltaAngle) # min car cos decroissant: on veut que le plus grand angle soit faible, donc le plus petit cos grand
            if np.mod(it,500) == 0:
                print(it)
                Delta = np.linalg.norm(A - Aold)/(1e-24 + np.linalg.norm(Aold))
                print('Delta angle: %s' %(np.arccos(deltaTabAngle[it])))
                print('Delta: %s' %(Delta))


            if it > 10000 and deltaTabAngle[it] > np.cos(9*1e-8):
                Go_On = 0

            Aold = deepcopy(A)

    if J > 0: # Giong back into the direct domain
        SFinW = np.zeros((n,t,J+1))
        SFinW[:,:,0:J] = S.reshape(n,t,J)
        SFinW[:,:,J] = np.dot(np.linalg.pinv(A),Xw[:,:,J])
        S = pys.backward1d(SFinW)

    return S,A

################# AMCA Main function

def bGMCA_NMF_ondt_naif(X,n,mints=3,nmax=100,L0=0,verb=0,Init=0,blocksize=2,optBlock=0,J=2):
    '''Usage:
        S,A,exception = bGMCA_NMF_ondt_naif(X,n,mints=3,nmax=100,L0=0,verb=0,Init=0,blocksize=2,optBlock=0,J=0)


      Inputs:
       X    : m x t array (input data, each row corresponds to a single observation)
       n     : scalar (number of sources to be estimated)
       mints : scalar (final value of the k-mad thresholding)
       nmax  : scalar (number of iterations)
       L0   : boolean (0: use of the L1 norm, 1: use of the L0 norm)
       verb   : boolean (if set to 1, in verbose mode)
       Init  : scalar (if set to 0: PCA-based initialization, if set to 1: random initialization)
       blocksize : int (size of the blocks)
       optBlock: int in {0,1,2,3} (way the block is constructed: 0 : Random creation of the block, 1 : Correlation, 2 : Loop over all the sources + take the sources that are the most correlated with the current one, 3 : We create the batchs sequentially)
       J : int (number of wavelet scales. If J = 0, no wavelet transform)

      Outputs:
       S    : n x t array (estimated sources)
       A    : m x n array (estimated mixing matrix)
       exception : boolean (equals to 1 is a SVD did not converge during the iterations)

      Description:
        Computes the sources and the mixing matrix with bGMCA using blocks. Enforces non-negativity in the direct domain and sparsity in the wavelet domain. Performs the initialization of A and S

      Example:
         S,A,exception = bGMCA_NMF_ondt_naif(X,n=50,mints=3,nmax=10000,L0=0,verb=1,Init=1,blocksize=5,optBlock=0,J=2)'''





    # X is the direct domain
    nX = np.shape(X);
    m = nX[0];t = nX[1]
    Xw = cp.copy(X)

    #Initialization could largely be improved -> it should be data-dependent

    if verb:
        print("Initializing ...")

    if Init != 3:
        if Init == 0:
            R = np.dot(Xw,Xw.T)
            D,V = lng.eig(R)
            A = V[:,0:n]

        if Init == 1:
            A = np.random.randn(m,n)

        A = np.dot(A,np.diag(1./np.sqrt(np.sum(A**2,axis=0))))

        S = np.zeros((n,t*J));
    else:
        print('Attention, pas d initialisation')

    #coarseScale = np.dot(np.linalg.pinv(A),coarseScale)
    S,A,exception = bGMCA_MainBlock_NMF_ondt_naif(Xw,n,A,S,kend = mints,nmax=nmax,L0=L0,verb=verb,blocksize=blocksize,optBlock=optBlock,J=J);

    return S,A,exception

################# AMCA internal code (Weighting the sources only)

def bGMCA_MainBlock_NMF_ondt_naif(X=0,n=0,A=0,S=0,kend=3,nmax=100,L0=0,verb=0,blocksize=2,tol=1e-12,optBlock=0,J=2):
    '''Usage:
        S,A,exception = bGMCA_MainBlock_NMF_ondt_naif(X=0,n=0,A=0,S=0,kend=3,nmax=100,L0=0,verb=0,blocksize=2,tol=1e-12,optBlock=0,J=0)


      Inputs:
       X    : m x t array (input data, each row corresponds to a single observation)
       n     : scalar (number of sources to be estimated)
       A    : m x n array (initialization of the mixing matrix - can be performed by the bGMCA function)
       S    : n x t array (initialization of the sources - can be performed by the bGMCA function)
       kend : scalar (final value of the k-mad thresholding)
       nmax  : scalar (number of iterations)
       L0   : boolean (0: use of the L1 norm, 1: use of the L0 norm)
       verb   : boolean (if set to 1, in verbose mode)
       blocksize : int (size of the blocks)
       tol : float (stopping criterion)
       optBlock: int in {0,1,2,3} (way the block is constructed: 0 : Random creation of the block, 1 : Correlation, 2 : Loop over all the sources + take the sources that are the most correlated with the current one, 3 : We create the batchs sequentially)
       J : int (number of wavelet scales. If J = 0, no wavelet transform)

      Outputs:
       S    : n x t array (estimated sources)
       A    : m x n array (estimated mixing matrix)
       exception : boolean (equals to 1 is a SVD did not converge during the iterations)

      Description:
        Computes the sources and the mixing matrix with bGMCA using blocks. Enforces non-negativity in the direct domain and sparsity in the wavelet domain. Does not perform the initialization of A and S

      Example:
         S,A,exception = bGMCA_MainBlock_NMF_ondt_naif(X,n=50,A,S,kend=3,nmax=10000,L0=0,verb=1,blocksize=5,tol=1e-12,optBlock=0,J=2)'''

    Xdir = cp.deepcopy(X)
    t = np.shape(Xdir)[1]
    Xw = pys.forward1d(X,J=J)
    n_Xw = np.shape(Xw)
    X = Xw[:,:,0:J].reshape(n_Xw[0],n_Xw[1]*J)

    #--- Initialization
    optPos = 1
    powCor = 0.5
    n_S = np.shape(S)
    exception = 0

    perc = 1./nmax

    thrdTab = np.zeros((nmax + 1,n_S[0]))

    tabCard = -np.ones((n,1))


    if optPos in [1,2,3]:
        for ii in range(0,n_S[0]): # Initialization to ensure that the given matrices A and S are non negative
            Sl = S[ii,:]
            Al = A[:,ii]

            if optPos in [1,2]:
                indMax = np.argmax(abs(Sl))
                if Sl[indMax] < 0:
                    Sl = -Sl
                    Al = -Al

                Sl[Sl < 0] = 0
                if optPos == 1:
                    Al[Al < 0] = 0

                    Al = Al/np.linalg.norm(Al)



            elif optPos == 3:
                indMax = np.argmax(abs(Al))
                if Al[indMax] < 0:
                    Sl = -Sl
                    Al = -Al

                Al[Al < 0] = 0

                Al = Al/(1e-24 + np.linalg.norm(Al))

            S[ii,:] = Sl
            A[:,ii] = Al

    # Stopping/iteration variables

    Go_On = 1
    it = 1



    if verb:
        print("Starting main loop ...")
        print(" ")
        print("  - Final k: ",kend)
        print("  - Maximum number of iterations: ",nmax)
        print("  - Batch size: ",blocksize)
        print("  - Using support-based threshold estimation")
        if L0:
            print("  - Using L0 norm rather than L1")
        print(" ")
        print(" ... processing ...")
        start_time = time.time()

    # Defines the residual

    Resi = cp.deepcopy(X)

    #---
    #--- Main loop
    #---






    while Go_On:
        it += 1


        if it == nmax:
            Go_On = 0

        #--- Estimate the sources (only when the corresponding column is non-zeros)

        sigA = np.sqrt(np.sum(A*A,axis=0))
        indS = np.where(sigA > 1e-24)[0]

        if optBlock == 0: # Random
            if blocksize < n_S[0]:
                IndBatch = randperm(len(indS))  #---- mini-batch amongst available sources
            else:
                IndBatch = range(len(indS))

            if blocksize < len(indS):
                indS = indS[IndBatch[0:blocksize]]

        elif optBlock == 1: # Correlation
            if blocksize < n_S[0]:
                currSrcInd = it % len(indS)
                currSrcInd = indS[currSrcInd]
                currSrc = np.power(S[currSrcInd,:].T,powCor)
                matCor = np.dot(np.power(S,powCor),currSrc)
                matCor[currSrcInd] = -1
                IndBatch = np.argsort(matCor)
                IndBatch = np.append(IndBatch,currSrcInd)
                IndBatch = IndBatch[::-1]

            else:
                IndBatch = range(len(indS))

            if blocksize < len(indS):
                if len(indS) == n_S[0]:
                    indS = indS[IndBatch[0:blocksize]]


                else:
                    print('Attention, cas special')

                    indSTemp = [currSrcInd]
                    ii = 0
                    while len(indSTemp) < blocksize:
                        ii = ii + 1
                        if IndBatch[ii] in indS:
                            indSTemp = indSTemp + [IndBatch[ii]]




        elif optBlock == 2: # Loop over all the sources + take the sources that are the most correlated with the current one
            if blocksize < n_S[0]:
                currSrcInd = it % len(indS)
                currSrcInd = indS[currSrcInd]
                angTab = np.dot(A[:,currSrcInd].T,A)
                angTab[currSrcInd] = -2 #0 : correspond a l'angle le plus grand, 90 degres
                IndBatch = np.argsort(angTab)
                IndBatch = np.append(IndBatch,currSrcInd)
                IndBatch = IndBatch[::-1]

            else:
                IndBatch = range(len(indS))

            if blocksize < len(indS):
                if len(indS) == n_S[0]:
                    indS = indS[IndBatch[0:blocksize]]
                else:
                    print('Attention, cas special')
                    indSTemp = [currSrcInd]
                    ii = 0
                    while len(indSTemp) < blocksize:
                        ii = ii + 1
                        if IndBatch[ii] in indS:
                            indSTemp = indSTemp + [IndBatch[ii]]


        elif optBlock == 4: # We create the batchs sequentially
            indStt = blocksize*(it-1) % np.shape(S)[0]
            indEnd = blocksize*it % np.shape(S)[0]

            if indStt < indEnd:
                IndBatch = range(int(indStt),int(indEnd))
            else:
                ind1 = range(indStt,np.shape(S)[0])
                if(indEnd > 0):
                    ind2 = range(0,indEnd)
                    IndBatch = np.concatenate((ind1,ind2))
                else:
                    IndBatch = ind1

            if blocksize < len(indS):
                if len(indS) == n_S[0]:
                    indS = indS[IndBatch[0:blocksize]]


                else:
                    print('Attention, cas special')
                    indSTemp = [indStt]
                    ii = 0
                    while len(indSTemp) < blocksize:
                        ii = ii + 1
                        if IndBatch[ii] in indS:
                            indSTemp = indSTemp + [IndBatch[ii]]





        Resi = Resi + np.dot(A[:,indS],S[indS,:])   # Putting back the sources


        if np.size(indS) > 0:

            if len(indS) > 1:

                # Least_square estimate

                Ra = np.dot(A[:,indS].T,A[:,indS])
                excThisIt = 0
                try:
                    Ua,Sa,Va = np.linalg.svd(Ra)
                except np.linalg.linalg.LinAlgError:
                    print(indS)
                    print(np.sum(np.isnan(Ra)))
                    print(np.sum(np.isinf(Ra)))
                    print('SVD DID NOT CONVERGE')
                    exception += 1
                    excThisIt = 1

                if excThisIt == 0:
                    cd_Ra = np.min(Sa)/np.max(Sa)
                else:
                    cd_Ra = np.linalg.norm(Ra,ord=-2)/np.linalg.norm(Ra,ord=2)

                if (cd_Ra > 1e-12) and not(excThisIt): # if the condition number is large enough
                    iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
                    piA = np.dot(iRa,A[:,indS].T)
                    piA = np.real(piA)
                    S[indS,:] = np.dot(piA,Resi)

                if (cd_Ra < 1e-12 or excThisIt == 1): # otherwise a few gradient descent steps
                    print('GRADIENT STEP')
                    if excThisIt == 0:
                        La = np.max(Sa)
                    else:
                        La = np.linalg.norm(Ra,ord=2)
                        print('SVD DID NOT CONVERGE')

                    for it_A in range(0,10):
                        S[indS,:] = S[indS,:] + 1/La*np.dot(A[:,indS].T,X - np.dot(A[:,indS],S[indS,:]))

            else:  # only a single element in the selection / the update is straightforward

                S[indS[0],:] = np.dot(1./np.linalg.norm(A[:,indS[0]])*A[:,indS[0]],Resi)

            # Non-negativity
            if optPos in [1,2]: # If the non-negativity is enforced on the sources
                StempMat = np.zeros((len(indS),t,J+1))
                StempMat[:,:,0:J] = S[indS,:].reshape(len(indS),t,J)
                StempMat[:,:,J] = np.dot(np.linalg.pinv(A),Xw[:,:,J])[indS,:] # coarse scale
                StempDir = pys.backward1d(StempMat) # Go back to the direct domain
                StempDir[StempDir < 0] = 0 # Projection on the positive orthant
                StempMat = pys.forward1d(StempDir,J=J) # Go again the the wavelet domain
                S[indS,:] = StempMat[:,:,0:J].reshape(len(indS),J*t)

            Stemp = S[indS,:]

            # Thresholding
            for r in range(len(indS)):
                St = Stemp[r,:]

                #Computation of the effective support
                indNZ = np.where(abs(St) > kend*mad(St))[0]
                thrd = mad(St[indNZ])

                # Computation of the threshold
                Kval = np.min([np.floor(np.max([0.01,perc*it])*len(indNZ)),n_S[1]-1.]) # Includes an increasing percentage of available entries
                I = abs(St[indNZ]).argsort()[::-1]
                Kval = np.int(np.min([np.max([Kval,n]),len(I)-1.]))
                tabCard[indS[r]] = Kval
                IndIX = np.int(indNZ[I[Kval]])
                thrd = abs(St[IndIX])

                thrdTab[it,indS[r]] = thrd


                St[(abs(St) < thrd)] = 0
                indNZ = np.where(abs(St) > thrd)[0]

                if L0 == 0:
                    St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ])




            S[indS,:] = Stemp


            # --- Updating the mixing matrix
            if len(indS) > 1:

                Atemp = unmix_cgsolve(S[indS,:].T,Resi.T,maxiter=n)    #---- Using CG
                A[:,indS] = Atemp.T

                if optPos in [1,3]: # Enforcing non-negativity on A
                    for ii in range(len(indS)):
                        Al = A[:,indS[ii]]
                        Al[Al < 0] = 0
                        A[:,indS[ii]] = Al

                A[:,indS] = np.dot(A[:,indS],np.diag(1./(1e-24 + np.sqrt(np.sum(A[:,indS]**2,axis=0)))))  #--- Renormalization

            else: # Case where the update is explicit

                Atemp = np.dot(Resi,S[indS,:].T)
                A[:,indS] = Atemp/(1e-24 + np.linalg.norm(Atemp))


            Resi = Resi - np.dot(A[:,indS],S[indS,:]) #--- Removing the sources




            if (np.mod(it,500) == 1) & (verb == 1):
                print("It #",it)
                if len(indS) == 1:
                    print("Deflation")
                elif len(indS) == n:
                    print("GMCA")
                else:
                    print("minibatch - ratio : ",len(indS)/n)


    if verb:
        elapsed_time = time.time() - start_time
        print("Stopped after ",it," iterations, in ",elapsed_time," seconds")




    SFinW = np.zeros((n,t,J+1))
    SFinW[:,:,0:J] = S.reshape(n,t,J)
    SFinW[:,:,J] = np.dot(np.linalg.pinv(A),Xw[:,:,J])
    S = pys.backward1d(SFinW)

    return S,A,exception



################ PALM internal code

def PALM_NMF_MainBlock_prox(X=0,n=0,A=0,S=0,kend=3,nmax=100,L0=0,verb=0,blocksize=2,tol=1e-12,J=2):
    '''Usage:
        S,A = PALM_NMF_MainBlock_prox(X=0,n=0,A=0,S=0,kend=3,nmax=100,L0=0,verb=0,blocksize=2,tol=1e-12,J=2)


      Inputs:
       X    : m x t array (input data, each row corresponds to a single observation)
       n     : scalar (number of sources to be estimated)
       A    : m x n array (initialization of the mixing matrix - can be performed by the bGMCA function)
       S    : n x t array (initialization of the sources - can be performed by the bGMCA function)
       kend : scalar (final value of the k-mad thresholding)
       nmax  : scalar (number of iterations)
       L0   : boolean (0: use of the L1 norm, 1: use of the L0 norm)
       verb   : boolean (if set to 1, in verbose mode)
       blocksize : int (size of the blocks)
       tol : float (stopping criterion)
       J : int (number of wavelet scales. If J = 0, no wavelet transform)

      Outputs:
       S    : n x t array (estimated sources)
       A    : m x n array (estimated mixing matrix)

      Description:
        Computes the sources and the mixing matrix with a PALM algorithm using blocks. Enforces the non-negativity in the direct domain while using sparsity in a transformed domain (using a non-explicit proximal operator). Does not perform the initialization of A and S

      Example:
        S,A = PALM_NMF_MainBlock_prox(X,n=50,A,S,kend=3,nmax=100000,L0=0,verb=0,blocksize=5,tol=1e-12,J=2)'''

    # Transform into wavelet domain to compute the thresholds
    Xw = pys.forward1d(X,J=J)
    n_Xw = np.shape(Xw)
    Xw = Xw[:,:,0:J].reshape(n_Xw[0],n_Xw[1]*J)
    Sw = pys.forward1d(S,J=J)
    n_Sw = np.shape(Sw)
    Sw = Sw[:,:,0:J].reshape(n_Sw[0],n_Sw[1]*J)




    # Computing the thresholds, which will stay fixed for the whole algorithm
    thrdTab = -np.ones((n,n_Xw[1]*J))

    dG = 1/np.linalg.norm(A,ord=2)**2*np.dot(A.T,np.dot(A,Sw)-Xw)

    for r in range(0,np.shape(S)[0]):
        dGTemp = dG[r,:]
        thrd = mad(dGTemp) # We have a single value for the threshold per line
        thrd = kend*thrd

        thrd = thrd*np.ones((1,n_Xw[1]*J))[0,:]

        thrdTab[r,:] = thrd





    # Stopping/iteration variables
    Go_On = 1
    it = 1
    Aold = deepcopy(A)
    deltaTabAngle = np.zeros(nmax+1)


    #---
    #--- Main loop
    #---


    while Go_On:

        it += 1
        if it == nmax:
            Go_On = 0




         # We create the batchs deterministically
        indStt = blocksize*(it-1) % np.shape(S)[0]
        indEnd = blocksize*it % np.shape(S)[0]

        if indStt < indEnd:
            indS = range(int(indStt),int(indEnd))
        else:
            ind1 = range(indStt,np.shape(S)[0])
            if(indEnd > 0):
                ind2 = range(0,indEnd)
                indS = np.concatenate((ind1,ind2))
            else:
                indS = ind1






        if np.size(indS) > 0:
            Stemp = S[indS,:].copy()
            Atemp = A[:,indS].copy()

            # Computation of the gradient step for A
            specNormStemp = np.linalg.norm(np.dot(Stemp,Stemp.T), ord=2)

            Atemp = Atemp - 1/specNormStemp*np.dot((np.dot(A,S)-X),Stemp.T)

            # We enforce non-negativity on A
            for r in range(0,Atemp.shape[1]):
                Ac = Atemp[:,r]
                indMax = np.argmax(abs(Ac))
                if Ac[indMax] < 0:
                    print('Attention, max < 0')
                Ac[Ac < 0] = 0
                Atemp[:,r] = Ac

            # Projection of the columns of gradA on the L1 ball

            for r in range(0,Atemp.shape[1]):
                if(np.linalg.norm(Atemp[:,r]) > 1):
                    Atemp[:,r] = Atemp[:,r]/np.linalg.norm(Atemp[:,r])# Normalisation validee


            A[:,indS] = Atemp.copy()
            S[indS,:] = Stemp.copy()



            # Computation of a gradient step for S

            Stemp = S[indS,:].copy()
            Atemp = A[:,indS].copy()

            specNormAtemp = np.linalg.norm(np.dot(Atemp.T,Atemp),ord=2)

            deltaF = np.dot(Atemp.T,np.dot(A,S)-X)
            Stemp = Stemp - 1/specNormAtemp*deltaF

            # Thresholding
            Stemp = psp.GFB_WSparse_Positivity_1D(Stemp,verb=0,tol=1e-6,niter=250,W=thrdTab[indS,:]/specNormAtemp,J=J,lam = 1,XisOndt=0,A=A[:,indS])


            A[:,indS] = Atemp.copy()
            S[indS,:] = Stemp.copy()



            # Stopping criterion
            Delta = np.linalg.norm(A - Aold)/(1e-24 + np.linalg.norm(Aold))

            DeltaAngle = np.sum(A*Aold,axis=0)
            deltaTabAngle[it] = np.min(DeltaAngle) # min car cos decroissant: on veut que le plus grand angle soit faible, donc le plus petit cos grand
            if np.mod(it,500) == 0:
                print(it)
                Delta = np.linalg.norm(A - Aold)/(1e-24 + np.linalg.norm(Aold))
                print('Delta angle: %s' %(np.arccos(deltaTabAngle[it])))
                print('Delta: %s' %(Delta))

            if it > 10000 and deltaTabAngle[it] > np.cos(9*1e-8):
                Go_On = 0

            Aold = deepcopy(A)



    return S,A
