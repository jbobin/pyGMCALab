# -*- coding: utf-8 -*-
"""
The following script performs a robustness test on different robust BSS techniques similarly
to the experiments presented in 'Unsupervised separation of sparse sources in the
presence of outliers', C.Chenot and J.Bobin.
Structure of the script:
-the parameters are chosen in the file 'parameters' (data generation, algorithms' parameters, Monte-Carlo parameters, verbose..)
-the data are generated via the file 'data', according to these parameters
- the algorithms GMCA, AMCA, rGMCA, rAMCA, PCP+GMCA, and the minimization of the beta-divergence perform the BSS problem
- the results are displayed and saved.

The following code can be used for Monte-Carlo/benchmark simulations.
"""


#Imports#

sys.path.insert(1,'/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/BSS/pyGMCALab')

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

from pyGMCA.bss.rgmca.data import sources, gaussianNoise, outliers, mixingMatrix
from pyGMCA.bss.rgmca.errors import errors
from pyGMCA.bss.rgmca.parameters import dS, pS
from pyGMCA.bss.rgmca.AGMCA import  rAMCA, AMCA, rGMCA,mad
from pyGMCA.bss.rgmca.BetaD_ICA import BetaD_ICA
from pyGMCA.bss.rgmca.rpcaAlgo import pcpGMCA



#Set the number of this run#

#import sys
#cmdargs=sys.argv
#numLancement=np.int(cmdargs[1]) #for a bash
numLancement=9 #otherwise

#Set the parameters#
numberParam=pS['nbreParam']#number of varying parameters
numberIter=pS['nbreIter']#number of runs for each parameters


#Build the matrices which will contain the different errors#

angle5A=np.zeros((numberParam,numberIter))#for AMCA#
deltaA=np.zeros((numberParam,numberIter))

angle5G=np.zeros((numberParam,numberIter))#for GMCA#
deltaG=np.zeros((numberParam,numberIter))

angle5rA=np.zeros((numberParam,numberIter))#for rAMCA
deltarA=np.zeros((numberParam,numberIter))

angle5rG=np.zeros((numberParam,numberIter))#for rGMCA
deltarG=np.zeros((numberParam,numberIter))

angle5B=np.zeros((numberParam,numberIter))#for the beta-div.
deltaB=np.zeros((numberParam,numberIter))

angle5P=np.zeros((numberParam,numberIter))# for PCP+GMCA
deltaP=np.zeros((numberParam,numberIter))



for indexParam in range(0, pS['nbreParam']):#Varying parameters

    if pS['verbose']:
        print 'Start for the ', indexParam+1, 'th parameter'


    #Set the varying parameter#
    dS['nbCol']=(pS['param'][indexParam])/100.*dS['t']



    for indexIte in range(0,pS['nbreIter']):
        if pS['verbose']:
            print 'Start for the ', indexIte+1, 'th job'

        #Set the seed#
        np.random.seed(numLancement+indexIte)

        #Data generation#
        S=sources(dS['n'],dS['t'],dS['pS'],dS['alphaS'],dS['ampliS'])#sources
        Aori=mixingMatrix(dS['m'],dS['n'])#mixing matrix
        X0=Aori.dot(S) #noiseless observations
        N=gaussianNoise(dS['m'],dS['t'],dS['ampliN'])#Gaussian noise#
        outL=outliers(dS['m'],dS['t'],dS['nbCol'],dS['alphaO'],dS['ampliO'])
        X= X0+outL+N#Noisy observations

        #Initializing mixing matrix#
        norm=np.linalg.norm(X, axis=0)
        perc=np.percentile(norm[norm>mad(norm)],50)#Do not trust the largest entries
        X3=X.copy()
        X3[:, norm>perc]=0
        R = np.dot(X3,X3.T)
        D,V = np.linalg.eig(R)
        A = V[:,0:dS['n']] #Initial mixing matrix, normalized

        #Calls of the algorithms#

        AMCA#
        Sest,Aest=AMCA(X,A, amca=1)
        angle5A[indexParam, indexIte] ,deltaA[indexParam, indexIte] =errors(Aest,Sest,Aori,S)
        if pS['verboseAMCA']:
            print 'Errors AMCA: ', angle5A[indexParam, indexIte]*dS['n'], 'recovered columns, ' ,deltaA[indexParam, indexIte] ,' for DeltaA'


        #GMCA#
        Sest,Aest=AMCA(X,A, amca=0)
        angle5G[indexParam, indexIte] ,deltaG[indexParam, indexIte] =errors(Aest,Sest,Aori,S)
        if pS['verboseGMCA']:
            print 'Errors GMCA: ', angle5G[indexParam, indexIte]*dS['n'], 'recovered columns, ', deltaG[indexParam, indexIte] ,' for DeltaA'

        #rAMCA#
        Sest,Aest,Oest=rAMCA(X,A, amca=1)
        angle5rA[indexParam, indexIte] ,deltarA[indexParam, indexIte] =errors(Aest,Sest,Aori,S)
        if pS['verboserAMCA']:
            print 'Errors rAMCA: ', angle5rA[indexParam, indexIte]*dS['n'], 'recovered columns, ', deltarA[indexParam, indexIte] ,' for DeltaA'

        #rGMCA#
        Sest,Aest,Oest=rGMCA(X,A)
        angle5rG[indexParam, indexIte] ,deltarG[indexParam, indexIte] =errors(Aest,Sest,Aori,S)
        if pS['verboserGMCA']:
            print 'Errors rGMCA: ', angle5rG[indexParam, indexIte]*dS['n'], 'recovered columns, ', deltarG[indexParam, indexIte] ,' for DeltaA'

        #PCP+GMCA#

        Sest,Aest,Ofin=pcpGMCA(X,Aori,S,A.copy())
        angle5P[indexParam, indexIte] ,deltaP[indexParam, indexIte] =errors(Aest,Sest,Aori,S)
        if pS['verbosePCP']:
            print 'Errors PCP+GMCA: ', angle5P[indexParam, indexIte]*dS['n'], 'recovered columns, ', deltaP[indexParam, indexIte] ,' for DeltaA'

        #Beta-Div#
        Sest,Aest=BetaD_ICA(X,Aori)
        angle5B[indexParam, indexIte] ,deltaB[indexParam, indexIte] =errors(Aest,Sest,Aori,S)
        if pS['verbosePCP']:
            print 'Errors beta-div.: ', angle5B[indexParam, indexIte]*dS['n'], 'recovered columns, ', deltaB[indexParam, indexIte] ,' for DeltaA'


#Save the results#
if pS['saveFile']==1:

    fileObject=open("{}_{}".format(pS['nameFile'],numLancement),'wb')

    pickle.dump(angle5A,fileObject)
    pickle.dump(deltaA,fileObject)

    pickle.dump(angle5G,fileObject)
    pickle.dump(deltaG,fileObject)

    pickle.dump(angle5rA,fileObject)
    pickle.dump(deltarA,fileObject)

    pickle.dump(angle5rG,fileObject)
    pickle.dump(deltarG,fileObject)

    pickle.dump(angle5B,fileObject)
    pickle.dump(deltaB,fileObject)

    pickle.dump(angle5P,fileObject)
    pickle.dump(deltaP,fileObject)


    fileObject.close()

#Display the results#
if pS['plot']:
    t=pS['param'][0:indexParam+1]


    plt.figure(figsize=(8,8))
    #Plot the mean#
    ax1=plt.subplot2grid((10,1), (0,0), rowspan=5)
    ax1.plot(t,np.mean(deltaA,1), label=r'AMCA', marker='x',markeredgewidth=2, markersize=8 )
    ax1.plot(t,np.mean(deltaG,1), label=r'GMCA', marker='*', markersize=9)
    ax1.plot(t,np.mean(deltarA,1),marker='o', label=r'rAMCA')
    ax1.plot(t,np.mean(deltarG,1), label=r'rGMCA', marker='+',markeredgewidth=2, markersize=8 )
    ax1.plot(t,np.mean(deltaB,1), label=r'$\beta$-div.', marker='^',color='m')
    ax1.plot(t,np.mean(deltaP,1), label=r'PCP+GMCA', marker='d', color='y')
    plt.yscale('log')
    plt.ylabel(r'Mean of $\Delta_A$',fontsize=20)
    ax1.set_xticklabels([])
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., fontsize=16)

    #Plot the probability to recover A
    ax2=plt.subplot2grid((10,1), (5,0), rowspan=5)
    ax2.plot(t,np.array(np.sum(np.array(angle5A==1,int),1),float)/numberIter, marker='x',markeredgewidth=2, markersize=8 )
    ax2.plot(t,np.array(np.sum(np.array(angle5G==1,int),1),float)/numberIter,  marker='*', markersize=9)
    ax2.plot(t,np.array(np.sum(np.array(angle5rA==1,int),1),float)/numberIter,marker='o')
    ax2.plot(t,np.array(np.sum(np.array(angle5rG==1,int),1),float)/numberIter,  marker='+',markeredgewidth=2, markersize=8 )
    ax2.plot(t,np.array(np.sum(np.array(angle5B==1,int),1),float)/numberIter, marker='^',color='m')
    ax2.plot(t,np.array(np.sum(np.array(angle5P==1,int),1),float)/numberIter, marker='d', color='y')
    plt.ylim(-0.1,1.1)
    plt.xlabel(r'Percentage of corrupted data',fontsize=20)
    plt.ylabel(r'Recovered $A$',fontsize=20)
    plt.show()


if pS['verbose']:
    print numLancement, "exit"
