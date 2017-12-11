
# -*- coding: utf-8 -*-
"""
The following script performs a comparison of some robust BSS techniques similarly
to the experiments presented in 'Blind Source Separation in transformed domains', C.Chenot and J.Bobin.
The algorithms implemented are: tr-rGMCA, GMCA, the oracle, MCA+GMCA, Outliers Pursuit + GMCA.
The sources are exactly sparse in DCT and the outliers in the direct domain.
Structure of the script:
- the parameters are chosen in the file 'parameters' (data generation, algorithms parameters, Monte-Carlo parameters...)
- the data are generated via the file 'data', according to these parameters
- the algorithms perform the BSS problem
"""

#Imports#
sys.path.insert(1,'/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/BSS/pyGMCALab')

import numpy as np
from pyGMCA.bss.tr_rgmca.data import *
from pyGMCA.bss.tr_rgmca.errors import *
from pyGMCA.bss.tr_rgmca.outliersPursuit import OP_GMCA
import pickle
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from pyGMCA.bss.tr_rgmca.XMCA import *
from pyGMCA.bss.tr_rgmca.parameters import dS, pS

#Set the seed of this run#
numLancement=1#


#Set the parameters#
numberParam=pS['nbreParam']#number of varying parameters
numberIter=pS['nbreIter']#number of runs for each parameters


#Build the matrices which will contain the different metrics#
angle5MaxrGMCA=np.zeros((numberParam,numberIter))# for tr-rGMCA
angle5MedrGMCA=np.zeros((numberParam,numberIter))
deltarGMCA=np.zeros((numberParam,numberIter))
SDRMinrGMCA=np.zeros((numberParam,numberIter))
SIRMinrGMCA=np.zeros((numberParam,numberIter))
SARMinrGMCA=np.zeros((numberParam,numberIter))
SNRMinrGMCA=np.zeros((numberParam,numberIter))
SDRMedrGMCA=np.zeros((numberParam,numberIter))
SIRMedrGMCA=np.zeros((numberParam,numberIter))
SARMedrGMCA=np.zeros((numberParam,numberIter))
SNRMedrGMCA=np.zeros((numberParam,numberIter))
OmserGMCA=np.zeros((numberParam,numberIter))



angle5MaxAW=np.zeros((numberParam,numberIter))# for the Oracle
angle5MedAW=np.zeros((numberParam,numberIter))
deltaAW=np.zeros((numberParam,numberIter))
SDRMinAW=np.zeros((numberParam,numberIter))
SIRMinAW=np.zeros((numberParam,numberIter))
SARMinAW=np.zeros((numberParam,numberIter))
SNRMinAW=np.zeros((numberParam,numberIter))
SDRMedAW=np.zeros((numberParam,numberIter))
SIRMedAW=np.zeros((numberParam,numberIter))
SARMedAW=np.zeros((numberParam,numberIter))
SNRMedAW=np.zeros((numberParam,numberIter))
OmseAW=np.zeros((numberParam,numberIter))



angle5MaxOP=np.zeros((numberParam,numberIter))# for Outlier Pursuit + GMCA
angle5MedOP=np.zeros((numberParam,numberIter))
deltaOP=np.zeros((numberParam,numberIter))
SDRMinOP=np.zeros((numberParam,numberIter))
SIRMinOP=np.zeros((numberParam,numberIter))
SARMinOP=np.zeros((numberParam,numberIter))
SNRMinOP=np.zeros((numberParam,numberIter))
SDRMedOP=np.zeros((numberParam,numberIter))
SIRMedOP=np.zeros((numberParam,numberIter))
SARMedOP=np.zeros((numberParam,numberIter))
SNRMedOP=np.zeros((numberParam,numberIter))
OmseOP=np.zeros((numberParam,numberIter))


angle5MaxMCA=np.zeros((numberParam,numberIter))# for MCA+GMCA
angle5MedMCA=np.zeros((numberParam,numberIter))
deltaMCA=np.zeros((numberParam,numberIter))
SDRMinMCA=np.zeros((numberParam,numberIter))
SIRMinMCA=np.zeros((numberParam,numberIter))
SARMinMCA=np.zeros((numberParam,numberIter))
SNRMinMCA=np.zeros((numberParam,numberIter))
SDRMedMCA=np.zeros((numberParam,numberIter))
SIRMedMCA=np.zeros((numberParam,numberIter))
SARMedMCA=np.zeros((numberParam,numberIter))
SNRMedMCA=np.zeros((numberParam,numberIter))
OmseMCA=np.zeros((numberParam,numberIter))


angle5MaxGMCA=np.zeros((numberParam,numberIter))# for GMCA
angle5MedGMCA=np.zeros((numberParam,numberIter))
deltaGMCA=np.zeros((numberParam,numberIter))
SDRMinGMCA=np.zeros((numberParam,numberIter))
SIRMinGMCA=np.zeros((numberParam,numberIter))
SARMinGMCA=np.zeros((numberParam,numberIter))
SNRMinGMCA=np.zeros((numberParam,numberIter))
SDRMedGMCA=np.zeros((numberParam,numberIter))
SIRMedGMCA=np.zeros((numberParam,numberIter))
SARMedGMCA=np.zeros((numberParam,numberIter))
SNRMedGMCA=np.zeros((numberParam,numberIter))
OmseGMCA=np.zeros((numberParam,numberIter))



for indexParam in range(0, pS['nbreParam']):#Varying parameters



    for indexIte in range(0,pS['nbreIter']):


        #Set the seed#
        np.random.seed(numLancement+indexIte)
        dS['m']=pS['param'][0,indexParam]
        dS['gS']=100.*8./pS['param'][0,indexParam]
        #Data generation#
        S=sources() #Sources
        Aori=mixingMatrix()#Mixing matrix
        while np.linalg.cond(Aori)>5:
                    Aori=mixingMatrix()#Mixing matrix



        N=gaussianNoise()
        N=10**(-dS['gN']/20.)*np.linalg.norm(Aori.dot(S), 'fro')/np.linalg.norm(N, 'fro')*N #Set the energy according to the SNR


        outL=outliers()
        outL=10**(-dS['gO']/20.)*np.linalg.norm(Aori.dot(S), 'fro')/np.linalg.norm(outL, 'fro')*outL #Set the energy according to the current SOR


        X = Aori.dot(S) + N + outL #Observations



        #Initialize A with PCA. One can prefer a random initialization.
        R = np.dot(X,X.T)
        D,V = np.linalg.eig(R)
        A = V[:,0:dS['n']]
        A=np.real(A) #Initialization of the mixing matrix
        A=A/np.linalg.norm(A,axis=0)




        ###  Calls of the algorithms####

        Sest,Aest,Oest=trRGMCA(X, Aori,Afix=1)#Oracle
        if np.linalg.norm(Oest)==0:
            Oest=X-Aest.dot(Sest)
        SDRMinAW[indexParam, indexIte] ,SDRMedAW[indexParam, indexIte] ,SIRMinAW[indexParam, indexIte] ,SIRMedAW[indexParam, indexIte] ,SNRMinAW[indexParam, indexIte] ,SNRMedAW[indexParam, indexIte] ,SARMinAW[indexParam, indexIte] ,SARMedAW[indexParam, indexIte],OmseAW[indexParam, indexIte] ,deltaAW[indexParam, indexIte] ,angle5MaxAW[indexParam, indexIte],angle5MedAW[indexParam, indexIte] =errors(Oest,Sest,Aest,outL,S,Aori)
        plt.figure(figsize=(8,4));plt.plot(Stransf(S)[0], 'r', linewidth=2, label=r'Initial $S_1\Phi_S^T$');plt.plot(Stransf(Sest)[0], 'g--', linewidth=2,label='Estimated Oracle');plt.legend(loc=0);plt.show()
        plt.figure(figsize=(8,4));plt.plot(outL[0], 'r', linewidth=2, label=r'Initial $O_1\Phi_O^T$');plt.plot(Oest[0], 'g--', linewidth=2,label='Estimated Oracle');plt.legend(loc=0);plt.show()
        plt.figure(figsize=(8,4));plt.scatter(Stransf(Aori.dot(S))[0],Stransf(Aori.dot(S))[1], label=r'Initial $(AS)_1\Phi_S^T$');plt.scatter(Stransf(Aest.dot(Sest))[0],Stransf(Aest.dot(Sest))[1], color='red', alpha=0.5, marker='*', label='Estimated Oracle');plt.legend(loc=0);plt.show()

        Sest,Aest,Oest=trRGMCA(X, A.copy(),Afix=0)#trRGMCA
        if np.linalg.norm(Oest)==0:
            Oest=X-Aest.dot(Sest)
        (Sest,Aest)=reOrder(Aest,Sest,Aori,S)
        SDRMinrGMCA[indexParam, indexIte] ,SDRMedrGMCA[indexParam, indexIte] ,SIRMinrGMCA[indexParam, indexIte] ,SIRMedrGMCA[indexParam, indexIte] ,SNRMinrGMCA[indexParam, indexIte] ,SNRMedrGMCA[indexParam, indexIte] ,SARMinrGMCA[indexParam, indexIte] ,SARMedrGMCA[indexParam, indexIte],OmserGMCA[indexParam, indexIte] ,deltarGMCA[indexParam, indexIte] ,angle5MaxrGMCA[indexParam, indexIte],angle5MedrGMCA[indexParam, indexIte] =errors(Oest,Sest,Aest,outL,S,Aori)
        plt.figure(figsize=(8,4));plt.plot(Stransf(S)[0], 'r',linewidth=2, label=r'Initial $S_1\Phi_S^T$');plt.plot(Stransf(Sest)[0], 'g--', linewidth=2,label='Estimated tr-rGMCA');plt.legend(loc=0);plt.show()
        plt.figure(figsize=(8,4));plt.plot(outL[0], 'r', linewidth=2, label=r'Initial $O_1\Phi_O^T$');plt.plot(Oest[0], 'g--', linewidth=2,label='Estimated tr-rGMCA');plt.legend(loc=0);plt.show()
        plt.figure(figsize=(8,4));plt.scatter(Stransf(Aori.dot(S))[0],Stransf(Aori.dot(S))[1],  label=r'Initial $(AS)_1\Phi_S^T$');plt.scatter(Stransf(Aest.dot(Sest))[0],Stransf(Aest.dot(Sest))[1], color='red', alpha=0.5, marker='*', label='Estimated tr-rGMCA');plt.legend(loc=0);plt.show()


        Sest,Aest,Oest=OP_GMCA(X,Aori,S,A.copy())#OutliersPursuit + GMCA
        SDRMinOP[indexParam, indexIte] ,SDRMedOP[indexParam, indexIte] ,SIRMinOP[indexParam, indexIte] ,SIRMedOP[indexParam, indexIte] ,SNRMinOP[indexParam, indexIte] ,SNRMedOP[indexParam, indexIte] ,SARMinOP[indexParam, indexIte] ,SARMedOP[indexParam, indexIte],OmseOP[indexParam, indexIte] ,deltaOP[indexParam, indexIte] ,angle5MaxOP[indexParam, indexIte],angle5MedOP[indexParam, indexIte] =errors(Oest,Sest,Aest,outL,S,Aori)
        plt.figure(figsize=(8,4));plt.plot(Stransf(S)[0], 'r', linewidth=2,label=r'Initial $S_1\Phi_S^T$');plt.plot(Stransf(Sest)[0], 'g--', linewidth=2,label='Estimated OP+GMCA');plt.legend(loc=0);plt.show()
        plt.figure(figsize=(8,4));plt.plot(outL[0], 'r', linewidth=2, label=r'Initial $O_1\Phi_O^T$');plt.plot(Oest[0], 'g--', linewidth=2,label='Estimated OP+GMCA');plt.legend(loc=0);plt.show()
        plt.figure(figsize=(8,4));plt.scatter(Stransf(Aori.dot(S))[0],Stransf(Aori.dot(S))[1],  label=r'Initial $(AS)_1\Phi_S^T$');plt.scatter(Stransf(Aest.dot(Sest))[0],Stransf(Aest.dot(Sest))[1], color='red', alpha=0.5, marker='*', label='Estimated OP+GMCA');plt.legend(loc=0);plt.show()


        Sestt,Aest=AMCA(Stransf(X),A.copy(), 0)#GMCA
        Sest=Sback(Sestt)
        Oest=X-Aest.dot(Sest)
        (Sest,Aest)=reOrder(Aest,Sest,Aori,S)
        SDRMinGMCA[indexParam, indexIte] ,SDRMedGMCA[indexParam, indexIte] ,SIRMinGMCA[indexParam, indexIte] ,SIRMedGMCA[indexParam, indexIte] ,SNRMinGMCA[indexParam, indexIte] ,SNRMedGMCA[indexParam, indexIte] ,SARMinGMCA[indexParam, indexIte] ,SARMedGMCA[indexParam, indexIte],OmseGMCA[indexParam, indexIte] ,deltaGMCA[indexParam, indexIte] ,angle5MaxGMCA[indexParam, indexIte],angle5MedGMCA[indexParam, indexIte] =errors(Oest,Sest,Aest,outL,S,Aori)
        plt.figure(figsize=(8,4));plt.plot(Stransf(S)[0], 'r',linewidth=2, label=r'Initial $S_1\Phi_S^T$');plt.plot(Stransf(Sest)[0], 'g--',linewidth=2, label='Estimated GMCA');plt.legend(loc=0);plt.show()
        plt.figure(figsize=(8,4));plt.scatter(Stransf(Aori.dot(S))[0],Stransf(Aori.dot(S))[1],  label=r'Initial $(AS)_1\Phi_S^T$');plt.scatter(Stransf(Aest.dot(Sest))[0],Stransf(Aest.dot(Sest))[1], color='red', alpha=0.5, marker='*', label='Estimated GMCA');plt.legend(loc=0);plt.show()


        AS,Oest=MCA(X) #MCA+GMCA
        Sestt,Aest= AMCA(Stransf(X-Oest),A.copy(), 0)
        Sest=Sback(Sestt)
        (Sest,Aest)=reOrder(Aest,Sest,Aori,S)
        SDRMinMCA[indexParam, indexIte] ,SDRMedMCA[indexParam, indexIte] ,SIRMinMCA[indexParam, indexIte] ,SIRMedMCA[indexParam, indexIte] ,SNRMinMCA[indexParam, indexIte] ,SNRMedMCA[indexParam, indexIte] ,SARMinMCA[indexParam, indexIte] ,SARMedMCA[indexParam, indexIte],OmseMCA[indexParam, indexIte] ,deltaMCA[indexParam, indexIte] ,angle5MaxMCA[indexParam, indexIte],angle5MedMCA[indexParam, indexIte] =errors(Oest,Sest,Aest,outL,S,Aori)
        plt.figure(figsize=(8,4));plt.plot(Stransf(S)[0], 'r', linewidth=2,label=r'Initial $S_1\Phi_S^T$');plt.plot(Stransf(Sest)[0], 'g--', linewidth=2,label='Estimated MCA+GMCA');plt.legend(loc=0);plt.show()
        plt.figure(figsize=(8,4));plt.plot(outL[0], 'r', linewidth=2, label=r'Initial $O_1\Phi_O^T$');plt.plot(Oest[0], 'g--', linewidth=2,label='Estimated MCA+GMCA');plt.legend(loc=0);plt.show()
        plt.figure(figsize=(8,4));plt.scatter(Stransf(Aori.dot(S))[0],Stransf(Aori.dot(S))[1], label=r'Initial $(AS)_1\Phi_S^T$');plt.scatter(Stransf(Aest.dot(Sest))[0],Stransf(Aest.dot(Sest))[1], color='red', alpha=0.5, marker='*', label='Estimated MCA+GMCA');plt.legend(loc=0);plt.show()




#Display the results#
p=pS['param'][0,0:pS['nbreParam']]

fig=plt.figure(figsize=(24,6))
sub1 = fig.add_subplot(141)
sub1.plot(p,np.median(SDRMedAW,axis=1),color='red', label='Oracle', linewidth=2, marker='*', markersize=10)
sub1.plot(p,np.median(SDRMedrGMCA,axis=1),color='blue', label='tr-rGMCA', linewidth=2, marker='^', markersize=8)
sub1.plot(p,np.median(SDRMedMCA,axis=1),color='black', label='MCA+GMCA', linewidth=2, marker='s', markersize=8)
sub1.plot(p,np.median(SDRMedOP,axis=1),color='green', label='OP+GMCA', linewidth=2, marker='o', markersize=8)
sub1.set_xlabel('SOR', fontsize=25)
sub1.set_ylabel('SDR', fontsize=25)

h1, l1 = sub1.get_legend_handles_labels()

plt.legend(h1, l1, loc=2,bbox_to_anchor=(0.1, 1.3),  ncol=4, fontsize=25)

sub1 = fig.add_subplot(142)
sub1.plot(p,np.median(OmseAW,axis=1),color='red', label='Oracle', linewidth=2, marker='*', markersize=10)
sub1.plot(p,np.median(OmserGMCA,axis=1),color='blue', label='tr-rGMCA', linewidth=2, marker='^', markersize=8)
sub1.plot(p,np.median(OmseMCA,axis=1),color='black', label='MCA+GMCA', linewidth=2, marker='s', markersize=8)
sub1.plot(p,np.median(OmseOP,axis=1),color='green', label='OP+GMCA', linewidth=2, marker='o', markersize=8)

sub1.set_xlabel('SOR', fontsize=25)
sub1.set_ylabel('Error Outliers', fontsize=25)


sub2 = fig.add_subplot(143)
sub2.plot(p,np.median(-10*np.log10(deltarGMCA),axis=1),color='blue', label='tr-rGMCA', linewidth=2, marker='^', markersize=8)
sub2.plot(p,np.median(-10*np.log10(deltaMCA),axis=1),color='black', label='MCA+GMCA', linewidth=2, marker='s', markersize=8)
sub2.plot(p,np.median(-10*np.log10(deltaOP),axis=1),color='green', label='OP+GMCA', linewidth=2, marker='o', markersize=8)

sub2.set_xlabel('SOR', fontsize=25)
sub2.set_ylabel(r'$\Delta_A$', fontsize=25,)


plt.show()
