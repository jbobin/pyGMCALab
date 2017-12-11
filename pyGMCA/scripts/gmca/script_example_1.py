"""
This code provides some examples of the AMCA algorithms

@author: J.Bobin
@date: April, 14 2015

"""

import sys
sys.path.insert(1,'/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/BSS/pyGMCALab')
from  pyGMCA.common import utils as bu
from  pyGMCA.bss.amca import amca
from  pyGMCA.bss.gmca import gmca
import matplotlib.pyplot as plt
import numpy as np

#
N_MC = 5 #- Number of Monte-Carlo simulations
#
print("Comparing GMCA and AMCA when sparsity is enforced in the sample domain")
#%%
print("Evolution of the mixing matrix criterion when the sparsity level varies")
#
n = 4
m = 4
t = 2048
p1 = 0.1
sigma1 = 4
pval = np.power(10,np.linspace(-2,-0.7,5))
#
C_G_sparse_level = np.zeros((5,N_MC))
C_A_sparse_level = np.zeros((5,N_MC))
#
for It_MC in range(0,N_MC):
    for R_p in range(0,5):
        p = pval[R_p]
        X,X0,A0,S0,N,sigma_noise,kern = bu.Make_Experiment_Coherent(n_s=n,n_obs=m,t_samp=t,w=0.1,noise_level=120,dynamic=0,sigma1=sigma1,p1=p1,ptot=p)
        Rg = gmca.GMCA(X0,n=n,mints=0,nmax=500,L0=0,UseP=1)
        crit = bu.EvalCriterion(A0,S0,Rg["mixmat"],Rg["sources"])
        C_G_sparse_level[R_p,It_MC] = crit["ca_med"]
        Ra = amca.AMCA(X0,n,mints=0,nmax=500,q_f = 0.1,L0=1,UseP=1)
        crit = bu.EvalCriterion(A0,S0,Ra["mixmat"],Ra["sources"])
        C_A_sparse_level[R_p,It_MC] = crit["ca_med"]
#
#%% PLOTTING THE RESULTS
plt.figure(0)
pval = np.power(10,np.linspace(-2,-0.7,5))
plt.title('Mixing matrix criterion as a function of the sparsity level')
tempG = np.median(C_G_sparse_level,1)
tempA = np.median(C_A_sparse_level,1)
Mrange = 5.**np.max([np.max(tempA),np.max(tempG)])
mrange = 0.2*np.min([np.min(tempA),np.min(tempG)])
plt.semilogy(pval,tempG,'k8',alpha=0.75)
plt.semilogy(pval,tempA,'ro',alpha=0.75)
plt.axis([0,1.1*np.max(pval), mrange,Mrange])
plt.xlabel("Sparsity level")
plt.ylabel("Median mixing matrix criterion")
#
print("Experiment complete")
#%%
print("Evolution of the mixing matrix criterion when the correlation level varies")
#
n = 4
m = 4
t =2048
p1 = 0.1
sigma1 = 4
p = 0.05
pval = np.power(10,np.linspace(-2,np.log10(0.95),10))
#
C_G_corr_level = np.zeros((10,N_MC))
C_A_corr_level = np.zeros((10,N_MC))
#
for It_MC in range(0,N_MC):
    for R_p in range(0,10):
        p1 = pval[R_p]
        X,X0,A0,S0,N,sigma_noise,kern = bu.Make_Experiment_Coherent(n_s=n,n_obs=m,t_samp=t,w=0.1,noise_level=120,dynamic=0,sigma1=sigma1,p1=p1,ptot=p)
        Rg = gmca.GMCA(X0,n=n,mints=0,nmax=500,L0=0,UseP=1)
        crit = bu.EvalCriterion(A0,S0,Rg["mixmat"],Rg["sources"])
        C_G_corr_level[R_p,It_MC] = crit["ca_med"]
        Ra = amca.AMCA(X0,n,mints=0,nmax=500,q_f = 0.1,L0=1,UseP=1)
        crit = bu.EvalCriterion(A0,S0,Ra["mixmat"],Ra["sources"])
        C_A_corr_level[R_p,It_MC] = crit["ca_med"]
#
#%% PLOTTING THE RESULTS
plt.figure(1)
pval = np.power(10,np.linspace(-2,np.log10(0.95),10))
plt.title('Mixing matrix criterion as a function of the correlation level')
tempG = np.median(C_G_corr_level,1)
tempA = np.median(C_A_corr_level,1)
Mrange = 5.**np.max([np.max(tempA),np.max(tempG)])
mrange = 0.2*np.min([np.min(tempA),np.min(tempG)])
plt.semilogy(pval,tempG,'k8',alpha=0.75)
plt.semilogy(pval,tempA,'ro',alpha=0.75)
plt.axis([0,1.1*np.max(pval), mrange,Mrange])
plt.xlabel("Correlation level")
plt.ylabel("Median mixing matrix criterion")

#
print("Experiment complete")
#%%
print("Evolution of the mixing matrix criterion when the dynamic varies")
#
print("Experiment complete")
