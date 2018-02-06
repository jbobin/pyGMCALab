"""

  COMMON TOOLS

"""

import numpy as np
from scipy import special
import numpy as np
import scipy.linalg as lng
import copy as cp
from copy import deepcopy as dp

################# RANDOM PERMUTATIONS

def randperm(n=1):

    X = np.random.randn(n)

    return X.argsort()

################# DEFINES THE MEDIAN ABSOLUTE DEVIATION

def mad(xin = 0):

    z = np.median(abs(xin - np.median(xin)))/0.6735

    return z

################# DEFINES THE MEDIAN ABSOLUTE DEVIATION (for the rows of a matrix)

def m_mad(xin = 0):

    nX = np.shape(xin)

    z = np.median(abs(xin - np.tile(np.median(xin,axis=1).reshape(nX[0],1),(1,nX[1]))),axis=1)/0.6735

    return z

##### Generating functions

def generate_2D_generalized_gaussian(rows, columns, alpha=2):

    m = rows
    n = columns
    r = 0.5 * np.random.random(m * n) + 0.5  # distribution is symmetric
    beta = np.sqrt(special.gamma(3.0 / alpha) /
                   special.gamma(1.0 / alpha))  # enough to consider r > 0.5
    y = r / beta
    ymin = 1e-20 * np.ones(m * n)
    ymax = 1000 * np.ones(m * n)
    # for simplicity, generated r.v. are bounded by 1000.
    for iter in range(0, 33):
        cdf = 0.5 + 0.5 * special.gammainc(1.0 / alpha, (beta * y) ** alpha)
        indplus = np.nonzero(cdf > r)
        if len(indplus) > 0:
            ymax[indplus] = y[indplus]
        indminus = np.nonzero(cdf < r)
        if len(indminus) > 0:
            ymin[indminus] = y[indminus]
        y = 0.5 * (ymax + ymin)
    ind = np.nonzero(np.random.random(m * n) > 0.5)
    if len(ind) > 0:
        y[ind] = -y[ind]
    x = y.reshape([n, m]).T.copy()
    return x

##### Generating functions

def MixtMod(n=2,t=1024,sigma1=1,p1=0.2,ptot=0.1):

    S = np.zeros((n,t))
    Sc_part = np.zeros((n,t))

    num_cor = np.int(np.floor(p1*ptot*t))
    ind_cor = np.random.permutation(t)
    ind_cor = ind_cor[0:num_cor]
    ind = np.ones((1,t))
    ind[0,ind_cor] = 0
    ind_nocor = np.where(ind[0,:] == 1)[0]

    rest = t - num_cor

    for r in range(0,n):
        p_active = np.int(np.floor(t*ptot*(1.-p1)))
        temp = np.random.randn(1,rest)
        ind = np.random.permutation(rest)
        temp[0,ind[p_active+1:rest]] = 0
        cor_val = sigma1*np.random.randn(1,num_cor)
        S[r,ind_cor] = cor_val
        S[r,ind_nocor] = temp
        Sc_part[r,ind_cor] = cor_val

    return S,Sc_part,ind_cor,ind_nocor

# Get a mixing matrix with a given condition number

def Get_MixmatCond(n,m,cd):

    nmax = 25
    A0 = np.random.randn(m,n)

    if (cd > 1):
        v = np.linspace(0,n,n)
        v = (1 - 1/cd)*v/n + 1/cd

    if (cd == 1):
        v = np.ones((n,1))

    Sq = np.zeros((m,n))
    for r in range(0,n):
        Sq[r,r] = v[r]

    for r in range(0,nmax):

        ue,se,ve = np.linalg.svd(A0)
        A0 = np.dot(np.dot(ue,Sq),ve.T)

        for t in range(0,n):
            A0[:,t] = A0[:,t]/np.linalg.norm(A0[:,t])

    return A0



#
def Make_Experiment_Coherent(n_s=2,n_obs=2,t_samp=1024,w=15,noise_level=40,dynamic=10,sigma1=1,p1=0.1,ptot=0.1):

    S,Sc_part,ind_cor,ind_nocor = MixtMod(n_s,t_samp,sigma1,p1,ptot)

    x = np.linspace(1,t_samp,t_samp)-t_samp/2

    kern = np.exp(-abs(x)/(w/np.log(2)))
    kern = kern/np.max(kern)

    for r in range(0,n_s):
        S[r,:] = np.convolve(S[r,:],kern,mode='same')

    val = np.power(10.,(-np.linspace(1,n_s-1,n_s)/(n_s-1)*dynamic))

    A0 = np.random.randn(n_obs,n_s)
    A0 = np.dot(A0,np.diag(1./np.sqrt(np.sum(A0*A0,axis=0))))

    S0= np.dot(np.diag(1./np.sqrt(np.sum(S*S,axis=1))),S)
    S0 = np.dot(np.diag(val),S0)
    X0 = np.dot(A0,S0)

    N = np.random.randn(n_obs,t_samp)
    sigma_noise = np.power(10.,(-noise_level/20.))*np.linalg.norm(X0,ord='fro')/np.linalg.norm(N,ord='fro')
    N = sigma_noise*N

    X = X0 + N

    return X,X0,A0,S0,N,sigma_noise,kern

def Make_Experiment_Exact(n_s=2,n_obs=2,t_samp=1024,noise_level=40,dynamic=10,ptot=0.1,cd=1):

    S = np.zeros((n_s,t_samp))

    p_active = np.int(np.floor(t_samp*ptot))

    for r in range(0,n_s):
        ind = randperm(t_samp)
        S[r,ind[0:p_active]] = np.random.randn(1,p_active)

    val = np.power(10,(-np.linspace(1,n_s-1,n_s)/(n_s-1)*dynamic))

    A0 = Get_MixmatCond(n_s,n_obs,cd)

    S0= np.dot(np.diag(1./np.sqrt(np.sum(S*S,axis=1))),S)
    S0 = np.dot(np.diag(val),S0)
    X0 = np.dot(A0,S0)

    N = np.random.randn(n_obs,t_samp)
    sigma_noise = np.power(10,(-noise_level/20.))*np.linalg.norm(X0,ord='fro')/np.linalg.norm(N,ord='fro')
    N = sigma_noise*N

    X = X0 + N

    return X,X0,A0,S0,N


def Make_Experiment_GG(n_s=2,n_obs=2,t_samp=1024,noise_level=40,dynamic=0,CondNumber=1,alpha=2):

    S = generate_2D_generalized_gaussian(n_s, t_samp, alpha=alpha)

    val = np.power(10,(-np.linspace(1,n_s-1,n_s)/(n_s-1)*dynamic))

    A0 = np.random.randn(n_obs,n_s)

    if CondNumber == 1:
        Uq,Sq,Vq = np.linalg.svd(A0)
        A0 = Uq[:,0:n_s]

    if CondNumber > 1:
        for r in range(10):
            Uq,Sq,Vq = np.linalg.svd(A0)
            Sq = 1./np.linspace(1,CondNumber,n_s)
            A0 = np.dot(Uq,np.dot(np.diag(Sq),Vq.T))
            A0 = np.dot(A0,np.diag(1./np.sqrt(np.sum(A0*A0,axis=0))))

    S0= np.dot(np.diag(1./np.sqrt(np.sum(S*S,axis=1))),S)
    S0 = np.dot(np.diag(val),S0)
    X0 = np.dot(A0,S0)

    N = np.random.randn(n_obs,t_samp)
    sigma_noise = np.power(10.,(-noise_level/20.))*np.linalg.norm(X0,ord='fro')/np.linalg.norm(N,ord='fro')
    N = sigma_noise*N

    X = X0 + N

    return X,X0,A0,S0,N


def Gen_BG_Sources(n=2,t=1024,p=0.1):

    S = np.zeros((n,t))

    K = np.floor(p*t)

    for r in range(0,n):
        I = randperm(t)
        S[r,I[0:K]] = np.random.randn(K)

    return S

def Gen_BG_Mixtures(m=2,n=2,t=1024,p=0.1):

    S = Gen_BG_Sources(n,t,p)

    A = np.random.randn(m,n)

    A = np.dot(A,np.diag(1./np.sqrt(np.sum(A*A,axis=0))))

    X = np.dot(A,S)

    return X,A,S

def Gen_BG_Mixtures_Outliers(m=2,n=2,t=1024,p=0.1,po=0.01):

    S = Gen_BG_Sources(n,t,p)

    O = np.random.randn(m,t)
    K = np.floor(po*t*m)
    I = np.sort(abs(np.reshape(O,(1,m*t))))
    thrd = I[0,m*t - K]
    O[abs(O) < thrd] = 0

    A = np.random.randn(m,n)

    A = np.dot(A,np.diag(1./np.sqrt(np.sum(A*A,axis=0))))

    X = np.dot(A,S) + O

    return X,A,S,O

################# CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)

def CorrectPerm(cA0,S0,cA,S,incomp=0):

    A0 = cp.copy(cA0)
    A = cp.copy(cA)

    nX = np.shape(A0)

    IndOut = np.linspace(0,nX[1]-1,nX[1])
    IndE = np.linspace(0,nX[1]-1,nX[1])

    if incomp==0:

        for r in range(0,nX[1]):
            A[:,r] = A[:,r]/(1e-24+lng.norm(A[:,r]))
            A0[:,r] = A0[:,r]/(1e-24+lng.norm(A0[:,r]))

        Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))

        Sq = np.ones(np.shape(S))
        ind = np.linspace(0,nX[1]-1,nX[1])

        for ns in range(0,nX[1]):
            indix = np.where(Diff[ns,:] == max(Diff[ns,:]))[0]
            ind[ns] = indix[0]

        Aq = A[:,ind.astype(int)]
        Sq = S[ind.astype(int),:]

        A0q = A0
        S0q = S0

        IndE = ind.astype(int)

        for ns in range(0,nX[1]):
            p = np.sum(Sq[ns,:]*S0[ns,:])
            if p < 0:
                Sq[ns,:] = -Sq[ns,:]
                Aq[:,ns] = -Aq[:,ns]

    else:

        n = nX[1]
        n_e = np.shape(A)[1]   # Estimated number of sources

        for r in range(n):
            A0[:,r] = A0[:,r]/(1e-24+lng.norm(A0[:,r]))

        for r in range(n_e):
            A[:,r] = A[:,r]/(1e-24+lng.norm(A[:,r]))

        Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))

        if n_e > n:

            # the number of source is over-estimated
            # Loop on the sources

            IndTot = np.argsort(np.max(Diff,axis=1))[::-1]
            ind = np.linspace(0,n-1,n)

            m_select = np.ones((n_e,))

            for ns in range(0,n):
                indS = np.where(m_select > 0.5)[0]
                indix = np.where(Diff[IndTot[ns],indS] == max(Diff[IndTot[ns],indS]))[0]
                ind[IndTot[ns]] = indS[indix[0]]
                m_select[indS[indix[0]]] = 0

            Aq = A[:,ind.astype(int)]
            Sq = S[ind.astype(int),:]
            A0q = A0
            S0q = S0

        if n_e < n:

            # the number of source is under-estimated
            # Loop on the sources

            IndTot = np.argsort(np.max(Diff,axis=0))[::-1]
            ind = np.linspace(0,n_e-1,n_e)

            m_select = np.ones((n,))

            for ns in range(0,n_e):
                indS = np.where(m_select > 0.5)[0]
                indix = np.where(Diff[indS,IndTot[ns]] == max(Diff[indS,IndTot[ns]]))[0]
                ind[IndTot[ns]] = indS[indix[0]]
                m_select[indS[indix[0]]] = 0

            A0q = A0[:,ind.astype(int)]
            S0q = S0[ind.astype(int),:]
            Aq = A
            Sq = S

        IndE = ind.astype(int)

        #Aq = A[:,ind.astype(int)]
        #Sq = S[ind.astype(int),:]

    return A0q,S0q,Aq,Sq,IndE

################# CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)

def EvalCriterion(A0,S0,A,S):

    from copy import deepcopy as dp

    n = np.shape(A0)[1]
    n_e = np.shape(A)[1]

    incomp = 0
    if abs(n-n_e) > 0:
        incomp=1

    gA0,gS0,gA,gS,IndE = CorrectPerm(A0,S0,A,S,incomp=incomp)
    n = np.shape(gA0)[1]
    n_e = np.shape(gA)[1]
    Diff = abs(np.dot(np.linalg.inv(np.dot(gA0.T,gA0)),np.dot(gA0.T,gA)))
    Diff2 = dp(Diff)
    z = np.min([n,n_e])

    Diff = Diff - np.diag(np.diag(Diff))
    Diff = Diff.reshape((z**2,))

    p = (np.sum(Diff))/(z*(z-1))
    pmax = np.max(abs(Diff))
    pmin = np.min(abs(Diff2))
    pmed = np.median(abs(Diff))

    Q = abs(np.arccos(np.sum(gA0*gA,axis=0))*180./np.pi)

    crit = {"ca_mean":p,"ca_med":pmed,"ca_max":pmax,"SAE":Q}

    return crit

################# COMPARING THE SUPPORT OF THE SOURCES

def Supp_Compare(S0,S):

    import numpy as np
    nS = np.shape(S0)
    CardC = np.zeros((nS[0],))

    for r in range(nS[0]):

        I0 = np.argsort(abs(S0[r,:]))[::-1]
        I = np.argsort(abs(S[r,:]))[::-1]
        R = np.cumsum(I0) - np.cumsum(I)
        J = np.where(R != 0)[0]
        CardC[r] = J[0]

    return CardC

################# COMPUTE THE SDR

def BSSEval(A0,S0,gA,gS):

     import numpy as np
     from copy import deepcopy as dp

     na = np.shape(A0);
     n = na[1]
     ns = np.shape(S0)
     t = ns[1]

     A,S = CorrectPerm(A0,S0,gA,gS)

     s_target = np.zeros((n,t))
     e_interf = dp(s_target)
     e_noise = dp(s_target)
     e_artif = dp(s_target)
     s_target = np.dot(np.dot(np.diag(1./np.diag(np.dot(S0,S0.T))),np.diag(np.diag(np.dot(S0,S.T)))),S0)

     Ps = np.dot(np.dot(np.linalg.inv(np.dot(S0,S0.T)),np.dot(S0,S.T)),S0)

     e_interf = Ps - s_target

     Psn = dp(Ps)

     SNR = 1e24

     e_artif = S - Psn;

     SDR = 10*np.log10(np.linalg.norm(s_target,ord='fro')**2/(np.linalg.norm(e_interf,ord='fro')**2 + np.linalg.norm(e_artif,ord='fro')**2 + np.linalg.norm(e_noise,ord='fro')**2))
     SIR = 10*np.log10(np.linalg.norm(s_target,ord='fro')**2/(np.linalg.norm(e_interf,ord='fro')**2));
     SAR = 10*np.log10((np.linalg.norm(s_target,ord='fro')**2 + np.linalg.norm(e_interf,ord='fro')**2 + np.linalg.norm(e_noise,ord='fro')**2)/(np.linalg.norm(e_interf,ord='fro')**2))

     sRES = np.zeros((n,3))
     for r in range(n):
         sRES[r,0] = 10*np.log10(np.linalg.norm(s_target[r,:],ord=2)**2/(np.linalg.norm(e_interf[r,:],ord=2)**2 + np.linalg.norm(e_artif[r,:],ord=2)**2 + np.linalg.norm(e_noise[r,:],ord=2)**2))
         sRES[r,1] = 10*np.log10(np.linalg.norm(s_target[r,:],ord=2)**2/(np.linalg.norm(e_interf[r,:],ord=2)**2));
         sRES[r,2] = 10*np.log10((np.linalg.norm(s_target[r,:],ord=2)**2 + np.linalg.norm(e_interf[r,:],ord=2)**2 + np.linalg.norm(e_noise[r,:],ord=2)**2)/(np.linalg.norm(e_interf[r,:],ord=2)**2))


     return SDR,SAR,SIR,sRES

################# COMPUTE THE SDR

def RecoveredA(MMC,tol=5):

    """
    Reliability measure based on the ability to get a mixing matrix that is tol deg. closer to either the optimal or the best run

    Input: MMC - N x M x N_MC

    Output: ToBest
            ToOpt

    """

    nm = np.shape(MMC)

    ToBest = np.zeros((nm[0],nm[1]))
    ToOpt = np.zeros((nm[0],nm[1]))

    for q in range(nm[0]):
        for r in range(nm[1]):
            Opt = np.min(MMC[q,r,:])
            I = np.where(abs(MMC[q,r,:] - Opt) < tol)
            ToBest[q,r] = np.size(np.where(abs(MMC[q,r,:] - Opt) < tol))/np.double(nm[2])
            ToOpt[q,r] = np.size(np.where(MMC[q,r,:] < tol))/np.double(nm[2])

    return ToBest,ToOpt
