# -*- coding: utf-8 -*-
"""
Part of tools based on proximal calculus

Solves : Argmin_(x>=0) lam *||W o (S Phi^T) ||_1

both in 1D and 2D isotropic undecimated wavelets

@Author: J.Bobin - CEA -
"""

import numpy as np
from copy import deepcopy as dp
from pyGMCA.common.wavelets import pyStarlet as pys



def Threshold_1D(X,W=None,J=2,lam=0.1,optCoarse=1):

    """

    Code that performs signal thresholding in the 1D isotropic undecimated wavelets

    Inputs:

        X : Input data (m x n_samp)
        W   : weights (m x n_coef)
        J : number of scales
        lam : threshold
        optCoarse : if 0, set the coarse scale coefficients to 0
    Output:
        X_thrd : Output data (m x n_samp)

    """

    CardS=None

    nS = np.shape(X)
    n = nS[0]
    t = np.int(nS[1])

    Wt_c = pys.forward1d(X,J = J)
    S = Wt_c[:,:,0:J].reshape(n,J*t)

    for r in range(n):

        if W == None:

            if CardS is not None:
                Kval = np.int(CardS[r])
                I = np.argsort(abs(S[r,:]))[::-1]
                th = abs(S[r,I[Kval]])
            else:
                th = lam


            S[r,:] = (S[r,:] - th*np.sign(S[r,:]))*(abs(S[r,:]) - th > 0)


        else:
            S[r,:] = (S[r,:] - lam*W[r,:]*np.sign(S[r,:]))*(abs(S[r,:]) - lam*W[r,:] > 0)


    Wt_c[:,:,0:J] = S.reshape(n,t,J)

    if optCoarse == 0:
        Wt_c[:,:,J] = np.zeros(np.shape(Wt_c[:,:,J]))


    rec = pys.backward1d(Wt_c)



    return rec

def GFB_WSparse_Positivity_1D(X,W=None,CardS = None,J=2,lam = 1,niter=250,tol=1e-6,Analysis=False,verb=0,XisOndt=0,coarseScale=0,A=0):

    """

    Solves : Argmin_(x>=0) lam *||W o (S Phi^T) ||_1 using 1D wavelets

    Inputs:

        X : Input data (m x n_samp)
        W   : weights (m x n_coef)
        niter : number of iterations
        J : number of scales
        tol : limit on the stopping criterion
        lam : threshold
        Analysis : if set, rather uses a synthesis prior
        verb : enables verbose mode
        XisOndt: set to 1 if X is already in the wavelet domain
        coarseScale : contains the coarse scale of X if X is already in the wavelet domain

    Output:
        X_thrd : Output data (m x n_samp)

        if XisOndt = 1, additional outputs:
        coarseScale : coarse scale of X

    """
    if XisOndt==1:
        n_X = np.shape(X)
        m = n_X[0]
        t = np.shape(coarseScale)[1]
        Wt_c = np.zeros((m,t,J+1))
        Wt_c[:,:,0:J] = X.reshape(m,t,J)
        Wt_c[:,:,J] = np.dot(np.linalg.pinv(A),coarseScale) #coarseScale
        X = pys.backward1d(Wt_c)


    u = dp(X)
    v = dp(X)
    Xout = dp(X)


    Xold= dp(Xout)

    w_u = 0.8
    w_v = 0.2
    gamma = 0.9
    mu = 1.5

    it = 0
    Go_On = 1
    tk = 1

    while Go_On:

        it+=1

        dG = Xout - X

        # Update positivity

        Xt = 2.*Xout - u - gamma*dG
        Xt = Xt*(Xt > 0) - Xout
        u = u + mu*Xt

        # Update sparsity

        Xt = 2.*Xout - v - gamma*dG
        if Analysis:
            Xt = Analysis_prox(Xt,W = W,J=J,lam=lam) - Xout
        else:
            Xt = Threshold_1D(Xt,W=W,J=J,lam=lam,CardS=CardS) - Xout
        v = v + mu*Xt

        # Update Sout

        tkp = 0.5*(1+np.sqrt(1. + 4.*(tk**2)))

        Xt = w_u*u + w_v*v

        Xout = Xt + (tk - 1.)/tkp*(Xt - Xold)

        diffX = np.sum(abs(Xold - Xout))/(1e-24 + np.sum(abs(Xold)))
        Xold = dp(Xout)

        if verb:
            print("It. #",it," - relative variation : " ,diffX)

        if diffX < tol:
            Go_On=0
        if it > niter:
            Go_On=0

    if XisOndt==1:
        Xout = pys.forward1d(Xout,J = J)
        coarseScale = Xout[:,:,J]
        Xout = Wt_c[:,:,0:J].reshape(m,J*t)

        return Xout,coarseScale

    else:

        return Xout

def Analysis_prox(X,W = None,niter=1000,J=2,alpha=0.9,lam=1,tol=1e-4,verb=0):

    """

    Proximal operator of the analysis prior in the 1D IUWT

    """

    nS = np.shape(X)
    n = nS[0]
    t = np.int(nS[1])

    mw = pys.forward1d(X,J = J)
    mw[:,:,J] = 0.

    u = np.sign(mw[:,:,0:J].reshape(n,t*J))*lam
    u_old = dp(u)

    if W == None:
        W = 1.

    L = np.max(W)**2

    it = 0
    tk = 1.

    Go_On = 1

    while Go_On:

        it += 1

        # Compute the gradient

        Wu = W*u

        mw[:,:,0:J] = Wu.reshape(n,t,J)
        dG =  X - np.sum(mw,axis=2)
        mw = pys.forward1d(dG,J = J)
        mw[:,:,J] = 0.
        du = - W*mw[:,:,0:J].reshape(n,t*J)

        # Gradient descent

        up = u - alpha/L*du

        # L_inf constraint

        for r in range(n):
            ind = np.where(up[r,:] > lam)[0]
            if np.size(ind)>0:
                up[r,ind] = lam
            ind = np.where(up[r,:] < -lam)[0]
            if np.size(ind)>0:
                up[r,ind] = -lam

        tkp = 0.5*(1. + np.sqrt(1 + 4.*tk*tk))

        u = up + (tk - 1.)/tkp*(up - u_old)

        diffX = np.linalg.norm(up - u_old)/(1e-12 + np.linalg.norm(u_old))
        tk = dp(tkp)
        u_old = dp(up)

        if verb:
            print("It. #",it," - relative variation : " ,diffX)

        if diffX < tol:
            Go_On=0
        if it > niter:
            Go_On=0

    Wu = W*u
    mw[:,:,0:J] = Wu.reshape(n,t,J)

    return X- np.sum(mw,axis=2)
