# -*- coding: utf-8 -*-
"""
This file contains the functions used during the minimization of the beta-divergence. These functions are called from the file BetaD_ICA.
This implementation is the python version of the Matlab code proposed by Neil Gadhok (http://home.cc.umanitoba.ca/~kinsner/projects/software/index.html).

"""
import numpy as np
import scipy as sp
from parameters import dS

#################################################################################################################################################################################################################################


def formv2(Wf):
    '''This function takes a n by n matrix W and reshapes it as a n*n column vector. 
    Input: Wf, a n by n matrix [[w11, w12,...],[w21,w22,..], [...,wnn]]
    Output: [w11,w12,w13,.., wnn].T, column vector.
    '''
    return Wf.reshape((np.shape(Wf)[0]**2,1))
#################################################################################################################################################################################################################################


def  deformv2(vf):
    '''This function takes a n*n column vector and creates a n by n matrix.
    Input: 
    vf: a n*n vector [v1,v2,..,vnn]
    Output: square matrix W=[[v1,v2,..],[vn,vn+1,...],[..,vnn]], of size n by n
    '''
    return vf.reshape( (  np.sqrt(np.shape(vf[1])), np.sqrt(np.shape(vf[1]))  )   )
    
  #################################################################################################################################################################################################################################
  

def formv(Wf,uf):
     ''' This function takes a n by n matrix Wf and a n vector uf and creates a long (n*(n+1)) column vector. 
     Input:
     Wf: a n by n matrix [[w11, w12,...],[w21,w22,..], [...,wnn]]
     uf: a vector of size n (u1,u2,..un)
     Output: the column vector of size (n*(n+1)): [w11,w12,..w21,w22,..,wnn,u1,..un]
     ''' 
     return np.vstack((Wf.reshape((np.size(uf)**2,1)),uf.reshape(np.size(uf),1)))
     
     
#################################################################################################################################################################################################################################
     


def deformv(vf):
    ''' This function takes a n*(n+1) column vector and create a n*n matrix and a column vector of size n.
    Input: 
    vf: a n*(n+1) vector of the form [w11,w12,..w21,w22,..,wnn,u1,..un]
    Output:
    Wf: a n by n matrix [[w11, w12,...],[w21,w22,..], [...,wnn]] and uf a column vector [u1,..un]
     ''' 


    
    Wf=vf[0:dS['n']**2,:].reshape((dS['n'],dS['n']))
    uf=vf[dS['n']**2:dS['n']**2+dS['n'],:].reshape((dS['n'],1))  
    
    

    return [Wf,uf]


#################################################################################################################################################################################################################################


def BetaD_Deriv(xf,Wf,uf,Bf,p_i):
    '''
    Computes the derivative, for a value of beta strictly larger than 0.
    
    Input:
    xf: the observation
    Wf: demixing matrix
    uf: mean
    Bf: value of the beta parameter
    p_i: index indicating which a-priori we use for the sources (2 for sparse sources)
    Output: Jf, the first derivative
    '''

    mf = np.size(uf); #number of sources (=number of observations)
    nf = np.shape(xf)[1]; # number of samples
   
    Jf = np.zeros((mf**2+mf,1)); # derivative
 
        

    WTf = (np.linalg.inv(Wf)).T; #Transpose of the inverse of the unmixing matrix
    
#       
    #compute ro(x(t),W,u)
    pif = np.ones((1,nf)); 
    for i in range ( 0,mf):
        for j in range (0,nf):
            pif[0,j] = pif[0,j] *(  BetaD_p_i( extraRow(Wf,i).dot(extraCol(xf,j)) - uf[i,0] , p_i[i,0] ));
    pif=pif*np.abs(np.linalg.det(Wf))
    
    # Compute the derivatives  (Assume that bBeta(W) is nul, as proposed in the article of Mihoko and Eguchi)
    Wdf = np.zeros((mf,mf,nf));
    for j in range (0,nf):
         Wdf[:,:,j] = (pif[0,j]**Bf) *( np.eye(mf)  - ( (BetaD_h_i( Wf.dot(extraCol(xf,j)) - uf, p_i )) .dot( (Wf.dot(extraCol(xf,j))).T) )).dot(WTf);
    
    Wdf = (1./nf)*np.sum(Wdf,2);

    
    W = Wdf 
    
    udf = np.zeros((mf,nf));
    for j in  range(0,nf):
        udf[:,j] = (pif[0,j]**Bf) *( (BetaD_h_i( Wf.dot(extraCol(xf,j)) - uf, p_i ))).reshape(mf,);

    u = (1./nf)*np.sum(udf,1);
  

    
    Jf = -formv(W,u); # The (-) are needed for minimization.
  
    return Jf
        
#################################################################################################################################################################################################################################


def BetaD_h_i(z,p_i):
    ''' This function takes a matrix and the indicator for the prior on the sources and returns
    the value of the derivative of the log(pi) (-pi'(zi)/pi(zi)).
    Input:
    z: a matrix (or a vector, a scalar)
    p_i: the indicator of the prior used for the sources (2 for sparse sources)
    Output: the value of the derivative
    '''



    result = np.zeros((np.size(p_i),1));
    
    for i in range( 0,np.size(p_i)):
        
        if p_i[i]==1:
       
            result[i,0] =  z[i,0]**3;
        else :
            result[i,0] =np.tanh(z[i,0])
       
        
    return result

#################################################################################################################################################################################################################################



def BetaD_p_i(z,p_i):
    ''' This function takes a matrix and the indicator for the prior on the sources and returns
    the value of the density function.
    Input:
    z: a matrix (or a vector, a scalar)
    p_i: the indicator of the prior used for the sources (2 for sparse sources)
    Output: the value of the density function
    '''

    
    if p_i==1:
        c=np.sqrt(2)/(sp.special.gamma(0.25)**(-1))
        d=0.25
        result=c*np.exp(-d*(z**4))
    else:
        c=1./np.pi
        result=c/(np.cosh(z))
        
    return result
     
#################################################################################################################################################################################################################################
     
     

def BetaD_bB(W,B,p_i):
    ''' Computes the value of b_beta(W), according to the priors of the sources. This is not used in our implementation: if the sources are assumed to be sparse, 
    it returns zero, the approximation proposed by Mihoko and Eguchi in their paper.
    '''
    if p_i==1:
        d=0.25
        result=(np.abs(np.linalg.det(W))**B) * ( ((2*(d**0.25)*(sp.special.gamma(0.25)**(-1)))**B) * ((1+B)**(-0.25) )) * (1./(B+1));
        
    if p_i==2:
        result = 0;#Return zero because there is not easy closed form to use
    
    return result 


#################################################################################################################################################################################################################################


def BetaD_Lbeta(x,W,u,B,p_i):
    ''' Calculate the quasi beta-likelihood function.
    Input:
    x: the observations
    W: the demixing matrix
    u: the mean
    B the value of the parameter beta
    p_i: the indicators of the priors used for the sources
    Output: the associated quasi beta-likelihood function.
    '''
    
    m = np.size(u);#number of sources (or of observations)
    n = np.shape(x)[1]; #number of samples
    
    #Caluclate rO
    pif = np.ones((1,n));
    for i in range( 0,m):
        for j in range (0,n):
            pif[0,j] = pif[0,j] *(  BetaD_p_i( extraRow(W,i).dot(extraCol(x,j)) - u[i,0] , p_i[i,0] ));

    r_0 = np.abs(np.linalg.det(W))*pif;

   
    #the value of b_beta is set to zero, as it is proposed by M.Mihoko and S. Eguchi since there is no closed-form for sparse sources.
    L = ( (1./n)*(1./B)*np.sum((r_0)**B) )  - ((1.-B)/B);
#    
    Lbeta = -L;
    
    return Lbeta
#################################################################################################################################################################################################################################

def extraCol(matM,indexCol):
    '''Extract a column of a matrix and return this column vector.
    Input:
    matM: a matrix
    indexCol: the index of the column to extract
    Output: the column vector [matM]_(:,indexCol)
    '''
    return matM[:,indexCol].reshape((np.shape(matM)[0],1))
#################################################################################################################################################################################################################################
    
def extraRow(matM,indexRow):
    '''Extract a row of a matrix and return this row vector.
    Input:
    matM: a matrix
    indexRow: the index of the row to extract
    Output: the column vector [matM]_(indexRow,:)
    '''
    return matM[indexRow,:].reshape((1,np.shape(matM)[1]))    
    