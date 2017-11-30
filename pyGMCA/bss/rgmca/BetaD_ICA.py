# -*- coding: utf-8 -*-
"""
Minimization of the beta-divergence with BFGS, with beta strictly larger than zero.
This implementation comes from the Matlab implementation proposed by N. Gadhok (http://home.cc.umanitoba.ca/~kinsner/projects/software/index.html),
presented in 'An implementation of beta-divergence for Blind Source Separation', N. Gadhok and W. Kinsner, IEEE CCECE/CCGEI, Ottawa, May 2006.
The functions, called by BetaD_ICA are placed in the file FunBetaD.
The method was proposed in 'Robust Blind Source Separation by Beta Divergence' by M.Mihoko and S. Eguchi in Neural Computation 2002.
"""
import numpy as np
from FunBetaD import*
from parameters import dS,aS,pS
from errors import reOrder

def BetaD_ICA(xori,Aori):
    '''
    Minization of the beta-divergence via BFGS. This implementation comes from the Matlab implementation proposed by N. Gadhok (http://home.cc.umanitoba.ca/~kinsner/projects/software/index.html) 
    The modifications of the original code are identified as so. In particular, this version performs the minimization of the beta-divergence for different values of the beta-divergence, and returns
    the mixing matrix with the minimal error.
    Input:
    xori: observations
    Aori: initial mixing matrix, to be retrieved.
    Output:
    S: estimated sources
    A: estimated mixing matrix
    '''
    
    #Setup parameters
    stopping_criterion=1e-5 #convergence criteria
    maxiteration_BFGS=aS['iteMaxBeta'] #maximal number of iterations
    maxiteration_linesearch=20#maximal number of iterations for the line search
    
    
    
    #variables
    p_i=2*np.ones((dS['n'],1))#should be set to 2 for sparse data
    x=xori.copy()#copy of the observations (assumed to have zero mean)
    xm=x.copy()#copy of the observations (assumed to have zero mean)


    ###
    #If the number of mesures is strictly larger than the number of sources: a PCA is performed
    #for the reduction of dimensions
    ### 

    if dS['m']>dS['n']: 
        U,d,V=np.linalg.svd((x).T )
        d2=np.zeros((dS['n'],dS['t'])).T
        for index in range(0,dS['n']):
            d2[index,index]=d[index]
        x=(U[:,:].dot(d2)).T #projected observations
    
    Wred=x.dot(np.linalg.pinv((xm)))#projection matrix, identity if n=m

    uori=np.mean(x,1).reshape((dS['n'],1)) #mean of x (should be close to 0)
    Wori=np.linalg.inv(Wred.dot(Aori)) #initialization of W, which should be close to the inverse of A
   
    W1=Wori.copy()#initiliazation of demixing matrix used in the algo.
    u1=uori.copy()#initiliazation of mean used in the algo.
    
    Wfinal=Wori#initialization of the returned results
    ufinal=uori#initialization of the returned results
    
    
    resultat=1e5 #initialization of the "best" error
    index=0#index of the loop
    (m,t)= np.shape(x);# dimension of the matrix
    G = np.eye(m**2+m) #initialization of the inverse of Hessian;


    #Each following loop performs a minimization given a different value of beta.
    while index<400:
        
        B=0.005+index*0.002 #current value of beta
        
        #Initialization of the variables and parameters
        W = W1.copy();
        u = u1.copy();

        v  = formv(W,u); # v is a column vector corresponding of the concatenation of W and u
        v1 = np.zeros((m**2+m,1)); # v1 will be used an update of v

        i = 1;#index for the minimization
        fxO=0# value of the cost function at the previous iteration
        fx=10#value of the cost function
        fxn=10 # vlaue of the cost function for the line search
        flag=1 #Set to zero when the line search has failed
        prev=1 #Previous decrease of the cost function
        
        ###
        #Minimization of the beta-divergence with the corresponding value of beta
        ###
        
        while ((prev>stopping_criterion or (np.abs(fx-fxO)/(np.abs(fxO)+1e-16)>stopping_criterion)) and (i < maxiteration_BFGS )) or i<10 : 
            #Main loop, which stop if convergence has been reached (decrease in the cost function smaller than stopping-criterion, twice)
            prev=np.abs(fx-fxO)/np.abs(fxO) #cost function decrease
            fxO=fx #previous value of the cost function
            if i>1 and flag:
                #update of the variables (only if the line search was successful (implementation difference with the original implementation))   
                v = v1;  #Update of v           
                [W,u] = deformv(v);  #Update of W and u
                
         
            
            p = -G.dot(BetaD_Deriv(x,W,u,B,p_i));#New direction given by p
            
            
            ###
            #Begin of the line search
            ###
            alpha = 1;  #line search coeffcient
            sigma = 1.e-4; #minimal decrease
             
            alphamin = 1.e-4;  #minimal coefficient for alpha
            pgx = p.T.dot(BetaD_Deriv(x,W,u,B,p_i)); #decrease of the function given the direction
            v1 = v + alpha*p;   #first update of v
           
            [W2,u2] = deformv(v1); #corresponding W and u

            fxn = BetaD_Lbeta(x,W2,u2,B,p_i); #cost function for the first update
            fx = BetaD_Lbeta(x,W,u,B,p_i);#actual cost function 

                   
            i_linesearch = 1;#index for the line-search
            
            while( fxn > fx + sigma*alpha*pgx) and (alpha > alphamin) and (i_linesearch < maxiteration_linesearch):
                mu = -0.5 * pgx * alpha / (fxn - fx - alpha * pgx );
                if mu < .01:
                    mu = .5;  #don't trust quadratic interpolation from far away
                
                alpha = mu * alpha; 
                v1 = v + alpha*p;  
                [W2,u2] = deformv(v1); 
                fxn = BetaD_Lbeta(x,W2,u2,B,p_i);
                i_linesearch +=1
            
          
            if (fxn > fx + sigma*alpha*pgx) or np.sum(np.isnan(W2)==True)>0 or np.sum(np.isnan(BetaD_Deriv(x,W,u,B,p_i) - BetaD_Deriv(x,W2,u2,B,p_i))==True)>0:
                # Line search failed. Thus, the BFGS approx of the Hessian has become corrupt, or we are near the minimum (flat valley) and the step size is too big. So restart.
                G = np.eye(m**2+m);
                p = -G.dot((BetaD_Deriv(x,W,u,B,p_i)));
                alpha = 1;
                flag=0
                prev+=1
                
                
             
               
            
            #End Line search.Update accordingly the variables
 
            else:
                flag=1
                v1 = v + alpha*p; 
       
                [W,u] = deformv(v1); 
                [W1,u1] = deformv(v);  
                s = alpha*p;
                y  = BetaD_Deriv(x,W,u,B,p_i) - BetaD_Deriv(x,W1,u1,B,p_i);
                I  = np.eye(m**2+m);
                G =  (I - (s.dot(y.T)) / ((y.T).dot(s)))   .dot(G.dot((I - (y.dot(s.T))/((y.T).dot(s)) ))) + (s.dot(s.T))/((y).T.dot(s)); #update inverse of the Hessian
            i+=1;
  
            
            
        [Wfinal,ufinal] = deformv(v1); #Final variables
        
         
        ###
        #Compute the corresponding sources and mixing matrix, and then the associated errors
        ###
        A=np.linalg.inv(Wfinal)#Inverse of W
        A=np.linalg.pinv(Wred).dot(A)# if dimension reduction was needed
        for indexC in range(0, dS['n']):
            A[:,indexC]=A[:,indexC]/np.linalg.norm(A[:,indexC]) #normalization of the columns
        S=np.sign(np.linalg.pinv(A).dot(xori))*np.maximum(np.abs(np.linalg.pinv(A).dot(xori))-aS['kSMax']*dS['ampliN'],0) #corresponding sources

        S,A=reOrder(A,S,Aori,S)# reorder the factorization
        D=np.sum(np.abs(np.linalg.pinv(A).dot(Aori)-np.eye(np.shape(Aori)[1]))/(dS['n']**2)) #Delta A error

     
        if D<=resultat:
           resultat=  D
           Afin=A.copy()#keep the corresponding mixing matrix
           Sfin=S.copy()#keep the corresponding sources
           index+=1
           if pS['verboseBeta']:
               print 'Beta-div.: current best result: ', D, 'at beta= 0.005+ 0.002*' , index, 'in ', i ,' iterations'
           

        elif D<resultat*1.5:
            index+=1
        else:
            #exit
            if pS['verboseBeta']:
               print 'exit at ', index ,'th index'
            index+=400
            
           
            

    return  Sfin, Afin