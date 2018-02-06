#  Version
#    v1 - September,28 2017 - C.Kervazo - CEA Saclay
#
#


#%%

from  pyGMCA.bss.bgmca import bgmca as cgb
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import copy as cp
from scipy import special
import scipy.linalg as lng
from scipy import signal as sng
import sample_nmr_spectra as snp


#%%
def launch_GMCA(n_sources=[50], blocksize=[1,2,3,4,5,7,10,13,16,18,20,28,36,45,50], numExp=1,N_MC=1,sauv=2,init=1,dataType=1,optBlock=0,folderInitSave='Results/', t_samples=1000,ptot=0.1,cd=1,SNR=120,verPos=0,J=0):

    """
    Usage:
    launch_GMCA(numExp=1,N_MC=1,sauv=2,init=1,optBlock=0,folderInitSave=0, t_samples=0,ptot=0.1,cd=1,colSize=0,palm=0,SNR=120,optWhite=0,verPos=0,J=0)

    Inputs:
    n_sources : list of intergers (list containing the number of sources of the experiments)
    blocksize : list of intergers (list of the different block sizes to be tested. For a given number of sources n in the n_sources list, valid if the blocksizes are in [1,n])
    numExp : int (number of the experiment (used for the seed of the random data generation))
    N_MC : int (number of Monte-Carlo simulations)
    sauv : int (0 : no saving, 1 : save the results, 2 : save all including data
    init : boolean (0: load some already existing X,X0,A0,S0,N data matrices, 1 : creation of new random data)
    dataType : int in {1,2,3} (1: Bernouilli Gaussian data, 2: Generalized Gaussian data (approximately sparse), 3: realistic 1H NMR data)
    optBlock : int in {1,2,3} (0 : blocks are crated randomly, 1 : creation according to the "correlation" of the sources, 2 : according to the angle between the columns of A)
    folderInitSave : string (name of the folder use to save or load the data. This folder must contain 2 subfolders 'Data' and 'MatricesAS')
    t_samples : int (number of samples for each source)
    ptot : float in (0,1) (sparsity of the sources. More sparse for low values:
                            if dataType = 1: ptot = number of non-zero coefficients
                            if dataType = 2: ptot = parameter of Generalized Gaussian: 1 => Laplacian, 2 = Gaussian
                            if dataType = 3: ptot is not used)
    cd : int >= 1 (condition number of A)
    SNR : float (signal to noise ratio. Do not use int)
    verPos : int in {0,1,2,3} (0 : no non-negativity, verPos = 1 : wavelets + non-negativity in the direct domain, verPos = 2 : wavelets without non-negativity, verPos = 3 : non-negativity)
    J : int (number of wavelets scales if applicable)

    Outputs:
    C_hyb : len(blocksize) x len(n_sources) x N_MC array (contains the C_A values for all the blocksizes and different number of sources for each monte carlo experiment)
    S_gmca : n x t array (found sources. Relevant only if len(blocksize) = len(n_sources) = N_MC = 1)
    A_gmca : m x n array (found mixing matrix. Relevant only if len(blocksize) = len(n_sources) = N_MC = 1)
    """

    nitGMCA = 1000 # Number of iterations, should be fixed so that nit > 100*n_sources/blocksize
    nitPALM = nitGMCA*10




    pltRes = 1 # Boolean value




    # bGMCA is randomly initialized
    initGMCA = 1



    if init == 0 and sauv == 2:
        print('Writing and reading of data at the same time')
        return



    # Creation of strings for the file name
    cdStr = cd


    if ptot == 1:
        ptotStr = '1'
    else:
        ptotStr = '0_' + str(ptot-int(ptot))[2:]


    sourcesStr = str(n_sources)

    # Initializations
    C_hyb = np.zeros((len(blocksize),len(n_sources),N_MC))
    boolVal = 1



    for It_MC in range(0,N_MC): # Loop for performing N_MC monte carlo simulations
        valSeed = numExp*N_MC + It_MC + 1
        np.random.seed(valSeed)
        print('Value of the seed %s'%(valSeed))
        for R_t in range(0,len(n_sources)): # Loop on the different number of sources to be tested

            n = n_sources[R_t]

            # Generating some random sparse sources with random mixing matrices
            # in the determined case (m=n)
            if init ==1:
                if dataType == 1:
                    X,X0,A0,S0,N = Make_Experiment_Exact(n_s=n,n_obs=n,t_samp=t_samples,noise_level=SNR,dynamic=0,ptot=ptot,cd=cd)
                elif dataType == 2:
                    X,X0,A0,S0,N = Make_Experiment_GeneralizedGaussian(n_s=n,n_obs=n,t_samp=t_samples,noise_level=120,dynamic=0,ptot=1,cd=cd,alpha=ptot)
                elif dataType == 3:
                    X,X0,A0,S0,N = Make_Experiment_HMR(n_s=n,n_obs=8*n,t_samp=t_samples,noise_level=SNR,standDev = 3,peak_width=3)


            if init == 0: # If we do not create new data matrices but rather use some already saved matrices
                X = sio.loadmat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_X_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,0,1,SNR,3,J),{'donnees':X})
                X = X['donnees']
                X0 = sio.loadmat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_X0_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,0,1,SNR,3,J),{'donnees':X0})
                X0 = X0['donnees']
                A0 = sio.loadmat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_A0_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,0,1,SNR,3,J),{'donnees':A0})
                A0 = A0['donnees']
                S0 = sio.loadmat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_S0_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,0,1,SNR,3,J),{'donnees':S0})
                S0 = S0['donnees']
                N = sio.loadmat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_N_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,0,1,SNR,3,J),{'donnees':N})
                N = N['donnees']



            if sauv == 2 and init ==  1: # The data matrices are saved
                sio.savemat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_X_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_optWhite%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,optBlock,initGMCA,SNR,0,verPos,J),{'donnees':X})
                sio.savemat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_X0_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_optWhite%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,optBlock,initGMCA,SNR,0,verPos,J),{'donnees':X0})
                sio.savemat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_A0_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_optWhite%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,optBlock,initGMCA,SNR,0,verPos,J),{'donnees':A0})
                sio.savemat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_S0_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_optWhite%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,optBlock,initGMCA,SNR,0,verPos,J),{'donnees':S0})
                sio.savemat(folderInitSave + 'Data/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_N_exp%s_MC%s_sourceN%s_optBlock%s_init%s_SNR%s_optWhite%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,optBlock,initGMCA,SNR,0,verPos,J),{'donnees':N})

            for R_p in range(0,len(blocksize)): # Loop on the different block sizes to be tested
                minibatch = blocksize[R_p]

                if minibatch > n:
                    C_hyb[R_p,R_t,It_MC] = -1

                else:
                    if verPos == 0:# No wavelet and no non-negativity constraint
                        g_S_gmca,g_A_gmca,exception = cgb.bGMCA(X,n,Init=initGMCA,mints=1,nmax=nitGMCA,L0=0,verb=1,blocksize=minibatch,optBlock=optBlock,J=0)

                    elif verPos == 1:# Wavelet and non-negativity in the direct domain
                        g_S_gmca,g_A_gmca,exception = cgb.bGMCA_NMF_ondt_naif(X,n,Init=initGMCA,mints=1,nmax=nitGMCA,L0=0,verb=1,blocksize=minibatch,optBlock=optBlock,J=J)

                    elif verPos == 2:# Wavelets without non-negativity
                        g_S_gmca,g_A_gmca,exception,Sw = cgb.bGMCA(X,n,Init=initGMCA,mints=1,nmax=nitGMCA,L0=0,verb=1,blocksize=minibatch,optBlock=optBlock,J=J)

                    elif verPos == 3: # Non-negativity in the direct domain
                        g_S_gmca,g_A_gmca,exception = cgb.bGMCA(X,n,Init=initGMCA,mints=1,nmax=nitGMCA,L0=0,verb=1,blocksize=minibatch,optBlock=optBlock,J=0,optPos=1)


                    print('End of GMCA stage')
                    S_gmca = g_S_gmca
                    A_gmca = g_A_gmca



                    if exception != 0:
                        boolVal = 0
                        print('Exception = %s' %exception)




                    if verPos == 0:# No wavelet and no non-negativity constraint
                        g_S_gmca,g_A_gmca = cgb.PALM_NMF_MainBlock(X,n,A=cp.deepcopy(A_gmca),S=cp.deepcopy(S_gmca),kend=1,nmax=nitPALM,L0=0,blocksize=minibatch,tol=1e-12)
                    elif verPos == 1:# Wavelet and non-negativity in the direct domain
                        g_S_palm,g_A_gmca = cgb.PALM_NMF_MainBlock_prox(X,n,A=cp.deepcopy(A_gmca),S=cp.deepcopy(S_gmca),kend=1,nmax=nitPALM,L0=0,verb=0,blocksize=minibatch,tol=1e-12,J=J)
                    elif verPos == 2:# Wavelets without non-negativity
                        g_S_gmca,g_A_gmca = cgb.PALM_NMF_MainBlock(X,n,A=cp.deepcopy(A_gmca),S=cp.deepcopy(Sw),kend=1,nmax=nitPALM,L0=0,blocksize=minibatch,tol=1e-12,J=J)
                    elif verPos == 3: # Non-negativity in the direct domain
                        g_S_gmca,g_A_gmca = cgb.PALM_NMF_MainBlock(X,n,A=cp.deepcopy(A_gmca),S=cp.deepcopy(S_gmca),kend=1,nmax=nitPALM,L0=0,blocksize=minibatch,tol=1e-12,optPos=1)


                    S_hyb = g_S_gmca
                    A_hyb = g_A_gmca



                    try:
                        C_hyb[R_p,R_t,It_MC] = EvalCriterion(A0,S0,cp.deepcopy(A_hyb),cp.deepcopy(S_hyb))

                    except ValueError:
                        C_hyb[R_p,R_t,It_MC] = np.log10(-1)





                    if sauv > 0:# We save the A and S matrices
                        sio.savemat(folderInitSave + 'MatricesAS/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_S_gmca_exp%s_MC%s_sourceN%s_blocksizeNumber%s_optBlock%s_init%s_SNR%s_optWhite%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,R_p,optBlock,initGMCA,SNR,0,verPos,J),{'donnees':S_gmca})
                        sio.savemat(folderInitSave + 'MatricesAS/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_A_gmca_exp%s_MC%s_sourceN%s_blocksizeNumber%s_optBlock%s_init%s_SNR%s_optWhite%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,R_p,optBlock,initGMCA,SNR,0,verPos,J),{'donnees':A_gmca})

                        sio.savemat(folderInitSave + 'MatricesAS/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_S_hyb_exp%s_MC%s_sourceN%s_blocksizeNumber%s_optBlock%s_init%s_SNR%s_optWhite%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,R_p,optBlock,initGMCA,SNR,0,verPos,J),{'donnees':S_hyb})
                        sio.savemat(folderInitSave + 'MatricesAS/GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_A_hyb_exp%s_MC%s_sourceN%s_blocksizeNumber%s_optBlock%s_init%s_SNR%s_optWhite%s_verPos%s_J%s'%(sourcesStr,numExp,It_MC,R_t,R_p,optBlock,initGMCA,SNR,0,verPos,J),{'donnees':A_hyb})


    if sauv > 0:# The block sizes and the corresponding C_A values are saved
        sio.savemat(folderInitSave + 'GMCA_cluster_GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_optBlock%s_blocksize_init%s_SNR%s_optWhite%s_verPos%s_J%s_exp%s'%(sourcesStr,optBlock,initGMCA,SNR,0,verPos,J,numExp),{'donnees':blocksize})

        sio.savemat(folderInitSave + 'GMCA_cluster_GMCA_cond%s_%sech_%sit_ptot'%(cdStr,t_samples,nitGMCA) + ptotStr + '_sources%s_optBlock%s_C_hyb_init%s_SNR%s_optWhite%s_verPos%s_J%s_exp%s'%(sourcesStr,optBlock,initGMCA,SNR,0,verPos,J,numExp),{'donnees':C_hyb})


    if pltRes > 0:
        font = {'weight' : 'bold', 'size'   : 20}

        plt.figure(1)
        plt.hold(True)
        plt.plot(blocksize,-10*np.log10(np.nanmedian(C_hyb,axis=2)),linewidth = 4)


        plt.figure(1)
        plt.xlabel('Block sizes',**font)
        plt.ylabel('Mixing matrix criterion in log scale',**font)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.title('Mixing matrix criterion using different block sizes')



        if boolVal == 0:
            print('WARNING, SVD DID NOT CONVERGE FOR SOME EXPERIMENTS')
        else:
            print('Correct termination of the algorithm')



    return C_hyb,S_gmca,A_gmca

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

    num_cor = np.floor(p1*ptot*t)
    ind_cor = randperm(t)
    ind_cor = ind_cor[0:num_cor]
    ind = np.ones((1,t))
    ind[0,ind_cor] = 0
    ind_nocor = np.where(ind[0,:] == 1)[0]


    rest = t - num_cor

    for r in range(0,n):
        p_active = np.floor(t*ptot*(1.-p1))
        temp = np.random.randn(1,rest)
        ind = randperm(rest)
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
        v = (1 - 1./cd)*v/n + 1./cd

    if (cd == 1):
        v = np.ones((n,1))

    Sq = np.zeros((m,n))

    for r in range(0,np.min((m,n))):
        Sq[r,r] = v[r]

    for r in range(0,nmax):
        ue,se,ve = np.linalg.svd(A0)
        A0 = np.dot(np.dot(ue,Sq),ve.T)

        for t in range(0,n):
            A0[:,t] = A0[:,t]/np.sqrt(np.sum(np.power(A0[:,t],2)))
    return A0



# Generating function. Creates data matrix with sources following a Bernouilli Gaussian distribution.
def Make_Experiment_Exact(n_s=2,n_obs=2,t_samp=1024,noise_level=40,dynamic=10,ptot=0.1,cd=1):

    S = np.zeros((n_s,t_samp))

    p_active = np.floor(t_samp*ptot)

    for r in range(0,n_s):
        ind = randperm(t_samp)
        S[r,ind[0:p_active.astype(int)]] = np.random.randn(p_active.astype(int))

    val = np.power(10,(-np.linspace(1,n_s-1,n_s)/(n_s-1)*dynamic))

    A0 = Get_MixmatCond(n_s,n_obs,cd)

    S0= np.dot(np.diag(1./np.sqrt(np.sum(S*S,axis=1))),S)
    S0 = np.dot(np.diag(val),S0)
    X0 = np.dot(A0,S0)

    N = np.random.randn(n_obs,t_samp)
    sigma_noise = np.power(10,(-noise_level/20.))*np.linalg.norm(X0,ord='fro')/np.linalg.norm(N,ord='fro')
    print(sigma_noise)
    N = sigma_noise*N

    X = X0 + N

    return X,X0,A0,S0,N



# Generating function. Generates a data matrix with sources following a generalized gaussian distribution
def Make_Experiment_GeneralizedGaussian(n_s=2,n_obs=2,t_samp=1024,noise_level=40,dynamic=10,ptot=0.1,cd=1,alpha=2):

    S = np.zeros((n_s,t_samp))

    p_active = np.floor(t_samp*ptot)

    for r in range(0,n_s):
        ind = randperm(t_samp)
        S[r,ind[0:p_active]] = generate_2D_generalized_gaussian(1, p_active.astype(int), alpha)

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

# Creates a random permutation
def randperm(n=1):

    X = np.random.randn(n)
    I = X.argsort()

    return I


# Correction of the permutation inderterminacy
def CorrectPerm(cA0,S0,cA,S,optEchAS=0):

    A0 = cp.copy(cA0)
    A = cp.copy(cA)

    nX = np.shape(A0)

    for r in range(0,nX[1]):
        S[r,:] = S[r,:]*(1e-24+lng.norm(A[:,r]))
        A[:,r] = A[:,r]/(1e-24+lng.norm(A[:,r]))
        S0[r,:] = S0[r,:]*(1e-24+lng.norm(A0[:,r]))
        A0[:,r] = A0[:,r]/(1e-24+lng.norm(A0[:,r]))

    try:
        Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))
    except np.linalg.LinAlgError:
        Diff = abs(np.dot(np.linalg.pinv(A0),A))
        print('WARNING, PSEUDO INVERSE TO CORRECT PERMUTATIONS')



    Sq = np.ones(np.shape(S))
    ind = np.linspace(0,nX[1]-1,nX[1])

    for ns in range(0,nX[1]):
        indix = np.where(Diff[ns,:] == max(Diff[ns,:]))[0]
        ind[ns] = indix[0]

    Aq = A[:,ind.astype(int)]
    Sq = S[ind.astype(int),:]

    for ns in range(0,nX[1]):
        p = np.sum(Sq[ns,:]*S0[ns,:])
        if p < 0:
            Sq[ns,:] = -Sq[ns,:]
            Aq[:,ns] = -Aq[:,ns]

    if optEchAS==1:
        return Aq,Sq,A0,S0
    else:
        return Aq,Sq


# CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)
def EvalCriterion(A0,S0,A,S,optMedian=1):

    gA,gS = CorrectPerm(A0,S0,A,S)

    try:
        Diff = abs(np.dot(np.linalg.inv(np.dot(A0.T,A0)),np.dot(A0.T,gA)))
    except np.linalg.LinAlgError:
        Diff = abs(np.dot(np.linalg.pinv(A0),gA))
        print('ATTENTION, PSEUDO-INVERSE POUR LE CRITERE SUR A')


    z = np.shape(gA)

    if optMedian==1:
        p = np.median(Diff - np.diag(np.diag(Diff)))
    else:
        p = (np.sum(Diff - np.diag(np.diag(Diff))))/(z[1]*(z[1]-1))


    return p

# Generates a mixing matrix simulating elution types
def generateAGauss(m,n,standDev,mBarre,nUnderMBarre):
    A = np.zeros((m,n))

    indP1 = np.linspace(0,mBarre,num=nUnderMBarre,dtype=int);
    print('mBarre: %s'%mBarre)
    for ii in range(nUnderMBarre):
        A[indP1[ii],ii] = 1
        A[:,ii] = np.convolve(A[:,ii],sng.gaussian(len(A[:,ii]), standDev),mode='same')

    indP2Temp = np.linspace(mBarre+1,m-1,num=n-nUnderMBarre+1,dtype=int);
    step = indP2Temp[1] - indP2Temp[0]
    indP2 = np.linspace(mBarre+step,m-1,num=n-nUnderMBarre,dtype=int);
    #    indP2 = np.linspace(mBarre+1,m-1,num=n-nUnderMBarre,dtype=int);
    for ii in range(nUnderMBarre,n):
        A[indP2[ii-nUnderMBarre],ii] = 1
        A[:,ii] = np.convolve(A[:,ii],sng.gaussian(len(A[:,ii]), standDev),mode='same')

    return A

# Creates data simulating a 1H NMR experiment. The mixing matrix simulates elution types and the sources are realistic sources convolved with a Laplacian to simulate a given spatial resolution.
def Make_Experiment_HMR(n_s=2,n_obs=2,t_samp=1024,noise_level=40,standDev = 3,peak_width=0.05,optNorm=1):
    notIn = [34,39,40,41,42]

    nUnderMBarre = np.ceil(n_s/2)
    nUnderMBarre = nUnderMBarre.astype(int)
    mBarre = np.floor(0.75*n_obs)

    peakList = snp.returnPeakList()
    S0 = np.zeros((len(peakList),t_samp))

    compt = 0
    for key in peakList:
        (spectrum, ppm) = snp.get_nmr_spectrum(peakList[key],[0,10],num_samples=t_samp,peak_width=peak_width)
        if optNorm == 1:
            S0[compt,:] = spectrum/np.sum(spectrum)
        else:
            S0[compt,:] = spectrum

        compt +=1
    S0 = np.delete(S0,notIn,0)
    ind = range(0,n_s)
    S0 = S0[ind,:]

    A0 = generateAGauss(n_obs,n_s,standDev,mBarre,nUnderMBarre)

    X0 = np.dot(A0,S0)

    N = np.random.randn(n_obs,t_samp)
    sigma_noise = np.power(10,(-noise_level/20.))*np.linalg.norm(X0,ord='fro')/np.linalg.norm(N,ord='fro')
    N = sigma_noise*N

    X = X0 + N

    return X,X0,A0,S0,N
