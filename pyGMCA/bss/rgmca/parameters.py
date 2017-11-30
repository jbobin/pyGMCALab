# -*- coding: utf-8 -*-
"""
Parameters for the data generation and the algorithms.
Dictionaries:
    data_setting: data generation
    algo_setting: parameters for the algorithms
    param_setting: other parameters, such as verbose
    

"""
#Import'
import numpy as np
#################################################################################################################################################################################################################################


#Data generation parameters
data_setting={}

#Dimensions of the problem
data_setting['m'] =16 #number of observations
data_setting['n'] =8 #number of sources
data_setting['t'] =4096 #number of samples

#sources
data_setting['alphaS'] =2 #parameter of the Generalized Gaussian law for the amplitude
data_setting['pS'] =0.05 # activation parameter for the Bernoulli law (for the support)
data_setting['ampliS'] = 100 # standard deviation (amplitude)

#outliers
data_setting['nbCol']=0.1*data_setting['t'] #Number of corrupted columns
data_setting['ampliO'] =100. # standard deviation (amplitude)
data_setting['alphaO'] =2 #parameter of the Generalized Gaussian law for the amplitude

#Gaussian noise
data_setting['ampliN'] =0.1 #standard deviation (amplitude)

dS=data_setting

#################################################################################################################################################################################################################################

#Algorithm parameters
algo_setting={};

algo_setting['iteMaxGMCA']=300 #Number of iterations for GMCA
algo_setting['iteMaxrAMCA']=100 #Number of outer loops for rAMCA
algo_setting['iteMaxrGMCA']=25 # Number of outer loops for rGMCA
algo_setting['iteMaxAMCA']=1000 #Number of iterations for AMCA
algo_setting['iteMaxBeta']=700 #Number of iterations for the beta-divergence
algo_setting['iteMaxPCP']=5000 #Number of iterations for PCP


algo_setting['kSMax']=3.# Final threshold for the sources with 3*mad
algo_setting['kOMax']=3 # Final threshold for the outliers with 3*mad

aS=algo_setting

#################################################################################################################################################################################################################################


#Other parameters
param_setting={}

#Verbose#
param_setting['verbose']=1 # print will be displayed if set to one
param_setting['verboseAMCA']=1 * param_setting['verbose'] #Specific for AMCA
param_setting['verboserAMCA']=1 * param_setting['verbose'] #Specific for rAMCA
param_setting['verboseGMCA']=1 * param_setting['verbose'] #Specific for GMCA
param_setting['verboserGMCA']=1 * param_setting['verbose'] #Specific for rGMCA
param_setting['verbosePCP']=1* param_setting['verbose'] #Specific for PCP
param_setting['verboseBeta']=1 * param_setting['verbose'] #Specific for Beta-div.


#ForMonte Carlo simulations and benchmark:
param_setting['nbreParam']=9 #number of varying parameters
param_setting['param']=np.array([1,5,10,15,20,25,30,35,40]) #Values of the varying parameters

#Number of iterations for each parameters
param_setting['nbreIter']=1

#Save the results
param_setting['saveFile']=0 #Set to 1 to save the results
param_setting['nameFile']="varyingNbreO" #Name of the file to be save

#Display the results
param_setting['plot']=1 #set to 1 to display the results

pS=param_setting