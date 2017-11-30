# -*- coding: utf-8 -*-
"""
File building the dictionaries, containing the parameters
"""
import numpy as np

"""Data parameters"""
data_setting={}
'''dimensions'''
data_setting['m'] =6 #Number of observations
data_setting['n'] =6 #Number of sources
data_setting['t'] =4096 # Number of Samples
'''Sources'''
data_setting['pS'] =0.05 #Activation parameter of the Bernoulli-Gaussian law
data_setting['gS'] = 100.  #Standard deviation of the Gaussian law

'''Gaussian Noise'''
data_setting['gN'] =50. # SNR in dB

'''Outliers'''
data_setting['nbCol']=0.1*data_setting['t'] #Number of corrupted columns
data_setting['gO']=-10. #Signal to Outliers ratio in dB

dS=data_setting




"""Algorithms Parameters"""
algo_setting={};

algo_setting['iteMaxXMCA']=300 #Number of iterations for GMCA, AMCA or MCA
algo_setting['iteMax']=10000 #(Maximal) Number of Iterations for iterative processes
algo_setting['Rew']=4 #Number of reweighting loops
algo_setting['kSMax']=3. # Thresholding of the sources with kSMax-mad
algo_setting['kOMax']=3. #Thresholding of the outliers with kOMax-mad

aS=algo_setting


"""Simulations Parameters"""
param_setting={}
param_setting['nbreParam']= 3# Number of parameters
param_setting['param']=np.array([[6,20,60]])  # Varying parameters
param_setting['nbreIter']=1 #Number of runs
param_setting['nameFile']="mVar" #Prefix of the name of the file to be saved

pS=param_setting