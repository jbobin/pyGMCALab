# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# from pyngmca import proximal
from pyGMCA import bss
#from pygmca.pyredwave import RedWave


# %% create sparse data or synthetic nmr data
nmr = False
data_settings = {}
np.random.seed(33) #32
if nmr:
    data_settings['rows'] = 24
    data_settings['rank'] = 12
    data_settings['dB'] = 40
    reference = bss.tools.create_realistic_nmr_mixtures(data_settings)
else:
    data_settings['rows'] = 100
    data_settings['rank'] = 10
    data_settings['columns'] = 200
    data_settings['dB'] = 10
    data_settings['bernoulli_S'] = 0.08
    data_settings['alpha_S'] = 1
    data_settings['verbose'] = 1
    reference = bss.tools.create_sparse_data(data_settings)
#plt.imshow(reference['data'], interpolation='nearest')

# create the noisy data
Y = reference['data'] + reference['noise']
# evaluation function (will be called at each iteration of the algorithm)
criteria_rec = lambda data: bss.tools.evaluation(data, reference)[0]
# sparsity parameter recording function (""")
lambda_rec = lambda data: np.max(data['lambda'])


#%% first method for using nGMCA
alg = bss.ngmca.Ngmca()


parameters = {'data': Y,
              'rank': data_settings['rank'],
              'verbose': 1,
              'maximum_iteration': 300,
              'S_parameters': {'tau_mad': 1},
              'recording_functions': {'lambda': lambda_rec,
                                      'criteria': criteria_rec}}
# display during algorithm
parameters['display_function'] = \
    lambda data: plt.plot(data['factorization'].S.T)
parameters['display_time'] = 1
    
# set a particular initialization (for repeatability)
np.random.seed(58)
# launch the algorithm
result = alg.run(parameters)
# evaluate the result
crit = bss.tools.evaluation(result, reference, True)[0]



#%% second method for using nGMCA
# updaters are classes which are called for the updates of
# A and S. One can choose between different types of updaters
alg = bss.ngmca.Framework()
lambda_rec = lambda data: np.max(data['S_updater'].lambdas)
parameters_upd = {'data': Y,
                  'rank': data_settings['rank'],
                  'verbose': 1,
                  'maximum_iteration': 300,
                  'S_updater': bss.ngmca.SparseUpdater(tau_mad=1),
                  'A_updater': bss.ngmca.SparseUpdater(tau_mad=0),
                  'recording_functions': {'lambda': lambda_rec,
                                          'criteria': criteria_rec}}
parameters_upd['display_function'] =\
    lambda data: plt.plot(data['factorization'].S.T)
parameters_upd['display_time'] = 1
 
np.random.seed(58)
result_upd = alg.run(parameters_upd)
crit = bss.tools.evaluation(result_upd, reference, True)[0]


#%% visualization of the evoluation of the SDR_S
plt.clf()
plt.plot(result['recording']['SDR_S'])
plt.plot(result_upd['recording']['SDR_S'])

#%% visualization of the evoluation of lambda
plt.clf()
plt.plot(result['recording']['lambda'])
plt.plot(result_upd['recording']['lambda'])

#%% decompostion
plt.clf()
(criteria, decomp) = bss.tools.evaluation(result, reference, 1)
plt.plot(decomp['interferences'])