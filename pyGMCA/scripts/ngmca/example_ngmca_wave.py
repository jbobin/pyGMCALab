# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from pyGMCA.bss.ngmca import base as bss
from redwave_toolbox.pyredwave import RedWave # Should be fixed


# %% create the data
np.random.seed(33) #32
data_settings= {'rows': 4,
                'rank': 2,
                'dB': 20,
                'width': 8}
reference = bss.tools.create_realistic_nmr_mixtures(data_settings)
Y = reference['data'] + reference['noise']
lambda_rec = lambda data: np.max(data['S_updater'].lambdas)
criteria_rec = lambda data: bss.tools.evaluation(data, reference)[0]
plt.plot(Y.T)





# %% NGMCA WITH WAVELET SPARSE UPDATES
# %% analysis and synthesis nGMCA
# in this version, the update of S is replaced so as to use wavelet sparsity.
# The updater requires the redwave instance. Both analysis and synthesis formulations
# can be used
wave = RedWave(Y, 1)
wY = wave.forward(Y)
plt.plot(wY.T)


alg = bss.ngmca.Framework()
parameters = {'data': Y,
              'rank': data_settings['rank'],
              'verbose': 1,
              'maximum_iteration': 300,
              'S_updater': bss.ngmca.updaters.RedWaveUpdater(tau_mad=2,
                  redwave_operator=wave, formulation="synthesis"),
              'A_updater': bss.ngmca.SparseUpdater(tau_mad=0),
              'recording_functions': {'lambda': lambda_rec,
                                      'criteria': criteria_rec}}
parameters['display_function'] = lambda data: plt.plot(data['factorization'].S.T)
parameters['display_time'] = 1

np.random.seed(58)
result_ana = alg.run(parameters)
crit = bss.tools.evaluation(result_ana, reference, True)[0]


# %% standard version of nGMCA for comparison (see example ngmca)
alg = bss.ngmca.Framework()
parameters = {'data': Y,
              'rank': data_settings['rank'],
              'verbose': 1,
              'maximum_iteration': 300,
              'S_updater': bss.ngmca.SparseUpdater(tau_mad=2),
              'A_updater': bss.ngmca.SparseUpdater(tau_mad=0),
              'recording_functions': {'lambda': lambda_rec,
                                      'criteria': criteria_rec}}
parameters['display_function'] = lambda data: plt.plot(data['factorization'].S.T)
parameters['display_time'] = 1

np.random.seed(58)
result = alg.run(parameters)
crit = bss.tools.evaluation(result, reference, True)[0]


#%% visualization of the evolution of the SDR_S
plt.clf()
plt.plot(result_ana['recording']['SDR_S'])
plt.plot(result['recording']['SDR_S'])



# %% FOR STAND ALONE USE OF THE UPDATERS
# %% analysis/synthesis updater
wave = RedWave(Y, 1)
wY = wave.forward(Y)
plt.plot(wY.T)
A = reference['factorization'].A
S = reference['factorization'].S
updater = bss.ngmca.RedWaveUpdater(tau_mad=2,
    redwave_operator=wave, formulation="synthesis")
lambdas = np.ones((S.shape[0], wY.shape[1])) * np.max(S) * 1
S_syn = updater.process(Y, A, S, lambdas)
plt.clf()
plt.plot(S_syn.T)



# %% DEALING WITH 2D SOURCES
# make image mixtures
reference = bss.tools.create_image_mixtures(rows=16, dB=30)
im_shape = (int(np.sqrt(reference["factorization"].S.shape[1])),) * 2
bss.tools.show_images(reference["factorization"].S, shape=im_shape)

# make data
Y = reference['data'] + reference['noise']
bss.tools.show_images(Y, shape=im_shape)

# useful functions
lambda_rec = lambda data: np.max(data['S_updater'].lambdas)
criteria_rec = lambda data: bss.tools.evaluation(data, reference)[0]

# wavelets
wave = RedWave(Y.reshape([Y.shape[0]] + list(im_shape)), (2, 1))
wS = wave.forward(reference["factorization"].S.reshape([4] + list(im_shape)))
bss.tools.show_images(wS)  # show reference images in the wavelet domain

# transformation between lines and standard shapes
# line to shape
l2s = lambda data: data.reshape([data.shape[0]] + list(im_shape))
# shape to line
s2l = lambda data: data.reshape([data.shape[0], np.prod(data.shape[1:])])

# function which modifies S and the data before the update of A:
# in order to go to the wavelet domain and remove the coarse scale
# (this helps the separation)
modifier = lambda S: s2l(wave.remove_coarse_scales(wave.forward(l2s(S)),
                                                   inplace=True))

alg = bss.ngmca.Framework()
parameters = {'data': Y,
              'rank': 4,
              'verbose': 1,
              'maximum_iteration': 120,
              'S_updater': bss.ngmca.updaters.RedWaveUpdater(tau_mad=2,
                  redwave_operator=wave, formulation="analysis",
                  direct_sparsity=False, reweighted_l1=3),
              'A_updater': bss.ngmca.SparseUpdater(tau_mad=0,
                                                  modifier=modifier),
              'recording_functions': {'lambda': lambda_rec,
                                      'criteria': criteria_rec}}
parameters['display_function'] =\
    lambda data: bss.tools.show_images(data['factorization'].S, 0, im_shape)
parameters['display_time'] = 1

np.random.seed(58)
result_ana_2D = alg.run(parameters)
crit = bss.tools.evaluation(result_ana_2D, reference, True)[0]
# to make this faster: improve the _update_lambda function so as to
# compute the standard deviation on a subset of the data at each
# iteration.
