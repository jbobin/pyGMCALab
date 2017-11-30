# -*- coding: utf-8 -*-

# so as to build the pyredwave toolbox,
# you need to execute "python setup.py build"
# in the pyredwave folder.
# The compilation requires Boost.Python
# (tested on Mac and Ubuntu, with Python 2.7).
from redwave import RedWave, mad_std
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import time
from PyQt4.QtGui import QApplication
from gc import collect


# process several slices in parallel (each line of x)
x = np.array([[1.0, 2.0, 3, 4], [3, 4, 3, 4]])
print("Input matrix:")
print(x)
# parametrize wavelets
wave_filter = "Haar"
number_of_scales = 2
isometric = True
wave_dimensions = 1
wave = RedWave(x, wave_dimensions, wave_filter,number_of_scales, isometric)
wx = wave.forward(x)
print("Output matrix (wavelet performed on dimension 1, on 2 slices):")
print(wx)
x2 = wave.backward(wx)
# in place transformation
wave.transform(x, wx, 1)
# create a wavelet object for inputs of size Lena
print("Wavelet can be performed on any 1 or 2 dimensions.")
    # load an image
lena = misc.lena()
# convert it to double
lena = lena.astype("float64")
wave_dimensions = (1, 0)
wave_filter = "Daubechies-4"
wave = RedWave(lena, wave_dimensions, wave_filter,
               number_of_scales, isometric)
# convert Lena into wavelets
wlena = wave.forward(lena)
# show Lena in the wavelet domain
plt.imshow(wlena, cmap=plt.cm.gray)
plt.get_current_fig_manager().window.raise_()
QApplication.processEvents()
# Only the first coarse scale is actually used for backward computation
print("Only the first coarse scale is actually used for backward computation.")
wlena = wave.remove_coarse_scales(wlena, inplace=True,
                                  keep_actual_coarse_scale=True)
time.sleep(3)
plt.clf()
collect()
plt.imshow(wlena, cmap=plt.cm.gray)
QApplication.processEvents()
time.sleep(1)
print("In isometric mode, norms are kept (after removing redundant coarse scales):")
print("- norm of Lena: %s." % np.linalg.norm(lena))
print("- norm of Lena wavelet coefficients: %s." % np.linalg.norm(wlena))
time.sleep(2)
print("One can also easily get estimate the noise on each scale using ")
print("extract_scale_vals with the mad estimator (or any other).")
scale_mad = wave.extract_scale_vals(wlena, mad_std)
print("Mad estimate on each scale:")
print(scale_mad)
print("From there, one can make a full matrix using make_full_vals.")
# wavelet thresholding
lambdas = 50 * np.ones(wlena.shape)
# do not penalize coarse scales
lambdas = wave.remove_coarse_scales(lambdas, inplace=True,
                                    keep_actual_coarse_scale=False)
smooth_lena = wave.sparse_proximal(lena, lambdas, number_of_iterations=6)
print("This toolbox can also compute the wavelet sparse analysis proximal")
print("operator using the Generalized Forward-Backward algorithm (displayed")
print("results), and the sparse analysis inversion using Chambolle-Pock\nalgorithm.")
plt.clf()
collect()
plt.imshow(smooth_lena, cmap=plt.cm.gray)
QApplication.processEvents()
time.sleep(1)  

