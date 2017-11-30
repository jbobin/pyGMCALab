# -\*- coding: utf-8 -\*-

r"""
redwave.py - This file is part of pygmca.
The pygmca package aims at performing non-negative matrix factorization.
This module provides 1D and 2D redundant wavelets.
Copyright 2014 CEA
Contributor : Jérémy Rapin (jeremy.rapin.math@gmail.com)
Created on December 13, 2014, last modified on December 14, 2014

This software is governed by the CeCILL  license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

__version__ = "1.0"
__author__ = "Jeremy Rapin"
__url__ = "http://www.cosmostat.org/GMCALab.html"
__copyright__ = "(c) 2014 CEA"
__license__ = "CeCill"

# -*- coding: utf-8 -*-

import numpy as np
try:
    from redwavecxx import RedWave as RW
except ImportError:
    pass


class RedWave(object):
    r"""
    This class provides the tools to use redundant wavelets in 1D and 2D.
    (see RedWave.__init__ for construction parameters).
    """

    def __init__(self, input_array, wave_dimensions, filtername="Daubechies-4",
                 number_of_scales=3, isometric=False):
        r"""
        Initializatin of a wavelet instance for a given type of wavelet transform.
        
        Inputs
        ------
        - input_array: numpy array
            Array on which to apply the wavelets. This initializes the size of the
            dimensions on which the wavelets will be used. The instance will only be
            able to be applied on array with identic sizes on these dimensions
            (wave_dimensions) but the other sizes can be changed.
        - wave_dimensions: list
            List of the dimensions on which to apply the wavelets (up to 2 different
            dimensions).
        - filtername (default: "Daubechies-4"): str
            Name of the filter to be used (see make_wavelet_filter for a list).
        - number_of_scales (default: 3): int
            Number of fine scales of the wavelet transform.
        - isometric (default: False): bool
            Use an isometric transform if yes (otherwise, the normalization tends
            to make Gaussian noise uniform on all scales).
        """
        # save all characteristics
        self._number_of_scales = number_of_scales
        self._wave_dimensions = wave_dimensions
        if isinstance(self._wave_dimensions, int):
            self._wave_dimensions = [wave_dimensions]
        elif isinstance(self._wave_dimensions, tuple):
            self._wave_dimensions = list(wave_dimensions)
        self._input_shape = list(input_array.shape)
        # make filter
        self._filter = make_wavelet_filter(filtername)
        # save shape
        for k in range(0, len(self._input_shape)):
            self._input_shape[k] =\
                self._input_shape[k] if (k in self._wave_dimensions) else 1
        # create the operator
        try:
            self._redwave = RW(input_array, wave_dimensions, self._filter,
                               number_of_scales, isometric)
        except NameError:
            raise Exception('pyredwave package not properly compiled.')
    
    def get_wavelet_dimensions(self):
        r"Returns the dimensions on which the wavelets are performed."
        return self._wave_dimensions
    
    def get_input_shape(self):
        r"""
        Returns the dimensions of the minimum size array on which the 
        wavelets can be performed. All inputs must have similar non-singleton
        dimensions, on which the wavelets are performed. Singleton dimensions
        may however take different values (operations will be applied on each
        slice).
        """
        return self._input_shape
    
    def get_filter(self):
        r"Returns the currently used wavelet filter."
        return self._filter.copy()
    
    def forward(self, signal_array):
        r"Returns the forward wavelet transform of the input array."
        return self._redwave.forward(signal_array)
    
    def backward(self, signal_array):
        r"Returns the backward wavelet transform of the input array."
        return self._redwave.backward(signal_array)

    def transform(self, signal_array, wavelet_array,
                  direction=1):
        r"""
        Inplace wavelet transform: performs forward or backward wavelet
        transform depending on the arguments and modifies the corresponding
        input.
        
        Inputs
        ------
        - signal_array: numpy array
            Signal array expresed in the direct domain.
        - wavelet_array: numpy array
            Array of wavelet coefficients.
        - direction (default: 1): float
            If >0, performs the forward transform and modifies
            wavelet_array accordingly, if <=0, performs the backward
            transform and modifies signal_array accordingly.
        """
        self._redwave.transform(signal_array, wavelet_array,
                                direction)
        
    def sparse_proximal(self, signal_array, mu, number_of_iterations=24):
        r"""
        Computes the proximal operator of mu * ||W x||_1,
        where W is the wavelet transform.
        
        Inputs
        ------
        - signal_array: numpy array
            Signal array expresed in the direct domain.
        - mu: float or numpy array
            Value of the sparsity parameter mu
        - number_of_iterations (default: 24): int
            Number of iterations of the algorithm.
        """
        return self._redwave.sparse_proximal(signal_array, mu,
                                             number_of_iterations)
    
    def sparse_inversion(self, x0, AtA, AtY, mu, number_of_iterations=24,
                         non_negative=False):
        r"""
        Computes the result of:
        argmin_x ||y - A x||_2^2 +  mu * ||W x||_1,
        potentially with a non-negative constraint on x.
        
        Inputs
        ------
        - x0: numpy array
            Initialization of the algorithm.
        - AtA: numpy array
            Product between transpose of A and A.
        - AtY: numpy array
            Product between transpose of A and Y.
        - mu: float or numpy array
            Value of the sparsity parameter mu
        - number_of_iterations (default: 24): int
            Number of iterations of the algorithm.
        - non_negative (default: False): bool
            If True, uses a non-negative constraint.
        
        Note
        ----
        Solves the following problem:
        
        
        .. math::
            \text{argmin}_{x} \frac{1}{2}\|y - \mathbf{A} x\|_2^2 + \|\mathbf{\mu} \odot (\mathbf{W} x)||_1
        """
        return self._redwave.sparse_inversion(x0, AtA, AtY, mu,
                                              number_of_iterations,
                                              non_negative)
    
    def extract_scale_vals(self, wavelet_array, function):
        r"""
        Computes a scalar per scale on the wavelet coefficients, using
        the provided function.
        
        Inputs
        ------
        - wavelet_array: numpy array
            Array of wavelet coefficients.
        - function: function
            Function returning a scalar, which will be applied independently on
            each scale.
        """
        scale_vals_shape = self._get_scale_vals_shape(wavelet_array)
        scale_vals = np.zeros(scale_vals_shape)
        # prepare offsets
        (ranges, offsets, make_slice_text) =\
            self._prepare_slice_text(scale_vals.shape)
        # iterate
        for k in range(0, scale_vals.size):
            (ind_script, inds) = make_slice_text(k)
            exec("scale_vals[inds] = function(wavelet_array%s)" %\
                ind_script)
        return scale_vals
    
    def _get_scale_vals_shape(self, wavelet_array):
        r"Returns the shape of the scale_vals variable (one value per scale)."
        scale_vals_shape = list(wavelet_array.shape)
        if len(self._wave_dimensions) == 1 :
            scale_vals_shape[self._wave_dimensions[0]] =\
                self._number_of_scales + 1
        else:
            scale_vals_shape[self._wave_dimensions[0]] =\
                2 * self._number_of_scales
            for k in self._wave_dimensions[1:]:
                scale_vals_shape[k] = 2
        return scale_vals_shape
    
    def make_full_vals(self, scale_vals):
        r"""
        Returns a wavelet size array which is constant on each
        scale, with values provided by the scale_vals variable.
        
        Inputs
        ------
        - scale_vals: numpy array
            Array of scalars associated to each of the wavelet scales.
        """
        full_vals_shape = list(scale_vals.shape)
        for k in self._wave_dimensions:
            full_vals_shape[k] *= self._input_shape[k]
        full_vals = np.ones(full_vals_shape)
        # prepare offsets
        (ranges, offsets, make_slice_text) =\
            self._prepare_slice_text(scale_vals.shape)
        for k in range(0, scale_vals.size):
            (ind_script, inds) = make_slice_text(k)
            make_script = "full_vals%s = scale_vals[inds]" % ind_script
            exec(make_script)
        return full_vals
    
    def _prepare_slice_text(self, shape):
        r"Used to prepare the extraction of a particular wavelet scale."
        ranges = [np.array([0])] * len(shape)
        offsets = [1] * len(shape)
        for k in self._wave_dimensions:
            ranges[k] = np.array(range(0, self._input_shape[k]))
            offsets[k] = self._input_shape[k]
        
        def make_slice_text(k):
            r"Returns a string used to extract a particular wavelet scale."
            inds = np.unravel_index(k, shape)
            extract_script = "[np.ix_("
            for i in range(0, len(shape)):
                extract_script +=\
                    "ranges[%s] + offsets[%s] * inds[%s]" % (i, i, i)
                extract_script +=\
                    ", " if (i < len(shape) - 1) else ")]"
            return (extract_script, inds)
        return (ranges, offsets, make_slice_text)
    
    def remove_coarse_scales(self, wavelet_array,
                             keep_actual_coarse_scale=False, inplace=False):
        r"""
        Sets to 0 the coarse scale of a wavelet coefficient array.
        In 2D, only the top left coarse scale is necessary for reconstruction,
        the others are artefacts from the coefficients computation.
        
        Inputs
        ------
        - wavelet_array: numpy array
            Array of wavelet coefficients.
        - keep_actual_coarse_scale (default: False): bool
            Whether to keep the top left coarse scale which is necessary for
            reconstruction.
        - inplace (default: False): bool
            Whether to modify or not the input (saves memory if True).
        """
        wave_array = wavelet_array
        if not inplace:
            wave_array = wavelet_array.copy()
        # prepare offsets
        scale_vals_shape = self._get_scale_vals_shape(wavelet_array)
        (ranges, offsets, make_slice_text) =\
            self._prepare_slice_text(scale_vals_shape)
        # modify ranges
        for k in range(0, len(wave_array.shape)):
            if k not in self._wave_dimensions:
                ranges[k] = np.array(range(0, wave_array.shape[k]))
        # loop
        beg = 1 if keep_actual_coarse_scale else 0
        last = 1 if len(self._wave_dimensions) == 1 else self._number_of_scales
        for k in range(beg, last):
            inds = [0] * len(wavelet_array.shape)
            inds[self._wave_dimensions[0]] = k * 2
            ind_script = make_slice_text(0)[0]
            make_script = "wave_array%s = 0.0" % ind_script
            exec(make_script)
        return wave_array


def mad_std(input_array):
    r"""
    Returns an estimation of the standard deviation of an array
    using the MAD estimator (works well on sparse arrays).
    """
    med = np.median(input_array)
    return 1.4826 * np.median(np.abs(input_array - med))


def make_wavelet_filter(name):
    r"""
    Returns a wavelet filter from its name, among:
    Haar, Beylkin, Daubechies-(4,6,8,10,12,14,16,18,20),
    Coiflet-(1,2,3,4,5), and Symmlet-(4,5,6,7,8,9,10)
    where only one number must be provided.
    Example: make_wavelet_filter("Daubechies-4").
    This function was adapted from the MakeONFilter 
    function from the WaveLab toolbox.
    """
    if name == 'Haar':
        filt = np.array([1, 1]) / np.sqrt(2.0)
    elif name == 'Beylkin':
        filt = np.array([0.099305765374, 0.424215360813, 0.699825214057,
                         0.449718251149, -0.110927598348, -0.264497231446,
                         0.026900308804, 0.155538731877, -0.017520746267,
                         -0.088543630623, 0.019679866044, 0.042916387274,
                         -0.017460408696, -0.014365807969, 0.010040411845,
                         0.001484234782, -0.002736031626, 0.000640485329])
    elif name == 'Daubechies-4':
        filt = np.array([0.482962913145, 0.836516303738, 0.224143868042,
                         -0.129409522551])
    elif name == 'Daubechies-6':
        filt = np.array([0.332670552950, 0.806891509311, 0.459877502118,
                         -0.135011020010, -0.085441273882, 0.035226291882])
    elif name == 'Daubechies-8':
        filt = np.array([0.230377813309, 0.714846570553, 0.630880767930,
                         -0.027983769417, -0.187034811719, 0.030841381836,
                         0.032883011667, -0.010597401785])
    elif name == 'Daubechies-10':
        filt = np.array([0.160102397974, 0.603829269797, 0.724308528438,
                         0.138428145901, -0.242294887066, -0.032244869585,
                         0.077571493840, -0.006241490213, -0.012580751999,
                         0.003335725285])
    elif name == 'Daubechies-12':
        filt = np.array([0.111540743350, 0.494623890398, 0.751133908021,
                         0.315250351709, -0.226264693965, -0.129766867567,
                         0.097501605587, 0.027522865530, -0.031582039317,
                         0.000553842201, 0.004777257511, -0.001077301085])
    elif name == 'Daubechies-14':
        filt = np.array([0.077852054085, 0.396539319482, 0.729132090846,
                         0.469782287405, -0.143906003929, -0.224036184994,
                         0.071309219267, 0.080612609151, -0.038029936935,
                         -0.016574541631, 0.012550998556, 0.000429577973,
                         -0.001801640704, 0.000353713800])
    elif name == 'Daubechies-16':
        filt = np.array([0.054415842243, 0.312871590914, 0.675630736297,
                         0.585354683654, -0.015829105256, -0.284015542962,
                         0.000472484574, 0.128747426620, -0.017369301002,
                         -0.044088253931, 0.013981027917, 0.008746094047,
                         -0.004870352993, -0.000391740373, 0.000675449406,
                         -0.000117476784])
    elif name == 'Daubechies-18':
        filt = np.array([0.038077947364, 0.243834674613, 0.604823123690,
                         0.657288078051, 0.133197385825, -0.293273783279,
                         -0.096840783223, 0.148540749338, 0.030725681479,
                         -0.067632829061, 0.000250947115, 0.022361662124,
                         -0.004723204758, -0.004281503682, 0.001847646883,
                         0.000230385764, -0.000251963189, 0.000039347320])
    elif name == 'Daubechies-20':
        filt = np.array([0.026670057901, 0.188176800078, 0.527201188932,
                         0.688459039454, 0.281172343661, -0.249846424327,
                         -0.195946274377, 0.127369340336, 0.093057364604,
                         -0.071394147166, -0.029457536822, 0.033212674059,
                         0.003606553567, -0.010733175483, 0.001395351747,
                         0.001992405295, -0.000685856695, -0.000116466855,
                         0.000093588670, -0.000013264203])
    elif name == 'Coiflet-1':
        filt = np.array([0.038580777748, -0.126969125396, -0.077161555496,
                         0.607491641386, 0.745687558934, 0.226584265197])
    elif name == 'Coiflet-2':
        filt = np.array([0.016387336463, -0.041464936782, -0.067372554722,
                         0.386110066823, 0.812723635450, 0.417005184424,
                         -0.076488599078, -0.059434418646, 0.023680171947,
                         0.005611434819, -0.001823208871, -0.000720549445])
    elif name == 'Coiflet-3':
        filt = np.array([-0.003793512864, 0.007782596426, 0.023452696142,
                         -0.065771911281, -0.061123390003, 0.405176902410,
                         0.793777222626, 0.428483476378, -0.071799821619,
                         -0.082301927106, 0.034555027573, 0.015880544864,
                         -0.009007976137, -0.002574517688, 0.001117518771,
                         0.000466216960, -0.000070983303, -0.000034599773])
    elif name == 'Coiflet-4':
        filt = np.array([0.000892313668, -0.001629492013, -0.007346166328,
                         0.016068943964, 0.026682300156, -0.081266699680,
                         -0.056077313316, 0.415308407030, 0.782238930920,
                         0.434386056491, -0.066627474263, -0.096220442034,
                         0.039334427123, 0.025082261845, -0.015211731527,
                         -0.005658286686, 0.003751436157, 0.001266561929,
                         -0.000589020757, -0.000259974552, 0.000062339034,
                         0.000031229876, -0.000003259680, -0.000001784985])
    elif name == 'Coiflet-5':
        filt = np.array([-0.000212080863, 0.000358589677, 0.002178236305,
                         -0.004159358782, -0.010131117538, 0.023408156762,
                         0.028168029062, -0.091920010549, -0.052043163216,
                         0.421566206729, 0.774289603740, 0.437991626228,
                         -0.062035963906, -0.105574208706, 0.041289208741,
                         0.032683574283, -0.019761779012, -0.009164231153,
                         0.006764185419, 0.002433373209, -0.001662863769,
                         -0.000638131296, 0.000302259520, 0.000140541149,
                         -0.000041340484, -0.000021315014, 0.000003734597,
                         0.000002063806, -0.000000167408, -0.000000095158])
    elif name == 'Symmlet-4':
        filt = np.array([-0.107148901418, -0.041910965125, 0.703739068656,
                         1.136658243408, 0.421234534204, -0.140317624179,
                         -0.017824701442, 0.045570345896])
    elif name == 'Symmlet-5':
        filt = np.array([0.038654795955, 0.041746864422, -0.055344186117,
                         0.281990696854, 1.023052966894, 0.896581648380,
                         0.023478923136, -0.247951362613, -0.029842499869,
                         0.027632152958])
    elif name == 'Symmlet-6':
        filt = np.array([0.021784700327, 0.004936612372, -0.166863215412,
                         -0.068323121587, 0.694457972958, 1.113892783926,
                         0.477904371333, -0.102724969862, -0.029783751299,
                         0.063250562660, 0.002499922093, -0.011031867509])
    elif name == 'Symmlet-7':
        filt = np.array([0.003792658534, -0.001481225915, -0.017870431651,
                         0.043155452582, 0.096014767936, -0.070078291222,
                         0.024665659489, 0.758162601964, 1.085782709814,
                         0.408183939725, -0.198056706807, -0.152463871896,
                         0.005671342686, 0.014521394762])
    elif name == 'Symmlet-8':
        filt = np.array([0.002672793393, -0.000428394300, -0.021145686528,
                         0.005386388754, 0.069490465911, -0.038493521263,
                         -0.073462508761, 0.515398670374, 1.099106630537,
                         0.680745347190, -0.086653615406, -0.202648655286,
                         0.010758611751, 0.044823623042, -0.000766690896,
                         -0.004783458512])
    elif name == 'Symmlet-9':
        filt = np.array([0.001512487309, -0.000669141509, -0.014515578553,
                         0.012528896242, 0.087791251554, -0.025786445930,
                         -0.270893783503, 0.049882830959, 0.873048407349,
                         1.015259790832, 0.337658923602, -0.077172161097,
                         0.000825140929, 0.042744433602, -0.016303351226,
                         -0.018769396836, 0.000876502539, 0.001981193736])
    elif name == 'Symmlet-10':
        filt = np.array([0.001089170447, 0.000135245020, -0.012220642630,
                         -0.002072363923, 0.064950924579, 0.016418869426,
                         -0.225558972234, -0.100240215031, 0.667071338154,
                         1.088251530500, 0.542813011213, -0.050256540092,
                         -0.045240772218, 0.070703567550, 0.008152816799,
                         -0.028786231926, -0.001137535314, 0.006495728375,
                         0.000080661204, -0.000649589896])
    else:
        raise Exception("""The specified wavelet filter is not implemented.
Check make_wavelet_filter for a full list of implemented filters.""")
    filt /= np.linalg.norm(filt)
    return filt


if __name__ == '__main__':
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
    wave = RedWave(x, wave_dimensions, wave_filter,
                   number_of_scales, isometric)
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
