# -\*- coding: utf-8 -\*-

r"""
updaters.py - This file is part of pygmca.
The pygmca package aims at performing non-negative matrix factorization.
This module implements updaters to be used with the generic GMCA framework.
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
from pyGMCA.bss.ngmca.framework import Updater
import numpy as np
from pyGMCA.core import tools
from pyGMCA import proximal
from pyGMCA.bss import tools as bsstools


class SparseUpdater(Updater):
    r"""
    Updaters implement specific updates for the mixing coefficients
    A and the spectra A. They are designed to be used with the 
    Framework class.
    This updater aims at solving problem:
    argmin_{S >= 0} ||Y - A S||_2^2 + ||lambda .* S||_1
    using FISTA (Beck & Teboulle 2009).
    Required and optional keyword parameters
    are listed below.
    
    Keyword parameters (optional)
    ------------------
    - linear_tau_mad_decrease (default: False): bool
        If True, lambda is computed as tau * noise and tau decreases linearly
        If False, lambda directly decreases linearly.
        (Set to True if the noise is not well estimated at the beginning, but
        False is more stable).
    - verbose (default: 0): bool
        Display important parameters
    - maximum_iteration (default: 80): int
        Maximum number of steps to perform.
    - hardthresholding (default: False): bool
        Use hard-thresholding instead of soft-thresholding.
    - relative_difference_tolerance (default: 1e-05): float
        Relative difference tolerance between two consecutive iterates
        for stopping the algorithm (not activated if 0).
    - tau_mad (default: 1): float
        Thresholding at tau_mad * sigma (>0).
    - reweighted_l1 (default: 0): float
        No reweighting if 0, else the reweighting is computing using
        a charactheristic value of reweighted_l1 * noise_estimation.
    - uniform_first_sparsity (default: True): bool
        If True, the first sparsity parameter (lambda) is the overall maximum
        on all the gradient, otherwise it is computed line by line of the
        gradient.
    - nonnegative (default: True): bool
        If True, uses a non-negative constraint.
    
    
    Note
    ----
    This updater aims at solving:
    
    .. math::
        \text{argmin}_{\mathbf{S}\ge 0} \frac{1}{2}\|\mathbf{Y} - \mathbf{A} \mathbf{S}\|_2^2 + \|\mathbf{\mu} \odot \mathbf{S}||_1
    
    .
    """

    def __init__(self, parameters=None, **kargs):
        r"""
        Updaters implement specific updates for the mixing coefficients
        A and the spectra A. They are designed to be used with the 
        Framework class.
        This updater aims at solving problem:
        argmin_{S >= 0} ||Y - A S||_2^2 + ||lambda .* S||_1
        using FISTA (Beck & Teboulle 2009).
        
        Inputs
        ------
        - parameters (default: None): dict
            Parameters dictionary, with potential keywords provided below.
        - any keyword argument from the ones listed below.
        
        Keyword parameters (optional)
        ------------------
        - linear_tau_mad_decrease (default: False): bool
            If True, lambda is computed as tau * noise and tau decreases linearly
            If False, lambda directly decreases linearly.
            (Set to True if the noise is not well estimated at the beginning, but
            False is more stable).
        - verbose (default: 0): bool
            Display important parameters
        - maximum_iteration (default: 80): int
            Maximum number of steps to perform.
        - hardthresholding (default: False): bool
            Use hard-thresholding instead of soft-thresholding.
        - relative_difference_tolerance (default: 1e-05): float
            Relative difference tolerance between two consecutive iterates
            for stopping the algorithm (not activated if 0).
        - tau_mad (default: 1): float
            Thresholding at tau_mad * sigma (>0).
        - reweighted_l1 (default: 0): float
            No reweighting if 0, else the reweighting is computing using
            a charactheristic value of reweighted_l1 * noise_estimation.
        - uniform_first_sparsity (default: True): bool
            If True, the first sparsity parameter (lambda) is the overall maximum
            on all the gradient, otherwise it is computed line by line of the
            gradient.
        - nonnegative (default: True): bool
            If True, uses a non-negative constraint.
        
        
        Note
        ----
        This updater aims at solving:
        
        .. math::
            \text{argmin}_{\mathbf{S}\ge 0} \frac{1}{2}\|\mathbf{Y} - \mathbf{A} \mathbf{S}\|_2^2 + \|\mathbf{\mu} \odot \mathbf{S}||_1
        
        .
        """
        Updater.__init__(self)
        self._default_keyword_parameters.update({"tau_mad": 1,
            "maximum_iteration": 80,
            "nonnegative": True,
            "hardthresholding": False,
            "reweighted_l1": 0,
            "relative_difference_tolerance": 10**-5,
            "linear_tau_mad_decrease": False,
            "uniform_first_sparsity": True})
        self._set_parameters(parameters, kargs)
        self._tau_mad = self._parameters["tau_mad"]
        self._current_tau_mad = None
        self.noise = None
    
    def _initialize(self, data, A, S, main_parameters, which):
        r"""
        Function called at the beginning of the algorithm so as to
        prepare the update steps.
        
        Inputs
        ------
        - data: numpy array
            Data array to be processed.
        - A: numpy array
            Mixing coefficients array.
        - S: numpy array
            Spectra array.
        - main_parameters: dict
            Parameters of the main loop.
        - which: str
            string taking value either "A" or "S" depending on whether
            to process A or S.
        """
        # prepare forward-backward algorithm
        rel_tol = self._parameters['relative_difference_tolerance']
        self._algo = proximal.algorithms.ForwardBackward(
            maximum_iteration=self._parameters['maximum_iteration'],
            relative_difference_tolerance=rel_tol)
        # if we use sparsity, initialize the threshold
        if self._tau_mad > 0:
            self.noise = tools.dim_mad_std(data - A.dot(S), 1)
            grad_noise = np.sqrt((A.T**2).dot(self.noise**2))
            # initialization can be made source by source
            # but this causes instabilities... better use a unique init.
            #self.lambdas = np.max(np.abs(A.T.dot(data - A.dot(S))),
            #                      axis=1)[:, None]
            self.lambdas = np.max(np.abs(A.T.dot(data - A.dot(S))),
                                  axis=1)[:, None]
            if self._parameters["uniform_first_sparsity"]:
                self.lambdas = np.max(self.lambdas)
            if self._parameters["linear_tau_mad_decrease"]:
                self._current_tau_mad = self.lambdas / grad_noise
    
    def _update(self, data, A, S, lambdas=None):
        r"""
        Function called in the Framework so as to update S
        (so as to update A, the inputs are transposed).
        
        Inputs
        ------
        - data: numpy array
            Data array to be processed.
        - A: numpy array
            Mixing coefficients array.
        - S: numpy array
            Spectra array.
        - lambdas (default: None): numpy array
            Values of the sparsity parameter.
        
        Output
        ------
        S: update value for S.
        """
        # update the threshold
        if lambdas is not None:
            # if a value is already provided
            # to be used independently with the process function
            self.lambdas = lambdas
        else:
            if self._iteration < self._refinement_iteration:
                if self._tau_mad > 0 and self._iteration > 0:
                    self._update_lambdas(data, A, S)
        # reweighted L1 (not tested)
        reweighted_l1 = self._parameters["reweighted_l1"]
        if (reweighted_l1 > 0) and (np.max(self.lambdas) > 0):
            S_inv = np.linalg.lstsq(data, A)[0]
            noise_estimate = reweighted_l1 * tools.dim_mad_std(S_inv, 1)
            self.lambdas /= 1.0 + (abs(S_inv) / noise_estimate)**2
        # proximal operator
        prox = self._choose_proximal(S)
        # prepare the input arrays
        AtY = A.T.dot(data)
        AtA = A.T.dot(A)
        # algorithm parameters
        parameters = {'gradient': (lambda z: AtA.dot(z) - AtY),
                      'proximal': prox,
                      'lambda': self.lambdas,
                      'lipschitz_constant': np.linalg.norm(AtA, 2),
                      'initialization': S}
        res = self._algo.run(parameters)
        return res["x"]
    
    def _choose_proximal(self, S):
        r"""
        Choose the proximal operator to be used among soft and hard-thresholding,
        non-negative or not, and norm projection for the refinement phase if
        tau_mad is null.
        """
        # non-negative constraint
        nn = self._parameters["nonnegative"]
        if self._parameters["hardthresholding"]:  # hard-thresholding
            prox = (proximal.operators.nonnegative_hard_thresholding if nn
                    else proximal.operators.hard_thresholding)
        else:  # soft-thresholding
            prox = (proximal.operators.nonnegative_soft_thresholding if nn
                    else proximal.operators.soft_thresholding)
            # norm constrained during refinement phase if no sparsity
            if self._tau_mad == 0:
                if self._iteration >= self._refinement_iteration:
                    n_S = tools.dim_norm(S, 1)
                    
                    def norm_prox(z, threshold):
                        r"""
                        Norm projection for the lines of S, used during the refinement
                        phase if tau_mad is null.
                        """
                        return proximal.operators.norm_projection(
                            prox(z, 0), n_S)
                    return norm_prox
        return prox
    
    def _update_lambdas(self, data, A, S):
        r"Updates the value of the sparsity parameter lambdas."
        # update noise
        key = 'additive_standard_deviation'
        if key in self._parameters.keys():
            self.noise = self._parameters[key]
        else:  # use MAD estimator otherwise
            self.noise = tools.dim_mad_std(data - A.dot(S), 1)
        # update lambdas
        grad_noise = np.sqrt((A.T**2).dot(self.noise**2))
        remaining_iterations = (self._refinement_iteration -
                                self._iteration)
        # two choices for decreasing lambda:
        # - linear decrease of lambda
        # - linear decrease of tau_mad (lambda is tau_mad * noise)
        if self._parameters["linear_tau_mad_decrease"]:
            self._current_tau_mad = bsstools.linear_decrease(
                remaining_iterations, self._current_tau_mad, self._tau_mad)
            self.lambdas = np.minimum(self.lambdas,
                                      self._current_tau_mad * grad_noise)
        else:
            self.lambdas = bsstools.linear_decrease(remaining_iterations,
                self.lambdas, self._tau_mad * grad_noise)


class RedWaveUpdater(Updater):
    r"""
    Updaters implement specific updates for the mixing coefficients
    A and the spectra A. They are designed to be used with the 
    Framework class.
    This updater aims at solving problem either
    (with W a redundant wavelet transform) the analysis formulation:
    argmin_{S >= 0} ||Y - A * S||_2^2 + ||lambda .* (S * W^T)||_1
    using the Chambolle-Pock algorithm (Chambolle & Pock 2010),
    or the synthesis formulation:
    argmin_{S_w * W >= 0} ||Y - A * S_w * W||_2^2 + ||lambda .* S_w)||_1
    using the Generalized Forward-Backward algorithm (Raguet et al, 2013).
    Required and optional keyword parameters
    are listed below.
    
    Keyword parameters (required)
    ------------------
    - redwave_operator (required): RedWave instance
        Wavelet operator to be applied on the sources.
    
    Keyword parameters (optional)
    ------------------
    - linear_tau_mad_decrease (default: False): bool
        If True, lambda is computed as tau * noise and tau decreases linearly
        If False, lambda directly decreases linearly.
        (Set to True if the noise is not well estimated at the beginning, but
        False is more stable).
    - direct_sparsity (default: True): bool
        Whether the sources are sparse in the direct domain
        (if False, the coarse scale will not be penalized).
    - verbose (default: 0): bool
        Display important parameters
    - maximum_iteration (default: 24): int
        Maximum number of steps to perform.
    - relative_difference_tolerance (default: 1e-06): float
        Relative difference tolerance between two consecutive iterates
        for stopping the algorithm (not activated if 0).
    - formulation (default: "analysis"): str
        Set to "analysis" or "synthesis" depending on the formulation
        you want to use.
    - tau_mad (default: 1): float
        Thresholding at tau_mad * sigma (>0).
    - reweighted_l1 (default: 0): float
        No reweighting if 0, else the reweighting is computing using
        a charactheristic value of reweighted_l1 * noise_estimation.
    - uniform_first_sparsity (default: True): bool
        If True, the first sparsity parameter (lambda) is the overall maximum
        on all the gradient, otherwise it is computed line by line of the
        gradient.
    - nonnegative (default: True): bool
        If True, uses a non-negative constraint.
    
    
    Note
    ----
    This updater aims at solving:
    
    .. math::
        \text{argmin}_{\mathbf{S}\ge 0} \frac{1}{2}\|\mathbf{Y} - \mathbf{A} \mathbf{S}\|_2^2 + \|\mathbf{\mu} \odot (\mathbf{W}^T\mathbf{S})||_1
    
    .
    """

    def __init__(self, parameters=None, **kargs):
        r"""
        Updaters implement specific updates for the mixing coefficients
        A and the spectra A. They are designed to be used with the 
        Framework class.
        This updater aims at solving problem either
        (with W a redundant wavelet transform) the analysis formulation:
        argmin_{S >= 0} ||Y - A * S||_2^2 + ||lambda .* (S * W^T)||_1
        using the Chambolle-Pock algorithm (Chambolle & Pock 2010),
        or the synthesis formulation:
        argmin_{S_w * W >= 0} ||Y - A * S_w * W||_2^2 + ||lambda .* S_w)||_1
        using the Generalized Forward-Backward algorithm (Raguet et al, 2013).
        
        Inputs
        ------
        - parameters (default: None): dict
            Parameters dictionary, with potential keywords provided below.
        - any keyword argument from the ones listed below.
        
        Keyword parameters (required)
        ------------------
        - redwave_operator (required): RedWave instance
            Wavelet operator to be applied on the sources.
        
        Keyword parameters (optional)
        ------------------
        - linear_tau_mad_decrease (default: False): bool
            If True, lambda is computed as tau * noise and tau decreases linearly
            If False, lambda directly decreases linearly.
            (Set to True if the noise is not well estimated at the beginning, but
            False is more stable).
        - direct_sparsity (default: True): bool
            Whether the sources are sparse in the direct domain
            (if False, the coarse scale will not be penalized).
        - verbose (default: 0): bool
            Display important parameters
        - maximum_iteration (default: 24): int
            Maximum number of steps to perform.
        - relative_difference_tolerance (default: 1e-06): float
            Relative difference tolerance between two consecutive iterates
            for stopping the algorithm (not activated if 0).
        - formulation (default: "analysis"): str
            Set to "analysis" or "synthesis" depending on the formulation
            you want to use.
        - tau_mad (default: 1): float
            Thresholding at tau_mad * sigma (>0).
        - reweighted_l1 (default: 0): float
            No reweighting if 0, else the reweighting is computing using
            a charactheristic value of reweighted_l1 * noise_estimation.
        - uniform_first_sparsity (default: True): bool
            If True, the first sparsity parameter (lambda) is the overall maximum
            on all the gradient, otherwise it is computed line by line of the
            gradient.
        - nonnegative (default: True): bool
            If True, uses a non-negative constraint.
        
        
        Note
        ----
        This updater aims at solving:
        
        .. math::
            \text{argmin}_{\mathbf{S}\ge 0} \frac{1}{2}\|\mathbf{Y} - \mathbf{A} \mathbf{S}\|_2^2 + \|\mathbf{\mu} \odot (\mathbf{W}^T\mathbf{S})||_1
        
        .
        """
        Updater.__init__(self)
        self._default_keyword_parameters.update({"tau_mad": 1,
            "maximum_iteration": 24,
            "nonnegative": True,
            "reweighted_l1": 0,
            "linear_tau_mad_decrease": False,
            "uniform_first_sparsity": True,
            "redwave_operator": None,
            "formulation": "analysis",
            "direct_sparsity": True})
        self._set_parameters(parameters, kargs)
        self._tau_mad = self._parameters["tau_mad"]
        self._current_tau_mad = None
        self._wave = self._parameters["redwave_operator"]
        self.noise = None
        
    def _initialize(self, data, A, S, main_parameters, which):
        r"""
        Function called at the beginning of the algorithm so as to
        prepare the update steps.
        
        Inputs
        ------
        - data: numpy array
            Data array to be processed.
        - A: numpy array
            Mixing coefficients array.
        - S: numpy array
            Spectra array.
        - main_parameters: dict
            Parameters of the main loop.
        - which: str
            string taking value either "A" or "S" depending on whether
            to process A or S.
        """
        if self._tau_mad > 0:
            if len(self._wave.get_wavelet_dimensions()) > 1:
                if data.strides[0] != max(data.strides):
                    raise Exception("Reshaping from 1D to 2D may fail since\
the sources are not continuous")
            # from line configuration to full shape
            # and from shape to line (if more than 1D transform)
            one_d = len(self._wave.get_wavelet_dimensions()) == 1
            if one_d:
                self._l2s = lambda data: data
                self._s2l = lambda data: data
            else:
                data_shape = self._wave.get_input_shape()[1:]
                self._l2s = lambda data:\
                    data.reshape([data.shape[0]] + data_shape)
                self._s2l = lambda data:\
                    data.reshape([data.shape[0], np.prod(data.shape[1:])])
            wave_temp = self._wave.forward(self._l2s(data - A.dot(S)))
            self.noise = self._wave.extract_scale_vals(wave_temp,
                                                       tools.dim_mad_std)
            # use einsum to be able to deal with 2D data
            # (equivalent to A.T.dot(wave_temp) in 1D)
            wave_temp = np.einsum('ij...,ik...->jk...', A, wave_temp)
            self.lambdas = self._wave.extract_scale_vals(
                np.abs(wave_temp),
                lambda x: np.max(x))  # lambda x: np.max(x, axis=(0, 1))
            # uniformize the first thresholds over all sources for more
            # stability
            if self._parameters["uniform_first_sparsity"]:
                self.lambdas = np.repeat(np.max(self.lambdas,
                                                axis=0)[np.newaxis, :],
                                         S.shape[0], axis=0)
            if self._parameters["linear_tau_mad_decrease"]:
                grad_noise = np.sqrt(np.einsum('ij...,ik...->jk...',
                                               A**2, self.noise**2))
                # grad_noise = np.sqrt((A.T**2).dot(self.noise**2))
                self._current_tau_mad = self.lambdas / grad_noise
            # for the synthesis update, intialize the GFB algorithm
            if self._parameters["formulation"] == "synthesis":
                print("Synthesis was not thoroughly tested yet.")
                rel_tol = self._parameters['relative_difference_tolerance']
                self._gfb = proximal.algorithms.GeneralizedForwardBackward(
                    maximum_iteration=self._parameters['maximum_iteration'],
                    relative_difference_tolerance=rel_tol)
        else:
            raise Exception("No sparsity: use SparseUpdater instead.")
            
    def _update(self, data, A, S, lambdas=None):
        r"""
        Function called in the Framework so as to update S
        (so as to update A, the inputs are transposed).
        
        Inputs
        ------
        - data: numpy array
            Data array to be processed.
        - A: numpy array
            Mixing coefficients array.
        - S: numpy array
            Spectra array.
        - lambdas (default: None): numpy array
            Values of the sparsity parameter.
        
        Output
        ------
        S: update value for S.
        """
        AtY = A.T.dot(data)
        AtA = A.T.dot(A)
        # update the threshold
        if lambdas is not None:
            # if a value is already provided
            # to be used independently with the process function
            self.lambdas = lambdas
        else:
            if self._iteration < self._refinement_iteration:
                if self._tau_mad > 0 and self._iteration > 0:
                    self._update_lambdas(data, A, S)
        # make full dimension lambda if it is only a per scale value
        if self.lambdas.shape[1] < AtY.shape[1]:
            lambdas = self._wave.make_full_vals(self.lambdas)
        # reweighted L1 (not tested, beware reference modification)
        reweighted_l1 = self._parameters["reweighted_l1"]
        if (reweighted_l1 > 0) and (np.max(lambdas) > 0):
            S_inv_w = self._wave.forward(self._l2s(np.linalg.lstsq(AtA,
                                                                  AtY)[0]))
            scale_std = self._wave.extract_scale_vals(S_inv_w,
                                                      tools.dim_mad_std)
            scale_std *= reweighted_l1 
            lambdas /= 1.0 + (abs(S_inv_w) /
                              self._wave.make_full_vals(scale_std))**2
        # remove coarse scale sparsity if the data is not sparse in the
        # direct domain
        if not self._parameters["direct_sparsity"]:
            one_d = len(self._wave.get_wavelet_dimensions()) == 1
            if one_d:
                self.lambdas[:, 0] = 0
            else:
                self.lambdas[:, 0, 0] = 0
        # launch the update (either analysis or synthesis)
        if self._parameters["formulation"] == "analysis":
            return self._s2l(self._wave.sparse_inversion(self._l2s(S),
                AtA, self._l2s(AtY), lambdas, non_negative=True))
        elif self._parameters["formulation"] == "synthesis":
            return self._s2l(self._synthesis_update(self._l2s(S), AtA,
                                                    self._l2s(AtY), lambdas))
        else:
            raise Exception("Type can only be analysis or synthesis.")
    
    def _update_lambdas(self, data, A, S):
        r"Updates the value of the sparsity parameter lambdas."
        # update noise
        key = 'additive_standard_deviation'
        if key in self._parameters.keys():
            self.noise = self._parameters[key]
        else:  # use MAD estimator otherwise
            # possible improvements: do not update at each iteration
            # or only partly at each iteration
            # and use the "transform" function to avoid new allocation
            wave_temp = self._wave.forward(self._l2s(data - A.dot(S)))
            self.noise = self._wave.extract_scale_vals(wave_temp,
                                                       tools.dim_mad_std)
        # update lambdas
        grad_noise = np.sqrt(np.einsum('ij...,ik...->jk...', A**2,
                                       self.noise**2))
        remaining_iterations = (self._refinement_iteration -
                                self._iteration)
        # two choices for decreasing lambda:
        # - linear decrease of lambda
        # - linear decrease of tau_mad (lambda is tau_mad * noise)
        if self._parameters["linear_tau_mad_decrease"]:
            self._current_tau_mad = bsstools.linear_decrease(
                remaining_iterations, self._current_tau_mad, self._tau_mad)
            self.lambdas = np.minimum(self.lambdas,
                                      self._current_tau_mad * grad_noise)
        else:
            self.lambdas = bsstools.linear_decrease(remaining_iterations,
                self.lambdas, self._tau_mad * grad_noise)
    
    def _synthesis_update(self, S, AtA, AtY, lambdas):
        r"Implements the update in the case of a synthesis formulation."
        parameters = {}
        parameters['gradient'] =\
            lambda Sw: self._wave.forward(np.einsum('ij...,jk...->ik...', AtA,
                self._wave.backward(Sw)) - AtY)
        
        def proj(x, threshold):  # non-negativity proximal operator
            temp = np.maximum(0.0, -self._wave.backward(x))
            return x + self._wave.forward(temp)
        parameters['proximals'] = [proximal.operators.soft_thresholding, proj]
        parameters['lambdas'] = [lambdas, 0.0]
        parameters['lipschitz_constant'] = np.linalg.norm(AtA, 2)
        parameters['initialization'] = self._wave.forward(S)
        res = self._gfb.run(parameters)
        return np.maximum(0.0, self._wave.backward(res['x']))
