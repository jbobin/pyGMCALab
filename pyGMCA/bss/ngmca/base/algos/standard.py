# -\*- coding: utf-8 -\*-
#
# r"""
# standard.py - This file is part of pygmca.
# The pygmca package aims at performing non-negative matrix factorization.
# This module implements the standard version of nGMCA.
# Copyright 2014 CEA
# Contributor : Jérémy Rapin (jeremy.rapin.math@gmail.com)
# Created on October 06, 2014, last modified on December 14, 2014
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# """
#
# __version__ = "1.0"
# __author__ = "Jeremy Rapin"
# __url__ = "http://www.cosmostat.org/GMCALab.html"
# __copyright__ = "(c) 2014 CEA"
# __license__ = "CeCill"

from pyGMCA.bss.ngmca import core
from pyGMCA.bss.ngmca.base import tools
import numpy as np


class Ngmca(core.Algorithm):
    r"""
    nGMCA algorithm.
    non-negative Generalized Morphological Component Analysis

    Aims at solving:
    argmin_(A >= 0, S >= 0) 1 / 2 * ||Y - A * S||_2^2 + lambda * ||S||_1
    using iterative soft-thresholding.
    Parameter lambda is decreasing during the iterations and set to
    tau_MAD*sigma_noise at the end of the algorithm, where tau_MAD is a
    constant (preferably in [1,3]) and sigma_noise is an online estimate of
    the noise standard deviation.

    For more information, this algorithm is described as nGMCA^S in:
    J. Rapin, J.Bobin, A. Larue and J.L. Starck,
    Sparse and Non-negative BSS for Noisy Data,
    IEEE Transactions on Signal Processing, 2013.
    Please use the above reference if using this code in a publication.


    Keyword parameters (required)
    ------------------
    - rank (required): int
        Rank of the factorization (number of source signals to recover).
    - data (required): numpy array
        Data array to be processed.

    Keyword parameters (optional)
    ------------------
    - linear_tau_mad_decrease (default: False): bool
        If True, lambda is computed as tau * noise and tau decreases linearly
        If False, lambda directly decreases linearly.
        (Set to True if the noise is not well estimated at the beginning, but
        False is more stable).
    - display_time (default: 0.5): float
        Minimum time between each call of the display function (to be used
        with display_function).
    - verbose (default: False): bool
        Display important parameters
    - maximum_iteration (default: 100): int
        Maximum number of steps to perform.
    - S_parameters (optional): dict
        Parameters for the update of S, with fields:
        - tau_mad (default: 1): float
            Thresholding at tau_mad * sigma (>0).
        - maximum_iteration (default: 80): int
            Maximum number of steps to perform.
    - relative_difference_tolerance (default: 0): float
        Relative difference tolerance between two consecutive iterates
        for stopping the algorithm (not activated if 0).
    - display_function (optional): function
        Function taking "_data" as parameter and making a display to be
        printed during the iterations.
    - refinement_ratio (default: 0.2): float
        Ratio of iterations kept for the refinement phase
        (between O and 1).
    - recording_functions (optional): dict
        Dictionary of functions taking "_data" as parameter and
        returning a scalar or a dictionary of scalars. They will be
        recorded in the dictionary _data["recording"] under the name
        of the field function of its output is a scalar, or of the field
        if the output is a structure.
    - A_parameters (optional): dict
        Parameters for the update of A, with fields:
        - maximum_iteration (default: 80): int
            Maximum number of steps to perform.
    - uniform_first_sparsity (default: True): bool
        If True, the first sparsity parameter (lambda) is the overall maximum
        on all the gradient, otherwise it is computed line by line of the
        gradient.


    Note
    ----
    Aims at solving the following problem:

    .. math::
        \text{argmin}_{\mathbf{S}\ge 0,~\mathbf{A}\ge 0} \frac{1}{2}\|\mathbf{Y} - \mathbf{A} \mathbf{S}\|_2^2 + \|\mathbf{\mu} \odot \mathbf{S}||_1
    """

    def __init__(self, parameters=None, **kargs):
        r"""
        Class constructor.
        Parameters can be provided upon instance creation
        (see inputs and keywords)

        Inputs
        ------
        - parameters (default: None): dict
            Parameters dictionary, with potential keywords provided below.
        - any keyword argument from the ones listed below.

        Keyword parameters (required)
        ------------------
        - rank (required): int
            Rank of the factorization (number of source signals to recover).
        - data (required): numpy array
            Data array to be processed.

        Keyword parameters (optional)
        ------------------
        - linear_tau_mad_decrease (default: False): bool
            If True, lambda is computed as tau * noise and tau decreases linearly
            If False, lambda directly decreases linearly.
            (Set to True if the noise is not well estimated at the beginning, but
            False is more stable).
        - display_time (default: 0.5): float
            Minimum time between each call of the display function (to be used
            with display_function).
        - verbose (default: False): bool
            Display important parameters
        - maximum_iteration (default: 100): int
            Maximum number of steps to perform.
        - S_parameters (optional): dict
            Parameters for the update of S, with fields:
            - tau_mad (default: 1): float
                Thresholding at tau_mad * sigma (>0).
            - maximum_iteration (default: 80): int
                Maximum number of steps to perform.
        - relative_difference_tolerance (default: 0): float
            Relative difference tolerance between two consecutive iterates
            for stopping the algorithm (not activated if 0).
        - display_function (optional): function
            Function taking "_data" as parameter and making a display to be
            printed during the iterations.
        - refinement_ratio (default: 0.2): float
            Ratio of iterations kept for the refinement phase
            (between O and 1).
        - recording_functions (optional): dict
            Dictionary of functions taking "_data" as parameter and
            returning a scalar or a dictionary of scalars. They will be
            recorded in the dictionary _data["recording"] under the name
            of the field function of its output is a scalar, or of the field
            if the output is a structure.
        - A_parameters (optional): dict
            Parameters for the update of A, with fields:
            - maximum_iteration (default: 80): int
                Maximum number of steps to perform.
        - uniform_first_sparsity (default: True): bool
            If True, the first sparsity parameter (lambda) is the overall maximum
            on all the gradient, otherwise it is computed line by line of the
            gradient.
        """
        core.Algorithm.__init__(self)
        self.add_parameters(parameters, kargs)
        self._default_keyword_parameters.update({'data': None,
            'rank': None,
            'refinement_ratio': 0.2,
            'S_parameters': {'maximum_iteration': 80, 'tau_mad': 1},
            'A_parameters': {'maximum_iteration': 80},
            'relative_difference_tolerance': 0,
            "linear_tau_mad_decrease": False,
            "uniform_first_sparsity": True})
        self.__refinement_iteration = 0
        self._current_tau_mad = None

    def _initialize(self):
        r"Function called at the beginning of the algorithm."
        #prepare useful constants
        self.__tau_mad = self._parameters['S_parameters']['tau_mad']
        self.__refinement_iteration =\
            np.floor(self._maximum_iteration *
                     (1.0 - self._parameters['refinement_ratio']))
        # use initialization if provided, else use the warm init. routine.
        if "initialization" in self._parameters.keys():
            self._data['factorization'] = self._parameters['initialization']
        else:
            self._data['factorization'] =\
                tools.warm_initialization(self._parameters['data'],
                                          self._parameters['rank'])

    def _terminate(self):
        r"Function called at the end of the algorithm."
        self._data['factorization'].set_scale_on('S')

    def _iterate(self):
        r"Function called at each iteration of the algorithm."
        # obtain handles
        factor = self._data['factorization']
        data = self._parameters['data']
        # update lambda
        factor.set_scale_on('S')
        self._update_lambda()
        # update S
        AtY = factor.A.T.dot(data)
        AtA = factor.A.T.dot(factor.A)
        factor.S = tools.nonnegative_sparse_inversion(AtY, AtA, factor.S,
            self._data['lambda'], self._parameters['S_parameters'])
        # reinitialize null sources
        factor = tools.reinitialize_null_sources(factor, data,
            self._parameters['verbose'])
        # update A
        if self._iteration >= self.__refinement_iteration:
            self._parameters['A_parameters']['norm_constrained'] = 1
        factor.set_scale_on('A')
        AtY = factor.S.dot(data.T)
        AtA = factor.S.dot(factor.S.T)
        factor.A = tools.nonnegative_sparse_inversion(AtY, AtA, factor.A.T,
            0, self._parameters['A_parameters']).T

    def _update_data_noise(self):
        r"Computes an estimate of the noise on the data."
        key = 'additive_standard_deviation'
        factor = self._data['factorization']
        data = self._parameters['data']
        if key in self._parameters.keys():
            self._data["data_noise"] = self._parameters[key]
        else:  # use MAD estimator otherwise
            self._data["data_noise"] = core.tools.dim_mad_std(
                data - factor.A.dot(factor.S), 1)

    def _update_lambda(self):
        r"""
        Updates the value of the lambda parameter from iteration to iteration.
        This value decreases down to the level of the noise.
        """
        factor = self._data['factorization']
        data = self._parameters['data']
        # initialize at first iteration
        if self._iteration < self.__refinement_iteration:
            self._update_data_noise()
            grad_noise =\
                np.sqrt((factor.A.T**2).dot(self._data["data_noise"]**2))
            if self._iteration == 0:
                # initialization can be made source by source
                self._data['lambda'] = np.max(np.abs(\
                    factor.A.T.dot(data - factor.A.dot(factor.S))),
                    axis=1)[:, None]
                # but this causes instabilities... better use a unique init.
                if self._parameters["uniform_first_sparsity"]:
                    self._data['lambda'] = np.max(self._data['lambda'])
                if self._parameters["linear_tau_mad_decrease"]:
                    self._current_tau_mad = self._data['lambda'] / grad_noise
            else:
                remaining_iterations = (self.__refinement_iteration -
                                        self._iteration)
                # two choices for decreasing lambda:
                # - linear decrease of lambda
                # - linear decrease of tau_mad (lambda is tau_mad * noise)
                if self._parameters["linear_tau_mad_decrease"]:
                    self._current_tau_mad =\
                        tools.linear_decrease(remaining_iterations,
                        self._current_tau_mad, self.__tau_mad)
                    # avoid huge increases of lambda when an estimate is bad
                    self._data['lambda']  = np.minimum(self._data['lambda'],
                        self._current_tau_mad * grad_noise)
                else:
                    self._data['lambda'] =\
                        tools.linear_decrease(remaining_iterations,
                        self._data['lambda'], self.__tau_mad * grad_noise)

    def _extract_current_iterate_matrix(self):
        r"Function called at each iteration to compute the relative difference."
        return self._data['factorization'].A.copy()
