# -\*- coding: utf-8 -\*-

r"""
framework.py - This file is part of pygmca.
The pygmca package aims at performing non-negative matrix factorization.
This module implements a generic GMCA framework, using updaters.
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

from pyGMCA import core
from pyGMCA.bss import tools
import numpy as np




class Updater(object):
    r"""
    Updaters implement specific updates for the mixing coefficients
    A and the spectra A. They are designed to be used with the 
    Framework class.
    Required and optional keyword parameters
    are listed below.
    
    Keyword parameters (optional)
    ------------------
    - maximum_iteration (default: 80): int
        Maximum number of steps to perform.
    - verbose (default: 0): bool
        Display important parameters
    - relative_difference_tolerance (default: 1e-06): float
        Relative difference tolerance between two consecutive iterates
        for stopping the algorithm (not activated if 0).
    """

    def __init__(self, parameters=None, **kargs):
        r"""
        Updaters implement specific updates for the mixing coefficients
        A and the spectra A. They are designed to be used with the 
        Framework class.
        
        Inputs
        ------
        - parameters (default: None): dict
            Parameters dictionary, with potential keywords provided below.
        - any keyword argument from the ones listed below.
        
        Keyword parameters (optional)
        ------------------
        - maximum_iteration (default: 80): int
            Maximum number of steps to perform.
        - verbose (default: 0): bool
            Display important parameters
        - relative_difference_tolerance (default: 1e-06): float
            Relative difference tolerance between two consecutive iterates
            for stopping the algorithm (not activated if 0).
        """
        self._parameters = {}
        self.__add_parameters(parameters, kargs)
        self._default_keyword_parameters = {'maximum_iteration': 80,
            'relative_difference_tolerance': 10**-6,
            'verbose': 0}
        self._parameters = {}
        self._iteration = -1
        self.lambdas = 0
        self._refinement_iteration = 0
        
    def __add_parameters(self, parameters, kargs):
        r"""
        Update parameter dictionary with the ones provided.
        See class or constructor docstring for possible keywords.
        """
        if parameters is not None:
            self._parameters.update(parameters)
        self._parameters.update(kargs)
    
    def _set_parameters(self, parameters, kargs):
        r"""
        Update parameter dictionary with the default ones provided.
        See class or constructor docstring for possible keywords.
        """
        self.__add_parameters(parameters, kargs)
        self._parameters = core.tools.fill_parameters(self._parameters,
            self._default_keyword_parameters, self.__class__.__name__)
        # modifier are used to modify the data before the update
        # main use is for removing the coarse scale
        self._modifier = self._parameters.setdefault("modifier", None)
    
    def initialize(self, data, A, S, main_parameters=None, which='S'):
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
        - main_parameters (default: None): dict
            Parameters of the main loop.
        - which (default: 'S'): str
            string taking value either "A" or "S" depending on whether
            to process A or S.
        """
        self._iteration = -1
        if main_parameters is None:
            self._refinement_iteration = 0
        else:
            self._refinement_iteration =\
                np.floor(main_parameters['maximum_iteration'] *
                (1.0 - main_parameters['refinement_ratio']))
        self._initialize(data, A, S, main_parameters, which)
        # prepare the modification on the data
        if self._modifier is not None:
            self._data_m = self._modifier(data.T).T
            self._prev_data = data
    
    def update(self, data, A, S, lambdas=None):
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
        self._iteration += 1
        if self._modifier is not None:
            A_m = self._modifier(A.T).T
            if self._prev_data is not data:
                self._data_m = self._modifier(data.T).T
                self._prev_data = data
            return self._update(self._data_m, A_m, S, lambdas)
        else:
            return self._update(data, A, S, lambdas)
    
    
    def _initialize(self, data, A, S, refinement_iteration=0):
        r"""
        *Virtual function which must be implemented*
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
        - refinement_iteration (default: 0): int
            Iteration at which to begin the refinement phase.
        """
        raise NotImplementedError
    
    def _update(self, data, A, S, lambdas=None):
        r"""
        *Virtual function which must be implemented*
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
        raise NotImplementedError
    
    def process(self, data, A, S, lambdas=None):
        r"""
        Implements the resolution of an update without requiring the use of the
        full framework.
        
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
        """
        # avoid using refinement
        self.initialize(data, A, S)
        # update
        if lambdas is None:
            return self.update(data, A, S)
        else:
            return self.update(data, A, S, lambdas)


class Framework(core.Algorithm):
    r"""
    GMCA framework class. This class implements GMCA algorithms and
    take as inputs updater instances for A and S.
    Required and optional keyword parameters are listed below
    (to be provided before running the algorithm).
    
    Keyword parameters (required)
    ------------------
    - rank (required): int
        Rank of the factorization (number of source signals to recover).
    - S_updater (required): Updater
        Instance of Updater which must be used to update S.
    - A_updater (required): Updater
        Instance of Updater which must be used to update A.
    - data (required): numpy array
        Data array to be processed.
    
    Keyword parameters (optional)
    ------------------
    - display_time (default: 0.5): float
        Minimum time between each call of the display function (to be used
        with display_function).
    - verbose (default: False): bool
        Display important parameters
    - maximum_iteration (default: 100): int
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
    - rescaling (default: True): bool
        If True, normalizes S when updating A and A when updating S.
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
        - S_updater (required): Updater
            Instance of Updater which must be used to update S.
        - A_updater (required): Updater
            Instance of Updater which must be used to update A.
        - data (required): numpy array
            Data array to be processed.
        
        Keyword parameters (optional)
        ------------------
        - display_time (default: 0.5): float
            Minimum time between each call of the display function (to be used
            with display_function).
        - verbose (default: False): bool
            Display important parameters
        - maximum_iteration (default: 100): int
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
        - rescaling (default: True): bool
            If True, normalizes S when updating A and A when updating S.
        """
        core.Algorithm.__init__(self)
        self.add_parameters(parameters, kargs)
        self._default_keyword_parameters.update({'data': None,
            'rank': None,
            'rescaling': True,
            'refinement_ratio': 0.2,
            'S_updater': None,
            'A_updater': None,
            'relative_difference_tolerance': 0})
        self._refinement_iteration = 0
    
    def _initialize(self):
        r"Function called at the beginning of the algorithm."
        #prepare useful constants
        
        # use initialization if provided, else use the warm init. routine.
        data = self._parameters['data']
        if "initialization" in self._parameters.keys():
            self._data['factorization'] = self._parameters['initialization']
        else:
            self._data['factorization'] =\
                tools.warm_initialization(data,
                                             self._parameters['rank'])
        # get handles to the factorizations
        # modify strides so that each column of A and row of S is contiguous
        A = self._data['factorization'].A
        if max(A.strides) != A.strides[1]:
            print('Never tested: flipping A to get continues columns.')
            self._data['factorization'].A = A.copy(order='F')
            A = self._data['factorization'].A
        S = self._data['factorization'].S
        if max(S.strides) != S.strides[0]:
            print('Never tested: flipping S to get continues rows.')
            self._data['factorization'].S = S.copy(order='C')
            S = self._data['factorization'].S
        # initialize
        self._data["S_updater"] = self._parameters["S_updater"]
        self._data["A_updater"] = self._parameters["A_updater"]
        if self._parameters['rescaling']:
            self._data['factorization'].set_scale_on('S')
        self._data["S_updater"].initialize(data, A, S,
                                           self._parameters, 'S')
        if self._parameters['rescaling']:
            self._data['factorization'].set_scale_on('A')
        self._data["A_updater"].initialize(data.T, S.T, A.T,
                                           self._parameters, 'A')
        
    def _terminate(self):
        r"Function called at the end of the algorithm."
        self._data['factorization'].set_scale_on('S')
    
    def _iterate(self):
        r"Function called at each iteration of the algorithm."
        # obtain handles
        factor = self._data['factorization']
        data = self._parameters['data']
        # update lambda
        if self._parameters['rescaling']:
            factor.set_scale_on('S')
        # update S
        factor.S = self._data["S_updater"].update(data, factor.A, factor.S)
        # reinitialize null sources
        factor = tools.reinitialize_null_sources(factor, data,
            self._parameters['verbose'])
        # update A
        if self._parameters['rescaling']:
            factor.set_scale_on('A')
        factor.A = self._data["A_updater"].update(data.T, factor.S.T,
                                                  factor.A.T).T
    
    def _extract_current_iterate_matrix(self):
        r"Function called at each iteration to compute the relative difference."
        return self._data['factorization'].A.copy()
