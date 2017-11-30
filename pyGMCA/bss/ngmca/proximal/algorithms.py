# -\*- coding: utf-8 -\*-

r"""
algorithms.py - This file is part of pygmca.
The pygmca package aims at performing non-negative matrix factorization.
This module provides proximal algorithms.
Copyright 2014 CEA
Contributor : Jérémy Rapin (jeremy.rapin.math@gmail.com)
Created on September 30, 2014, last modified on December 14, 2014

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

from pyGMCA.core import Algorithm
import numpy as np

class ForwardBackward(Algorithm):
    r"""
    Forward-Backward algorithm.
    The Forward-Backward algorithm (see Combettes and Wajs 2005) solves
    min_x f(x)+\lambda g(x) with f convex differentiable with Lipschitz
    gradient, and g proper convex and lower semi-continuous function.
    
    Necessary and optional parameters can be provided in a dictionary or
    as keyword parameter at instanciating the class or when running the
    algorithm. In case of double affectation of a parameter, the one
    provided at runtime is used. The keywords parameters are listed below.
    
    Keyword parameters (required)
    ------------------
    - gradient (required): Function
        Gradient of function f.
    - initialization (required): algorithm dependent
        Initialization point of the algorithm.
    - lipschitz_constant (required): float
        Lipschitz constant of the gradient function of f.
    - proximal (required): Function
        Proximal of function g, with two arguments: current point and step.
    
    Keyword parameters (optional)
    ------------------
    - display_time (default: 0.5): float
        Minimum time between each call of the display function (to be used
        with display_function).
    - verbose (default: False): bool
        Display important parameters
    - maximum_iteration (default: 100): int
        Maximum number of steps to perform.
    - relative_difference_tolerance (default: 1e-05): float
        Relative difference tolerance between two consecutive iterates
        for stopping the algorithm (not activated if 0).
    - display_function (optional): function
        Function taking "_data" as parameter and making a display to be
        printed during the iterations.
    - recording_functions (optional): dict
        Dictionary of functions taking "_data" as parameter and
        returning a scalar or a dictionary of scalars. They will be
        recorded in the dictionary _data["recording"] under the name
        of the field function of its output is a scalar, or of the field
        if the output is a structure.
    - lambda (default: 1): float
        Value of lambda.
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
        - gradient (required): Function
            Gradient of function f.
        - initialization (required): algorithm dependent
            Initialization point of the algorithm.
        - lipschitz_constant (required): float
            Lipschitz constant of the gradient function of f.
        - proximal (required): Function
            Proximal of function g, with two arguments: current point and step.
        
        Keyword parameters (optional)
        ------------------
        - display_time (default: 0.5): float
            Minimum time between each call of the display function (to be used
            with display_function).
        - verbose (default: False): bool
            Display important parameters
        - maximum_iteration (default: 100): int
            Maximum number of steps to perform.
        - relative_difference_tolerance (default: 1e-05): float
            Relative difference tolerance between two consecutive iterates
            for stopping the algorithm (not activated if 0).
        - display_function (optional): function
            Function taking "_data" as parameter and making a display to be
            printed during the iterations.
        - recording_functions (optional): dict
            Dictionary of functions taking "_data" as parameter and
            returning a scalar or a dictionary of scalars. They will be
            recorded in the dictionary _data["recording"] under the name
            of the field function of its output is a scalar, or of the field
            if the output is a structure.
        - lambda (default: 1): float
            Value of lambda.
        """
        Algorithm.__init__(self)
        self.add_parameters(parameters, kargs)
        self._default_keyword_parameters.update({'gradient': None,
            'proximal': None,
            'lipschitz_constant': None,
            'lambda': 1,
            'initialization': None,
            'relative_difference_tolerance': 0.00001})
    
    def _initialize(self):
        r"Function called at the beginning of the algorithm."
        self.__gradient = self._parameters['gradient']
        self.__proximal = self._parameters['proximal']
        self.__L = self._parameters['lipschitz_constant']
        self.__lambda = self._parameters['lambda']
        self._data['x'] = self._parameters['initialization']
        self.__t = 1
        self._data['previous_x'] = self._data['x'].copy()
    
    def _terminate(self):
        r"Function called at the end of the algorithm."
        del self._data['previous_x']
    
    def _iterate(self):
        r"Function called at each iteration of the algorithm."
        t_next = (1 + np.sqrt(1 + 4 * self.__t * self.__t)) / 2
        w = (self.__t - 1) / t_next
        x = self._data['x']
        y = (1 + w) * x - w * self._data['previous_x']
        self._data['previous_x'] = x.copy()
        self.__t = t_next
        self._data['x'] = self.__proximal(y - self.__gradient(y) / self.__L,
                                          self.__lambda / self.__L)
    
    def _extract_current_iterate_matrix(self):
        r"Function called at each iteration to compute the relative difference."
        return self._data['x']


class GeneralizedForwardBackward(Algorithm):
    r"""
    Generalized Forward Bacward algorithm
    The Generalized Forward-Backward algorithm (see Raguet et al. 2001)
    solves:
    min_x f(x)+\sum_i \lambda_i g_i(x)
    with f convex differentiable with Lipschitz gradient and
    g_i proper convex and lower semi-continuous functions.
    Necessary and optional parameters can be provided in a dictionary or
    as keyword parameter at instanciating the class or when running the
    algorithm. In case of double affectation of a parameter, the one
    provided at runtime is used. The keywords parameters are listed below.
    
    Keyword parameters (required)
    ------------------
    - proximals (required): list of Functions
        List of proxima operators of functions g_i, with two
        arguments: current point and step.
    - gradient (required): Function
        Gradient of function f.
    - initialization (required): algorithm dependent
        Initialization point of the algorithm.
    - lipschitz_constant (required): float
        Lipschitz constant of the gradient function of f.
    - lambdas (required): numpy array
        Values of the sparsity parameter.
    
    Keyword parameters (optional)
    ------------------
    - display_time (default: 0.5): float
        Minimum time between each call of the display function (to be used
        with display_function).
    - verbose (default: False): bool
        Display important parameters
    - maximum_iteration (default: 100): int
        Maximum number of steps to perform.
    - relative_difference_tolerance (default: 1e-05): float
        Relative difference tolerance between two consecutive iterates
        for stopping the algorithm (not activated if 0).
    - display_function (optional): function
        Function taking "_data" as parameter and making a display to be
        printed during the iterations.
    - recording_functions (optional): dict
        Dictionary of functions taking "_data" as parameter and
        returning a scalar or a dictionary of scalars. They will be
        recorded in the dictionary _data["recording"] under the name
        of the field function of its output is a scalar, or of the field
        if the output is a structure.
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
        - proximals (required): list of Functions
            List of proxima operators of functions g_i, with two
            arguments: current point and step.
        - gradient (required): Function
            Gradient of function f.
        - initialization (required): algorithm dependent
            Initialization point of the algorithm.
        - lipschitz_constant (required): float
            Lipschitz constant of the gradient function of f.
        - lambdas (required): numpy array
            Values of the sparsity parameter.
        
        Keyword parameters (optional)
        ------------------
        - display_time (default: 0.5): float
            Minimum time between each call of the display function (to be used
            with display_function).
        - verbose (default: False): bool
            Display important parameters
        - maximum_iteration (default: 100): int
            Maximum number of steps to perform.
        - relative_difference_tolerance (default: 1e-05): float
            Relative difference tolerance between two consecutive iterates
            for stopping the algorithm (not activated if 0).
        - display_function (optional): function
            Function taking "_data" as parameter and making a display to be
            printed during the iterations.
        - recording_functions (optional): dict
            Dictionary of functions taking "_data" as parameter and
            returning a scalar or a dictionary of scalars. They will be
            recorded in the dictionary _data["recording"] under the name
            of the field function of its output is a scalar, or of the field
            if the output is a structure.
        """
        Algorithm.__init__(self)
        self.add_parameters(parameters, kargs)
        self._default_keyword_parameters.update({'gradient': None,
            'proximals': None,
            'lipschitz_constant': None,
            'lambdas': None,
            'initialization': None,
            'relative_difference_tolerance': 0.00001})
        self._num_prox = 0
    
    def _initialize(self):
        r"Function called at the beginning of the algorithm."
        self.__gradient = self._parameters['gradient']
        self.__proximals = self._parameters['proximals']
        self.__lambdas = self._parameters['lambdas']
        self._data['x'] = self._parameters['initialization'].copy()
        self.__num_prox = len(self._parameters['proximals'])
        if len(self.__proximals) != len(self.__lambdas):
            raise Exception("Number of lambdas and proximals should be equal.")
        self._data['z'] = [None] * self.__num_prox
        for k in range(0, self.__num_prox):
            self._data['z'][k] = self._data['x'].copy()
        #other constants
        self.__omega_i = 1.0 / self.__num_prox
        self.__beta = 1.0 / self._parameters['lipschitz_constant']
        self.__gamma_t = self.__beta
        # lambda_t in the paper
        self.__mu_t = min((1.0 + 2.0 * self.__beta / self.__gamma_t) / 2.0,
                          3.0 / 2.0) * 0.9
    
    def _terminate(self):
        r"Function called at the end of the algorithm."
        del self._data['z']
    
    def _iterate(self):
        r"Function called at each iteration of the algorithm."
        x = self._data['x'].copy()
        self._data['x'] *= 0
        temp = 2 * x - self.__gamma_t * self.__gradient(x)
        for k in range(0, self.__num_prox):
            self._data['z'][k] += self.__mu_t *\
                (self.__proximals[k](temp - self._data['z'][k],
                self.__lambdas[k] * (self.__gamma_t / self.__omega_i)) - x)
            # update x
            self._data['x'] += self.__omega_i * self._data['z'][k]
    
    def _extract_current_iterate_matrix(self):
        r"Function called at each iteration to compute the relative difference."
        return self._data['x']
