# -\*- coding: utf-8 -\*-
#
# r"""
# algorithm.py - This file is part of pygmca.
# The pygmca package aims at performing non-negative matrix factorization.
# This module provides a generic algorithm class.
# Copyright 2014 CEA
# Contributor : Jérémy Rapin (jeremy.rapin.math@gmail.com)
# Created on September 30, 2014, last modified on December 14, 2014
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

from pyGMCA.bss.ngmca.core.tools import fill_parameters
from PyQt4.QtGui import QApplication
from gc import collect
from matplotlib import pyplot
from numpy import NaN, ones, isscalar, ndarray, linalg, finfo
import time


class Algorithm(object):
    r"""
    This virtual class provides a framework for algorithms.

    Notes
    -----
    Each heir class must implement functions:
        - _initialize, called at the beginning of the algorithm.
        - _iterate, called at each iteraion of the algorithm.
        - _terminate, called at the end of the algorithm.
    The iterates should be recorded in the dictionary _data
    Algorithms are called by function "run" which returns the results in a
    dictionary.

    Inputs
    ------
    Necessary and optional parameters depend on the heriting class and can be
    provided in a dictionary or as keyword parameter at instanciating the
    class or when running the algorithm. In case of double affectation of a
    parameter, the one provided at runtime is used.
    All algorithms share the keyword parameters below.

    Keyword parameters (optional)
    ------------------
    - display_function (optional): function
        Function taking "_data" as parameter and making a display to be
        printed during the iterations.
    - display_time (default: 0.5): float
        Minimum time between each call of the display function (to be used
        with display_function).
    - recording_functions (optional): dict
        Dictionary of functions taking "_data" as parameter and
        returning a scalar or a dictionary of scalars. They will be
        recorded in the dictionary _data["recording"] under the name
        of the field function of its output is a scalar, or of the field
        if the output is a structure.
    - verbose (default: False): bool
        Display important parameters
    - maximum_iteration (default: 100): int
        Maximum number of steps to perform.
    - relative_difference_tolerance (default: 0): float
        Relative difference tolerance between two consecutive iterates
        for stopping the algorithm (not activated if 0).
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

        Keyword parameters (optional)
        ------------------
        - display_function (optional): function
            Function taking "_data" as parameter and making a display to be
            printed during the iterations.
        - display_time (default: 0.5): float
            Minimum time between each call of the display function (to be used
            with display_function).
        - recording_functions (optional): dict
            Dictionary of functions taking "_data" as parameter and
            returning a scalar or a dictionary of scalars. They will be
            recorded in the dictionary _data["recording"] under the name
            of the field function of its output is a scalar, or of the field
            if the output is a structure.
        - verbose (default: False): bool
            Display important parameters
        - maximum_iteration (default: 100): int
            Maximum number of steps to perform.
        - relative_difference_tolerance (default: 0): float
            Relative difference tolerance between two consecutive iterates
            for stopping the algorithm (not activated if 0).
        """
        self._default_keyword_parameters = {'maximum_iteration': 100,
            'verbose':  False,
            'recording_functions': {},
            'display_time': 0.5,
            'display_function': self.__no_display}
        # add the relative difference tolerance
        # not in the above list to avoid referencing in the default parameters
        # since it is not always implemented in the subclasses
        self._default_keyword_parameters.setdefault(
            'relative_difference_tolerance', 0)
        self._iteration = 0
        self._maximum_iteration = 0
        self._data = {}  # variable which gathers everything that moves
        self._data['recording'] = {}
        self._parameters = {'verbose': 0}
        self.add_parameters(parameters, kargs)
        self.__recording_functions = {}
        self._current_iterate = None  # for relative difference tolerance

    def __no_display(self, data):
        r"Abstract class used only for default parameter. Does nothing."
        pass

    def add_parameters(self, parameters, kargs):
        r"""
        Update parameter dictionary with the one provided
        See class or constructor docstring for possible keywords.
        """
        if parameters is not None:
            self._parameters.update(parameters)
        self._parameters.update(kargs)

    def _initialize(self):
        r"Function called at the beginning of the algorithm."
        raise NotImplementedError

    def _terminate(self):
        r"Function called at the end of the algorithm."
        raise NotImplementedError

    def _iterate(self):
        r"Function called at each iteration of the algorithm."
        raise NotImplementedError

    def name(self):
        r"Name of the algorithm"
        return self.__class__.__name__

    def reset_parameters(self):
        r"Reset all keyword parameters to their default value."
        self._parameters = {}

    def _extract_current_iterate_matrix(self):
        r"Function called at each iteration to compute the relative difference."
        raise NotImplementedError

    def run(self, parameters=None, **kargs):
        r"""
        Run the algorithm.

        Inputs
        ------
        - parameters (default: None): dict
            Parameters dictionary, with potential keywords provided below.
        - any keyword argument from the ones listed below.

        Keywords
        --------
        See class or constructor docstring for possible keywords.
        """
        self.add_parameters(parameters, kargs)
        self._parameters = fill_parameters(self._parameters,
                                           self._default_keyword_parameters,
                                           self.name())
        display = 0
        if self._parameters['display_function'].__name__ is not '__no_display':
            self._parameters['verbose'] = 1
            display = 1
            pyplot.figure(self.__class__.__name__ + " display")
        self._maximum_iteration = self._parameters["maximum_iteration"]
        # initialize the algorithm
        self._initialize()
        # extract current iterate for relative difference tolerance checking
        rel_tol = self._parameters['relative_difference_tolerance']
        if rel_tol > 0:
            try:
                self._current_iterate = self._extract_current_iterate_matrix().copy()
            except NotImplementedError:
                rel_tol = 0
                if self._parameters['verbose'] != 0:
                    print("""Relative difference tolerance stopping criteria
desactivated (_extract_current_iterate_matrix not implemented).""")
        # launch the loop
        last_display_time = time.time()
        beginning_time = last_display_time
        for self._iteration in range(0, self._maximum_iteration):
            self._iterate()
            # recording
            self.__recording()
            # drawing
            if display:
                if (time.time() - last_display_time >
                        self._parameters['display_time']) or\
                        (self._maximum_iteration == self._iteration + 1):
                    pyplot.clf()  # close()
                    collect()
                    self._parameters['display_function'](self._data)
                    QApplication.processEvents()  # plt.draw()#time.sleep(0.01)
                    last_display_time = time.time()
            if self._parameters['verbose']:
                print("Iteration %s/%s." % (self._iteration + 1,
                                            self._maximum_iteration))
            # check for relative error
            if rel_tol > 0:
                new_iterate = self._extract_current_iterate_matrix().copy()
                if (linalg.norm(new_iterate - self._current_iterate) /
                    (linalg.norm(self._current_iterate)
                     + finfo(float).eps)) < rel_tol:
                    if self._parameters['verbose']:
                        print("Stopping (relative difference is too small)")
                    break
                else:
                    self._current_iterate = new_iterate
        # terminate
        self._terminate()
        if self._parameters['verbose']:
            print("Total computation time: %s." % (time.time() -
                                                   beginning_time))
        return self._data

    def __recording(self):
        r"""
        Function aiming at recording the information computed by
        the recording_functions (from the parameters).
        """
        for k in self._parameters['recording_functions'].keys():
            temp = self._parameters['recording_functions'][k](self._data)
            if self._iteration == 0:
                if (isscalar(temp) or
                    (((type(temp) == list) or
                        (type(temp) == ndarray)) and
                        len(temp) == 1)):
                    self._data['recording'][k] = NaN * ones(
                        self._maximum_iteration)
                    if self._parameters['verbose']:
                        print("Adding recorded feature: %s." % k)
                else:
                    if type(temp) == dict:
                        string = "Adding recorded features: "
                        for i in temp.keys():
                            string = string + i + ", "
                            self._data['recording'][i] = NaN * ones(
                                self._maximum_iteration)
                        if self._parameters['verbose'] == 1:
                            print(string[0: len(string) - 2] + ".")
                    else:
                        raise("Recording function is neither\
                            scalar nor dictionary")
            if (isscalar(temp) or
               (((type(temp) == list) or
                   (type(temp) == ndarray)) and len(temp) == 1)):
                self._data['recording'][k][self._iteration] = temp
            else:
                if type(temp) == dict:
                    for i in temp.keys():
                        self._data['recording'][i][self._iteration] = temp[i]
