# # -\*- coding: utf-8 -\*-
#
# r"""
# benchmark.py - This file is part of pygmca.
# The pygmca package aims at performing non-negative matrix factorization.
# This module provides processing tools.
# Copyright 2014 CEA
# Contributor : Jérémy Rapin (jeremy.rapin.math@gmail.com)
# Created on December 13, 2014, last modified on December 14, 2014
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

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:07:23 2014

@author: jr232749
"""
import cPickle
import numpy as np
import datetime
import time
import sys
import os
from pyGMCA.bss.ngmca.base.tools import evaluation as bss_eval
import matplotlib.pyplot as plt
from PyQt4.QtGui import QApplication
from gc import collect


class Benchmark(object):
    r"""
    Class implementing the computation, loading and display of a benchmark.
    Benchmarks can be initialized from a config file (.py) or a
    computed benchmark file (.bch). Examples of files are provided in a separate
    benchmark directory.

    Input
    -----
    filepath: string
        relative path to a config file (.py) or a  computed
        benchmark file (.bch).
    """

    def __init__(self, filepath=None):
        r"""
        Initialization of the benchmark from a config file (.py) or a
        computed benchmark file (.bch). Examples of files are provided in a separate
        benchmark directory.

        Input
        -----
        filepath: string
            relative path to a config file (.py) or a  computed
            benchmark file (.bch).
        """
        # if opening setting file
        if filepath.split(".")[-1] == "py":
            f = open(filepath)
            file_lines = f.readlines()
            f.close()
            file_string = "\n".join("".join(file_lines).splitlines())
            self._file_string = file_string
            self._hash = abs(hash(self._file_string)) % (10 ** 6)
            #
            # check settings
            error_text = "Setting file should have defined %s."
            exec file_string
            try:
                data_settings
            except NameError:
                raise NameError(error_text % "data_settings")
            try:
                algorithm_settings
            except NameError:
                raise NameError(error_text % "algorithm_settings")
            try:
                processing_settings
            except NameError:
                raise NameError(error_text % "processing_settings")
            try:
                algorithms
            except NameError:
                raise NameError(error_text % "algorithms")
            # bench variable
            self._bench_variable = None
            self._num_points = None
            self._check_bench_variable(data_settings)
            self._check_bench_variable(algorithm_settings)
            if self._bench_variable is None:
                raise Exception("""There should be exactly one bench variable.
(bench are defined by tuples)""")
            # algorithm names
            self._algorithm_names = algorithms.keys()
            # number of repetitions
            self._num_repet = self._extract_value(processing_settings,
                                                  "processing_settings",
                                                  "number_of_repetitions")
            if self._num_repet < 2:
                raise Exception("At least 2 repetitions are necessary.")
            self._criteria = self._extract_value(processing_settings,
                                                       "processing_settings",
                                                       "criteria")
            # prepare results
            self._end_time = None
            self._type = ("mean", "std", "median")
            self.results = None
            self.data_structure = 'n_points x n_algo x n_crit x [mean, std, med]'
            # prepare filename
            self.name = self._make_name()
        # if opening benchmark file
        elif filepath.split(".")[-1] == "bch":
            self._load(filepath)
        else:
            raise Exception("""Unknown file type, only settings (.py) and
benchmaks (.bch) are allowed.""")

    def _make_name(self):
        r"Creates the name of the benchmark."
        return "bench_%s_P%sMC%s_%s_%s.bch" % (self._bench_variable,
            self._num_points, self._num_repet,
            datetime.datetime.now().strftime("%d%b%y_%H%M"),
            self._hash)

    def save(self, save_relative_directory=""):
        r"""
        Saves this benchmark instance to a given folder.
        File name is autogenerated.

        Inputs
        ------
        - save_relative_directory (default: ""): str
            Relative path to the directory where to save the data.
        """
        name = save_relative_directory
        name += ("" if name[-1] == "/" else "/") + self.name
        sfile = open(os.getcwd() + "/" + name, 'w')
        print("Saving in %s." % name)
        sfile.write(cPickle.dumps(self.__dict__))
        sfile.close()

    def _load(self, filepath):
        r"Loads a benchmark file (.bch)."
        sfile = open(filepath, 'r')
        data_pickle = sfile.read()
        sfile.close()
        self.__dict__ = cPickle.loads(data_pickle)

    def _check_bench_variable(self, settings):
        r"""
        Looks for a tuple within the variables. Such a variable is the bench
        variable (each computation will be done considering one element of the
        tuple).

        Input
        -----
        settings: dict
            dictionary of settings (either algorithm or data settings).
        """
        for key in settings:
            value = settings[key]
            if isinstance(value, tuple):
                if self._bench_variable is None:
                    self._bench_variable = key
                    self._bench_values = value
                    self._num_points = len(value)
                else:
                    raise Exception("""Only one bench variable is allowed.
(bench are defined by tuples)""")

    def _extract_value(self, settings, settings_name, key):
        r"Extracts a given value from a settings dictionary."
        try:
            return settings[key]
        except:
            raise Exception("Variable %s should contain key value %s." %
                            (settings_name, key))

    def run(self, save_relative_directory=""):
        r"""
        Run the benchmark and save it to the given directory.

        Inputs
        ------
        - save_relative_directory (default: ""): str
            Relative path to the directory where to save the data.
        """
        offrand = 0
        exec self._file_string
        num_criteria = len(self._criteria)
        num_algo = len(algorithms)
        if not self.results:
            self.results = np.nan * np.ones((self._num_points,
                                              num_algo,
                                              num_criteria,
                                              len(self._type)))
        # check if drawing works correctly
        if "display" in processing_settings:
            self.display(processing_settings["display"],
                         self.name + " display")
        # timer
        beginning_time = time.time()
        total_iterations = self._num_points * self._num_repet
        current_iteration = 0
        for point in range(0, self._num_points):

            current_data_settings = self._extract_point_settings(data_settings,
                                                                 point)
            current_algorithm_settings = self._extract_point_settings(\
                algorithm_settings, point)
            # launch the Monte-Carlo sampling
            repet_results = np.nan * np.ones((self._num_repet,
                                              num_algo,
                                              num_criteria))
            for repet in range(0, self._num_repet):
                current_iteration += 1
                time_string = ""
                if 1 < current_iteration:
                    laps = float(time.time() - beginning_time) /\
                        (current_iteration - 1)
                    time_string = self._seconds_to_string((total_iterations -
                        current_iteration + 1) * laps)
                print('Repetition %s/%s of point %s/%s%s:' %
                      (repet + 1, self._num_repet,
                       point + 1, self._num_points,
                       time_string))
                # make it random
                seed = offrand + repet + 10**6 * point
                np.random.seed(seed)
                # create the data
                reference = processing_settings["data_generator"](\
                    current_data_settings)
                # inputs
                parameters = {"data": reference['data'] + reference['noise'],
                              "reference": reference,
                              "rank": current_data_settings["rank"]}
                parameters.update(current_algorithm_settings)
                # potentially add more inputs
                if "additional_inputs" in processing_settings.keys():
                    parameters.update(processing_settings["additional_inputs"](
                        current_data_settings, reference))
                #plt.plot(parameters['data'].T)
                #QApplication.processEvents()
                # start algoritms
                for algo_num in range(0, num_algo):
                    algo = self._algorithm_names[algo_num]
                    sys.stdout.write("%s%s" % (algo, ", "
                                     if algo_num < num_algo - 1 else ".\n"))
                    # use same initialization
                    seed = 10**9 + 10**4 * point + repet
                    np.random.seed(seed)
                    result = algorithms[algo].run(parameters)
                    # save the results
                    criteria = bss_eval(result, reference)[0]
                    for crit in range(0, len(self._criteria)):
                        repet_results[repet, algo_num, crit] =\
                            criteria[self._criteria[crit]]
                    # need refreshing (to avoid stuck plots) ?
                    if "display" in processing_settings:
                        collect()
                        QApplication.processEvents()
            # repet_results
            # compute and store the mean
            self.results[point, :, :, 0] = np.mean(repet_results, axis=0)
            # compute and store the standard deviation
            self.results[point, :, :, 1] = np.std(repet_results, axis=0)
            # compute the median
            self.results[point, :, :, 2] = np.median(repet_results, axis=0)
            # check if drawing works correctly
            if "display" in processing_settings:
                self.display(processing_settings["display"],
                             self.name + " display")
                QApplication.processEvents()
        # end time
        self.end_time = datetime.datetime.now().strftime("%d%b%y_%H%M")
        self.save(save_relative_directory)


    def _extract_point_settings(self, settings, num):
        r"Extract the settings for a given point of the benchmark."
        if self._bench_variable in settings.keys():
            values = settings[self._bench_variable]
            if isinstance(values, tuple):
                settings_copy = settings.copy()
                settings_copy[self._bench_variable] =\
                    settings_copy[self._bench_variable][num]
                return settings_copy
        return settings

    def display(self, options=None, name=None, **kargs):
        r"""
        Displays the computed benchmark.

        Inputs
        ------
        - options (default: None): dict
            Options dictionary, with potential keywords provided below.
        - name (default: None): str
            name of the figure to plot.
        - any keyword argument from the ones listed below.

        Keyword parameters (optional)
        ------------------
        - linestyles (optional): list
            Styles of the lines for each algorithm.
        - markers (optional): list
            Type of marker for each algorithm.
        - colors (optional): list
            Color for each algorithm.
        - fontsize (default: 14): int
            Size of the writings.
        - plottype (default: "mean"): str
            Type of plot among: mean, std, median or mean-std (errorbar plot).
        - criterion (optional): str
            Criterion to be ploted, among the ones computed.
        """
        # made with the help of:
        # http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
        # create default values
        first_criterion = self._criteria[0]
        # default parameters
        def_line_styles = ['-', '-.', '--', ':']
        def_markers = ['D', 's', 'x', '^', 'd', 'h', '+', '*',
                      'o', '.', '1', 'p', '3', '2', '4', 'H',
                      'v', '8', '<', '>']
        def_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        _default_keyword_parameters = {"criterion": first_criterion,
             "fontsize": 14,
             "linestyles": def_line_styles,
             "markers": def_markers,
             "colors": def_colors,
             "plottype": "mean"}  # can be mean, mean-std, std, median
        # update with provided options input
        if options is None:
            options = {}
        _default_keyword_parameters.update(options)
        _default_keyword_parameters.update(kargs)
        # make shorter handles
        criterion = _default_keyword_parameters["criterion"]
        markers = _default_keyword_parameters["markers"]
        fontsize = _default_keyword_parameters["fontsize"]
        linestyles = _default_keyword_parameters["linestyles"]
        colors = _default_keyword_parameters["colors"]
        plottype = _default_keyword_parameters["plottype"]
        linewidth = 2
        markersize = 12
        markeredgewidth = 1
        # other shorter handles
        coordinates = self._bench_values
        num_algo = len(self._algorithm_names)
        criterion_num = self._criteria.index(criterion)
        # create figure and set size
        if name is not None:
            plt.figure(name, figsize=(12, 14))
        else:
            plt.figure(figsize=(12, 14))
        plt.clf()  # close()
        collect()
        # Remove the plot frame lines. They are unnecessary chartjunk.
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Ensure that the axis ticks only show up on the bottom and
        # left of the plot.
        # Ticks on the right and top of the plot are generally unnecessary
        # chartjunk.
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # make them point outward and add minor ticks
        plt.minorticks_on()
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', direction='out')
        ax.tick_params(axis='x', which="minor", direction='out')
        ax.tick_params(axis='y', which="minor", direction='out')
        ax.grid('on', axis='y')
        # ranges
        m = min(coordinates)
        M = max(coordinates)
        border = 0.02
        plt.xlim(m - border * (M - m), M + border * (M - m))
        #plt.ylim(0, 90)
        #plt.xlim(1968, 2014)
        # Make sure your axis ticks are large enough to be easily read.
        # You don't want your viewers squinting to read your plot.
        #plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # labels
        plt.xlabel(self._bench_variable)
        plt.ylabel(plottype + " " + criterion)
        # draw
        for algo_num in range(0, num_algo):
            color = colors[algo_num % len(colors)]
            linestyle = linestyles[algo_num % len(linestyles)]
            marker = markers[algo_num % len(markers)]
            c_markeredgewidth = markeredgewidth
            c_markersize = markersize
            # deal with too little markers
            if marker in ['x', '+', '1', '3', '2', '4']:
                c_markeredgewidth *= 5
                if marker in ['1', '3', '2', '4']:
                    c_markersize *= 1.5
            if plottype == "mean":
                values = self.results[:, algo_num, criterion_num, 0]
                plt.plot(coordinates, values,
                         linestyle=linestyle, linewidth=linewidth,
                         color=color, marker=marker,
                         markersize=c_markersize,
                         markeredgewidth=c_markeredgewidth,
                         label=self._algorithm_names[algo_num])
            elif plottype == "median":
                values = self.results[:, algo_num, criterion_num, 2]
                plt.plot(coordinates, values,
                         linestyle=linestyle, linewidth=linewidth,
                         color=color, marker=marker,
                         markersize=c_markersize,
                         markeredgewidth=c_markeredgewidth,
                         label=self._algorithm_names[algo_num])
            elif plottype == "std":
                values = self.results[:, algo_num, criterion_num, 1]
                plt.plot(coordinates, values,
                         linestyle=linestyle, linewidth=linewidth,
                         color=color, marker=marker,
                         markersize=c_markersize,
                         markeredgewidth=c_markeredgewidth,
                         label=self._algorithm_names[algo_num])
            elif plottype == "mean-std":
                values = self.results[:, algo_num, criterion_num, 0]
                stds = self.results[:, algo_num, criterion_num, 1]
                plt.errorbar(coordinates, values, stds,
                             linestyle=linestyle, linewidth=linewidth,
                             color=color, marker=marker,
                             markersize=c_markersize,
                             markeredgewidth=c_markeredgewidth,
                             label=self._algorithm_names[algo_num])
            else:
                raise Exception("Unknown plottype.")
        plt.legend()

    def _seconds_to_string(self, seconds):
        r"Converst an amount of second to a string."
        string = ""
        seconds = int(seconds)
        # days
        val = int(seconds / 24 / 60 / 60)
        seconds -= val * 24 * 60 * 60
        string += "" if val == 0 else "%sd" % val
        # hours
        val = int(seconds / 60 / 60)
        seconds -= val * 60 * 60
        string += "" if val == 0 else "%sh" % val
        # minuts
        val = int(seconds / 60)
        seconds -= val * 60
        string += "" if val == 0 else "%smin" % val
        # seconds
        string += "%ss" % seconds
        return " (%s remaining)" % string
