# -\*- coding: utf-8 -\*-
#
# r"""
# tools.py - This file is part of pygmca.
# The pygmca package aims at performing non-negative matrix factorization.
# This module provides processing tools.
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

import numpy as np
from scipy import special


def dim_norm(data, dim=0, norm_type=2):
    r"""
    Computes the norm of X along a given dimension.

    Inputs
    ------
    - data: numpy array
        Data array to be processed.
    - dim (default: 0): int
        Dimension on which to process the data.
    - norm_type (default: 2): int
        Norm type to be used for the computation (norms 1 or 2).
    """

    if norm_type == 2:
        norms = np.sqrt(np.sum(data * data, axis=dim))
    else:
        if norm_type == 1:
            norms = np.sum(np.abs(data), axis=dim)
        else:
            raise Exception("Norm type can be either \"1\" or \"2\" (not \"" +
                            str(norm_type) + "\).")
    shape = np.array(data.shape)
    shape[dim] = 1
    return norms.reshape(shape)


def dim_mad_std(data, dim=None):
    r"""
    Computes the standard deviation of X based on the MAD estimator
    along a given dimension (when dim is None, computes the MAD from
    all the values).

    Inputs
    ------
    - data: numpy array
        Data array to be processed.
    - dim (default: None): int
        Dimension on which to process the data.
    """
    # compute the standard deviation of X based on the MAD estimator
    # along dimension dim
    sizes = np.array(data.shape)
    if dim is None:
        data = data.flatten()
        dim = 0
        sizes = (1,)
    else:
        sizes[dim] = 1
    outvals = np.median(data, axis=dim).reshape(sizes)
    outvals = 1.4826 * np.median(np.abs(data - outvals),
                                 axis=dim).reshape(sizes)

    return outvals


def generate_2D_generalized_gaussian(rows, columns, alpha=2):
    r"""
    from Matlab code BSSGUI by J. Petkov by Jakub Petkov
    adapted to Python by J. Rapin in order to have exact same simulated data
    between Matlab and Python versions

    Generates random variables with generalized Gaussian distribution
    with parameter alpha > 0 and variance 1.
    The generator is only approximate, the generated r.v. are bounded by 1000.

    Method: numerical inversion of the distribution function at points
    uniformly distributed in [0,1].

    Inputs
    ------
    - rows: int
        Number of rows of the data.
    - columns: int
        Number of columns of the data.
    - alpha (default: 2): float
        Generalized Gaussian shape parameter for the coefficients
        distribution.
    """
    m = rows
    n = columns
    r = 0.5 * np.random.random(m * n) + 0.5  # distribution is symmetric
    beta = np.sqrt(special.gamma(3.0 / alpha) /
                   special.gamma(1.0 / alpha))  # enough to consider r > 0.5
    y = r / beta
    ymin = 1e-20 * np.ones(m * n)
    ymax = 1000 * np.ones(m * n)
    # for simplicity, generated r.v. are bounded by 1000.
    for iter in range(0, 33):
        cdf = 0.5 + 0.5 * special.gammainc(1.0 / alpha, (beta * y) ** alpha)
        indplus = np.nonzero(cdf > r)
        if len(indplus) > 0:
            ymax[indplus] = y[indplus]
        indminus = np.nonzero(cdf < r)
        if len(indminus) > 0:
            ymin[indminus] = y[indminus]
        y = 0.5 * (ymax + ymin)
    ind = np.nonzero(np.random.random(m * n) > 0.5)
    if len(ind) > 0:
        y[ind] = -y[ind]
    x = y.reshape([n, m]).T.copy()
    return x


def fill_parameters(parameters, default_parameters,
                    name="unknown function"):
    """
    Function used to fill the inputs with default parameters if required.
    It prints the added default parameters if the verbose parameters is True.

    Inputs
    ------
    - parameters: dict
        Parameters dictionary, with potential keywords provided below.
    - default_parameters: dict
        Dictionary of default parameters.
    - name (default: "unknownfunction"): str
        Nname of the calling function or class (for printing if
        verbose is True.
    """
    # check verbose first
    string = ""
    if 'verbose' not in parameters.keys():
        if 'verbose' not in default_parameters.keys():
            default_parameters["verbose"] = 0
        parameters["verbose"] = default_parameters["verbose"]
    verbose = parameters["verbose"]
    #check each parameter
    for key in default_parameters.keys():
        check = False
        # check if the parameter is not found
        if key not in parameters.keys():
            check = True
        else:
            # or if it is a parameter dictionary
            if ("param" in key) or ("opt" in key):
                if isinstance(parameters[key], dict):
                    check = True
        # if the value requires checking
        if check:
            val = default_parameters[key]
            if val is None:
                raise Exception("In " + name +
                                ", field \"" + key + "\" is required.")
            else:
                if isinstance(val, dict) and (("param" in key) or
                                              ("opt" in key)):
                    # create if it does not exist
                    if key not in parameters.keys():
                        parameters[key] = {}
                    # propagate verbose
                    in_verbose = 0
                    if "verbose" in parameters[key]:
                        in_verbose = parameters[key]["verbose"]
                    parameters[key]["verbose"] = verbose
                    parameters[key] = fill_parameters(parameters[key],
                                                      default_parameters[key],
                                                      name +
                                                      "'s subfield \"%s\"" %
                                                      key)
                    # get back to standard verbose
                    parameters[key]["verbose"] = in_verbose
                else:
                    parameters[key] = val
                    if verbose:
                        string = "In " + name + ", field \"" +\
                            key + "\" set to default value."
                        if type(val) in (tuple, float, bool, int):
                            string = string[:-1] + ": %s." % val
                        else:
                            if isinstance(val, str):
                                string = string[:-1] + ": \"%s\"." % val
                        print(string)
    return parameters


def power_iteration(linear_operator_handle, x0, tolerance=0.0001):
    "Computes the L2 operator norm (largest eigen value) of a linear operator"
    lipsch_old = 0
    lipsch = 1
    x = x0 / np.linalg.norm(x0, ord='fro')
    iteration = 0
    while abs(lipsch_old - lipsch) / lipsch > tolerance:
        x = x / lipsch
        x = linear_operator_handle(x)
        lipsch_old = lipsch
        lipsch = np.linalg.norm(x, ord='fro')
        iteration += 1
    return lipsch * (1 + tolerance)  # better overestimate it


def broacast_to_shape(data, shape):
    r"Create a fake array broacasting an array to another shape."
    strides = list(data.strides)
    num_add_dims = len(shape) - len(strides)
    data_shape = list(data.shape) + [1] * num_add_dims
    strides = strides + [0] * num_add_dims
    for k in range(0, len(shape)):
        if data_shape[k] != shape[k]:
            if data_shape[k] == 1 and shape[k] > 1:
                strides[k] = 0
            else:
                raise Exception("Incompatible shape")

    output = np.lib.stride_tricks.as_strided(data, shape=shape,
                                             strides=tuple(strides))
    return output
