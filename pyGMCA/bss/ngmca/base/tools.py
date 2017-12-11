# # -\*- coding: utf-8 -\*-
#
# r"""
# tools.py - This file is part of pygmca.
# The pygmca package aims at performing non-negative matrix factorization.
# This module provides a class allowing to easily compute and display benchmarks.
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
import matplotlib.pyplot as plt
import os
import scipy.ndimage
from pyGMCA.bss.ngmca import core, proximal
from pyGMCA.bss.ngmca.munkres import munkres
from scipy import linalg, misc
from pyGMCA.bss.ngmca.base import sample_nmr_spectra

# import pdb

class Factorization(object):
    r"""
    Class with fields A and S representing the
    factorization A.dot(S)
    """

    def __init__(self, shape, rank, random_initialization=True):
        r"""
        Initialize a factorization of given shape and rank.

        Inputs
        ------
        - shape: tuple or list
            Factorization shape.
        - rank: int
            Rank of the factorization (number of source signals to recover).
        - random_initialization (default: True): bool
            Initialization of the factorization with random values.
        """
        if random_initialization:
            self.A = np.random.random([shape[0],
                                       rank]).reshape(rank, shape[0]).T
            self.S = np.random.random([rank,
                                       shape[1]]).reshape(shape[1], rank).T
        else:
            self.A = np.zeros([shape[0], rank])
            self.S = np.zeros([rank, shape[1]])

    def set_scale_on(self, which):
        r"""
        Set the scale of each source on either A or S.
        Inputs
        ------
        - which: str
            string taking value either "A" or "S" depending on whether
            to process A or S.
        """
        n_A = core.tools.dim_norm(self.A, 0).flatten()
        n_S = core.tools.dim_norm(self.S, 1).flatten()
        active = (n_A * n_S) > 0
        self.A[:, active == False] = 0
        self.S[active == False, :] = 0
        if sum(active > 0):
            if which == 'A':
                self.A[:, active] = self.A[:, active] * n_S[active]
                self.S[active, :] = self.S[active, :] /\
                    n_S[active].reshape(sum(active), 1)
            else:
                if which == 'S':
                    self.A[:, active] = self.A[:, active] / n_A[active]
                    self.S[active, :] = self.S[active, :] *\
                        n_A[active].reshape(sum(active), 1)
                else:
                    raise Exception(
                        "\"which\" must be either \"A\" or \"S\" (not \"" +
                        which + "\).")


def create_sparse_data(parameters=None, **kargs):
    r"""
    Synthesize mixtures of sparse sources.

    Inputs
    ------
    - parameters (default: None): dict
        Parameters dictionary, with potential keywords provided below.
    - any keyword argument from the ones listed below.

    Keyword parameters (required)
    ------------------
    - rows (required): int
        Number of rows of the data.
    - rank (required): int
        Rank of the factorization (number of source signals to recover).
    - columns (required): int
        Number of columns of the data.

    Keyword parameters (optional)
    ------------------
    - bernoulli_S (default: 0.1): float
        Bernoulli parameter for the activation of the coefficients of S
        (between 0 and 1).
    - alpha_A (default: 2): float
        Generalized Gaussian shape parameter for the distribution of the
        coefficients of A (>0).
    - dB (default: 20): float
        Additive Gaussian noise SNR in dB.
    - multiplicative_std (default: 0): float
        Multiplicate noise level on a variable X, so that its standard deviation
        is given by multiplicative_std * X.
    - alpha_S (default: 2): float
        Generalized Gaussian shape parameter for the distribution of the
        coefficients of S (>0).
    - bernoulli_A (default: 1): float
        Bernoulli parameter for the activation of the coefficients of A
        (between 0 and 1).
    """
    _default_keyword_parameters = {"bernoulli_A": 1,
                                   "bernoulli_S": 0.1,
                                   "alpha_A": 2,
                                   "alpha_S": 2,
                                   "rows": None,
                                   "columns": None,
                                   "rank": None,
                                   "dB": 20,
                                   "multiplicative_std": 0}
    if not parameters:
        parameters = {}
    parameters.update(kargs)
    parameters = core.tools.fill_parameters(parameters,
                                             _default_keyword_parameters,
                                             "create_sparse_data")
    # create the reference structure
    rows = parameters["rows"]
    columns = parameters["columns"]
    rank = parameters["rank"]
    decibel = parameters["dB"]
    alpha_a = parameters["alpha_A"]
    alpha_s = parameters["alpha_S"]
    bern_a = parameters["bernoulli_A"]
    bern_s = parameters["bernoulli_S"]
    mult_std = parameters["multiplicative_std"]
    reference = {}
    reference['factorization'] = Factorization([rows, columns], rank, 0)
    reference['factorization'].A = np.abs(
        core.tools.generate_2D_generalized_gaussian(rows, rank, alpha_a)) *\
        ((np.random.rand(rows, rank) < bern_a).reshape(rank, rows).T)
    reference['factorization'].S = np.abs(
        core.tools.generate_2D_generalized_gaussian(rank, columns,
                                                     alpha_s)) *\
        ((np.random.rand(rank, columns) < bern_s).reshape(columns, rank).T)
    return make_reference_data_and_noise(reference, decibel, mult_std)

def create_realistic_nmr_mixtures(parameters=None, **kargs):
    r"""
    Synthesize mixtures of NMR spectra convolved with a Laplacian filter.

    Inputs
    ------
    - parameters (default: None): dict
        Parameters dictionary, with potential keywords provided below.
    - any keyword argument from the ones listed below.

    Keyword parameters (required)
    ------------------
    - rows (required): int
        Number of rows of the data.
    - rank (required): int
        Rank of the factorization (number of source signals to recover).

    Keyword parameters (optional)
    ------------------
    - alpha_A (default: 2): float
        Generalized Gaussian shape parameter for the distribution of the
        coefficients of A (>0).
    - dB (default: 20): float
        Additive Gaussian noise SNR in dB.
    - width (default: 4): float
        Width of the Laplacian filter.
    - bernoulli_A (default: 1): float
        Bernoulli parameter for the activation of the coefficients of A
        (between 0 and 1).
    - multiplicative_std (default: 0): float
        Multiplicate noise level on a variable X, so that its standard deviation
        is given by multiplicative_std * X.
    """
    _default_keyword_parameters = {"bernoulli_A": 1,
                                   "alpha_A": 2,
                                   "rows": None,
                                   "rank": None,
                                   "dB": 20,
                                   "width": 4,
                                   "multiplicative_std": 0}
    if not parameters:
        parameters = {}
    parameters.update(kargs)
    parameters = core.tools.fill_parameters(parameters,
        _default_keyword_parameters, "create_realistic_nmr_mixtures")
    rows = parameters["rows"]
    columns = 2048
    width = parameters["width"]
    rank = parameters["rank"]
    decibel = parameters["dB"]
    alpha_a = parameters["alpha_A"]
    bern_a = parameters["bernoulli_A"]
    mult_std = parameters["multiplicative_std"]
    # create A
    reference = {}
    reference['factorization'] = Factorization([rows, columns], rank, 0)
    reference['factorization'].A = np.abs(
        core.tools.generate_2D_generalized_gaussian(rows, rank, alpha_a)) *\
        ((np.random.rand(rows, rank) < bern_a).reshape(rank, rows).T)
    # create S
    names = sample_nmr_spectra.peak_list.keys()
    if rank > len(names):
        raise Exception("Not enough sample nmr spectra, please use \
a rank strictly smaller than %s." % len(sample_nmr_spectra.peak_list) + 1)
    # extract "rank" sources
    perm = np.random.permutation(len(names))
    names = [names[i] for i in perm[range(0, rank)]]
    reference["names"] = names
    (spectrum, ppm) = sample_nmr_spectra.get_nmr_spectrum(
        sample_nmr_spectra.peak_list[names[0]], (0, 10), columns, width)
    reference['factorization'].S[0, :] = spectrum
    reference['ppm'] = ppm
    for k in range(1, rank):
        reference['factorization'].S[k, :] =\
            sample_nmr_spectra.get_nmr_spectrum(
            sample_nmr_spectra.peak_list[names[k]],
            (0, 10), columns, width)[0]
    return make_reference_data_and_noise(reference, decibel, mult_std)


def make_reference_data_and_noise(reference, dB, multiplicative_std=0):
    r"""
    Adds the noise and data to the reference data dictionary in which the
    factorization is provided.

    Inputs
    ------
    - reference: dict
        dictionary containing the reference factorization as well as the noise and
        other relevant information if need be.
    - dB: float
        Additive Gaussian noise SNR in dB.
    - multiplicative_std (default: 0): float
        Multiplicate noise level on a variable X, so that its standard deviation
        is given by multiplicative_std * X.
    """
    # data
    reference['data'] = reference['factorization'].A.dot(
        reference['factorization'].S)
    # normalize so as to have std(Y_ij) = 1
    (rows, columns) = reference['data'].shape
    reference['data'] = reference['factorization'].A.dot(
        reference['factorization'].S)
    coeff = np.sqrt(rows * columns) / np.linalg.norm(reference['data'], 'fro')
    reference['data'] = reference['data'] * coeff
    reference['factorization'].A = reference['factorization'].A * coeff
    # noise
    if np.isinf(dB):
        noise = 0
    else:
        noise = 10 ** (-dB / 20.0) / np.sqrt(rows * columns) *\
            np.linalg.norm(reference['data'], 'fro')
    reference['noise'] =\
        core.tools.generate_2D_generalized_gaussian(rows, columns, 2) * noise
    reference['factorization'].set_scale_on('S')
    # record the values of noise std, for algorithm inputs
    reference['additive_std'] = noise
    reference['multiplicative_std'] = multiplicative_std
    if multiplicative_std > 0:
        reference['additive_noise'] = reference['noise']
        reference['multiplicate_noise'] =\
            reference['data'] * multiplicative_std *\
            core.tools.generate_2D_generalized_gaussian(rows, columns, 2)
        reference['noise'] = reference['additive_noise'] +\
            reference['multiplicate_noise']
    return reference


def reinitialize_null_sources(factorization, data, verbose=0):
    r"""
    Fast reinitialization of null sources in the factorization
    by picking one column in the residue.

    Inputs
    ------
    - factorization: Factorization instance
        Current factorization of the data.
    - data: numpy array
        Data array to be processed.
    - verbose (default: 0): bool
        Display important parameters

    Output
    ------
    Same factorization with the reinitialized sources.
    """
    indices = np.where((core.tools.dim_norm(factorization.A, 0) *
                        core.tools.dim_norm(factorization.S, 1).T) == 0)[0]
    if (len(indices) > 0):
        if verbose:
            print("Reinitialization of " +
                  str(len(indices)) + " null sources.\n")
        for k in indices:
            # compute residual
            R = np.maximum(data - factorization.A.dot(factorization.S), 0)
            # compute square norm of residual to select maximum one
            res2 = np.sum(R * R, 0)
            j = np.where(res2 == np.max(res2))[0][0]
            if res2[j] > 0:
                factorization.A[:, k] = R[:, j] / np.sqrt(res2[j])
                # compute scalar product with the rest od the residual and
                # keep only positive coefficients
                factorization.S[k, :] = np.maximum(factorization.A[:, k].T.dot(R), 0)
    return factorization


def warm_initialization(data, rank):
    r"""
    Initialization function alternating between ALS updates and constrained
    updates.

    Inputs
    ------
    - data: numpy array
        Data array to be processed.
    - rank: int
        Rank of the factorization (number of source signals to recover).
    """
    F = Factorization(data.shape, rank)
    options = {}
    options['maximum_iteration'] = 50
    for k in range(0, 2):
        # unconstrained least square
        ind = core.tools.dim_norm(F.A, 0).flatten() > 0
        F.S[ind, :] = np.maximum(np.linalg.lstsq(F.A[:, ind], data)[0], 0)
        ind = core.tools.dim_norm(F.S, 1).flatten() > 0
        F.A[:, ind] = np.maximum(np.linalg.lstsq(F.S[ind, :].T, data.T)[0].T, 0)
        # constrained least square
        AtY = F.A.T.dot(data)
        AtA = F.A.T.dot(F.A)
        F.S = nonnegative_sparse_inversion(AtY, AtA, F.S, 0, options)
        F = reinitialize_null_sources(F, data, 0)
        AtY = F.S.dot(data.T)
        AtA = F.S.dot(F.S.T)
        F.A = nonnegative_sparse_inversion(AtY, AtA, F.A.T, 0, options).T
        F = reinitialize_null_sources(F, data, 0)
    return F


def nonnegative_sparse_inversion(AtY, AtA, S0, mu, options):
    r"""
    "Solves problem:
    argmin_{S >= 0} ||Y - A S||_2^2 + ||lambda .* S||_1
    using FISTA (Beck & Teboulle 2009).

    Output
    ------
        S: solution of the problem.

    Inputs
    ------
    - AtY: numpy array
        Product between transpose of A and Y.
    - AtA: numpy array
        Product between transpose of A and A.
    - S0: numpy array
        Initialization matrix for S
    - mu: float or numpy array
        Value of the sparsity parameter mu
    - options: dict
        Options dictionary, with potential keywords provided below.

    Keyword parameters (optional)
    ------------------
    - hardthresholding (default: 0): bool
        Use hard-thresholding instead of soft-thresholding.
    - norm_constrained (default: 0): bool
        Use a non-increasing norm constraint.
    - reweighted_l1 (default: 0): float
        No reweighting if 0, else the reweighting is computing using
        a charactheristic value of reweighted_l1 * noise_estimation.
    - relative_difference_tolerance (default: 1e-05): float
        Relative difference tolerance between two consecutive iterates
        for stopping the algorithm (not activated if 0).
    - maximum_iteration (default: 100): int
        Maximum number of steps to perform.

    Note
    ----

    .. math::
        \text{argmin}_{\mathbf{S}\ge 0} \frac{1}{2}\|\mathbf{Y} - \mathbf{A} \mathbf{S}\|_2^2 + \|\mathbf{\mu} \odot \mathbf{S}||_1

    Requires the precomputation of :math:`\mathbf{A}^T\mathbf{Y}` and :math:`\mathbf{A}^T\mathbf{A}`.
    """
    _default_keyword_parameters = {"maximum_iteration": 100,
                                   "norm_constrained": 0,
                                   "hardthresholding": 0,
                                   "reweighted_l1": 0,
                                   'relative_difference_tolerance': 0.00001}
    options = core.tools.fill_parameters(options, _default_keyword_parameters,
                                          "nonnegative_sparse_inversion")
    # reweighted L1 (not tested, beware reference modification)
    if (options["reweighted_l1"] > 0) and (np.max(mu) > 0):
        S_inv = np.linalg.lstsq(AtA, AtY)[0]
        noise_estimate = (options["reweighted_l1"] *
                          core.tools.dim_mad_std(S_inv, 1))
        mu = mu / (1.0 + (abs(S_inv) / noise_estimate)**2)
    # proximal operator
    if options["hardthresholding"]:
        prox = proximal.operators.nonnegative_hard_thresholding
    else:
        if options["norm_constrained"]:
            n_S = core.tools.dim_norm(S0, 1)

            def prox(z, threshold):
                r"Proximal operator handle."
                return proximal.operators.nonnegative_soft_thresholding(
                    proximal.operators.norm_projection(z, n_S), threshold)
        else:
            prox = proximal.operators.nonnegative_soft_thresholding
    rel_tol = options['relative_difference_tolerance']
    parameters = {'gradient': (lambda z: AtA.dot(z) - AtY),
                  'proximal': prox,
                  'lambda': mu,
                  'lipschitz_constant': np.linalg.norm(AtA, 2),
                  'maximum_iteration': options['maximum_iteration'],
                  'initialization': S0.copy(),
                  'relative_difference_tolerance': rel_tol}
    FB = proximal.algorithms.ForwardBackward()
    res = FB.run(parameters)
    return res["x"]


def evaluation(result, reference, verbose=0):
    r"""
    Evaluate BSS results using criteria from Vincent et al.
    This function reorders the sources and mixtures so as to match
    the reference factorization.

    Inputs
    ------
    - result: dict
        output of a BSS algorithm, with field "factorization".
    - reference: dict
        dictionary containing the reference factorization as well as the noise and
        other relevant information if need be.
    - verbose (default: 0): bool
        Display important parameters

    Outputs
    ------
    criteria: dict
        value of each of the criteria.
    decomposition: dict
        decomposition of the estimated sources under target, interference,
        noise and artifacts.
    """
    factorization = result['factorization']
    # get column signals
    Ar = reference['factorization'].A.copy()
    Sr = reference['factorization'].S.T.copy()
    Se = factorization.S.T.copy()
    r = Sr.shape[1]
    # set nan values to 0
    Ar[np.isnan(Ar)] = 0
    Sr[np.isnan(Sr)] = 0
    Se[np.isnan(Se)] = 0
    # precomputation
    SDR_S = compute_sdr_matrix(Sr, Se)
    # order computation
    costMatrix = -SDR_S
    hungarian = munkres.Munkres()
    ind_list = hungarian.compute(costMatrix.tolist())
    indices = np.zeros(r, dtype=int)
    for k in range(0, r):
        indices[k] = ind_list[k][1]
    # reorder the factorization
    result['factorization'].A = result['factorization'].A[:, indices]
    result['factorization'].S = result['factorization'].S[indices, :]
    # get reordered results
    Ae = result['factorization'].A.copy()
    Ae[np.isnan(Ae)] = 0
    Se = result['factorization'].S.T.copy()
    Se[np.isnan(Se)] = 0
    # compute criteria
    criteria = {}
    # on S
    output = decomposition_criteria(Se, Sr, reference['noise'].T)
    decomposition = output[1]
    criteria['SDR_S'] = output[0]['SDR']
    criteria['SIR_S'] = output[0]['SIR']
    criteria['SNR_S'] = output[0]['SNR']
    criteria['SAR_S'] = output[0]['SAR']
    # on A
    output = decomposition_criteria(Ae, Ar, reference['noise'])
    criteria['SDR_A'] = output[0]['SDR']
    criteria['SIR_A'] = output[0]['SIR']
    criteria['SNR_A'] = output[0]['SNR']
    criteria['SAR_A'] = output[0]['SAR']
    if verbose != 0:
        print("Results of the reconstruction:")
        print("Decomposition criteria on S:")
        print("    - Mean SDR: " + str(criteria['SDR_S']) + ".")
        print("    - Mean SIR: " + str(criteria['SIR_S']) + ".")
        print("    - Mean SNR: " + str(criteria['SNR_S']) + ".")
        print("    - Mean SAR: " + str(criteria['SAR_S']) + ".")
        print("Decomposition criteria on A:")
        print("    - Mean SDR: " + str(criteria['SDR_A']) + ".")
        print("    - Mean SIR: " + str(criteria['SIR_A']) + ".")
        print("    - Mean SNR: " + str(criteria['SNR_A']) + ".")
        print("    - Mean SAR: " + str(criteria['SAR_A']) + ".")
    return (criteria, decomposition)


def compute_sdr_matrix(X, Y):
    r"""
    Computes the SDR of each couple reference/estimate sources.

    Inputs
    ------
    X: numpy array
        reference of column signals.
    Y: numpy array
        estimate of column signals.

    Output
    ------
    MSDR: numpy array
        numpy array such that MSDR(i,j) is the SNR between the i-th row of
        X with the j-th column of Y.
    """
    # normalize the reference
    X = X / core.tools.dim_norm(X, 0)
    # get shape and initialize
    n_x = X.shape[1]
    n_y = Y.shape[1]
    L = X.shape[0]
    MSDR = np.zeros([n_x, n_y])
    # computation
    for n in range(0, n_x):
        targets = X[:, n].reshape([L, 1]) * (X[:, n].T.dot(Y))
        diff = Y - targets
        norm_diff_2 = np.maximum(np.sum(diff * diff, 0), np.spacing(1))
        norm_targets_2 = np.maximum(
            np.sum(targets * targets, 0), np.spacing(1))
        MSDR[n, :] = -10 * np.log10(norm_diff_2 / norm_targets_2)
    return MSDR


def decomposition_criteria(Se, Sr, noise):
    r"""
    Computes the SDR of each couple reference/estimate sources.

    Inputs
    ------
    Se: numpy array
        estimateof column signals.
    Sr: numpy array
        reference  of column signals.
    noise: numpy array
        noise matrix cotaminating the data.

    Outputs
    ------
    criteria: dict
        value of each of the criteria.
    decomposition: dict
        decomposition of the estimated sources under target, interference,
        noise and artifacts.
    """
    # compute projections
    Sr = Sr / core.tools.dim_norm(Sr, 0)
    pS = Sr.dot(np.linalg.lstsq(Sr, Se)[0])
    SN = np.hstack([Sr, noise])
    #pSN = SN.dot(np.linalg.lstsq(SN, Se)[0])  # this crashes on MAC
    pSN = SN.dot(linalg.lstsq(SN, Se)[0])
    eps = np.spacing(1)
    # compute decompositions
    decomposition = {}
    # targets
    decomposition['target'] = np.sum(Se * Sr, 0) * Sr   # Sr is normalized
    # interferences
    decomposition['interferences'] = pS - decomposition['target']
    # noise
    decomposition['noise'] = pSN - pS
    # artifacts
    decomposition['artifacts'] = Se - pSN
    # compute criteria
    criteria = {}
    # SDR: source to distortion ratio
    num = decomposition['target']
    den = decomposition['interferences'] +\
        (decomposition['noise'] + decomposition['artifacts'])
    norm_num_2 = np.sum(num * num, 0)
    norm_den_2 = np.sum(den * den, 0)
    criteria['SDR'] = np.mean(10 * np.log10(
        np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    criteria['SDR median'] = np.median(10 * np.log10(
        np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    # SIR: source to interferences ratio
    num = decomposition['target']
    den = decomposition['interferences']
    norm_num_2 = sum(num * num, 0)
    norm_den_2 = sum(den * den, 0)
    criteria['SIR'] = np.mean(10 * np.log10(
        np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    # SNR: source to noise ratio
    if np.max(np.abs(noise)) > 0:  # only if there is noise
        num = decomposition['target'] + decomposition['interferences']
        den = decomposition['noise']
        norm_num_2 = sum(num * num, 0)
        norm_den_2 = sum(den * den, 0)
        criteria['SNR'] = np.mean(10 * np.log10(
            np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    else:
        criteria['SNR'] = np.inf
    # SAR: sources to artifacts ratio
    if (noise.shape[1] + Sr.shape[1] < Sr.shape[0]):
        # if noise + sources form a basis, there is no "artifacts"
        num = decomposition['target'] +\
            (decomposition['interferences'] + decomposition['noise'])
        den = decomposition['artifacts']
        norm_num_2 = sum(num * num, 0)
        norm_den_2 = sum(den * den, 0)
        criteria['SAR'] = np.mean(10 * np.log10(
            np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    else:
        criteria['SAR'] = np.inf
    return (criteria, decomposition)


def linear_decrease(remaining_iterations, current_value, aimed_value):
    r"""
    Gets a linear decrease between a current value and a goal value
    considering a given number of remaining iterations.
    """
    if remaining_iterations > 0:
        return np.maximum(aimed_value, current_value -
                          (1.0 / (remaining_iterations + 1)) *
                          (current_value - aimed_value))
    else:
        return aimed_value


def load_images(smaller_shape=True):
    r"""
    Load 4 images stacked into a 3D array along the last dimension.

    Inputs
    ------
    - smaller_shape (default: True): bool
        Use 128x128 images (instead of 256x256).
    """
    imfolder = os.path.dirname(__file__) + '/images/'
    lena = misc.imread(imfolder + 'lena256.png').astype("float64")
    if smaller_shape:
        lena = scipy.ndimage.interpolation.zoom(lena, 0.5)
    lena /= np.max(lena)
    peppers = misc.imread(imfolder + 'peppers256.png').astype("float64")
    if smaller_shape:
        peppers = scipy.ndimage.interpolation.zoom(peppers, 0.5)
    peppers /= np.max(peppers)
    boat = misc.imread(imfolder + 'boat256.png').astype("float64")
    if smaller_shape:
        boat = scipy.ndimage.interpolation.zoom(boat, 0.5)
    boat /= np.max(boat)
    barbara = misc.imread(imfolder + 'barbara256.png').astype("float64")
    if smaller_shape:
        barbara = scipy.ndimage.interpolation.zoom(barbara, 0.5)
    barbara /= np.max(barbara)
    images = np.concatenate((lena[..., np.newaxis], peppers[..., np.newaxis],
                             barbara[..., np.newaxis], boat[..., np.newaxis]),
                            axis=2).transpose([2, 0, 1])
    return images


def show_images(images, stack_dim=0, shape="optional",
                indices="all"):
    r"""
    Show images which are stacked into an array along a given dimension.

    Inputs
    ------
    - images: numpy array
        Array of images.
    - stack_dim (default: 0): int
        Dimension along which the images are stacked.
    - shape (default: "optional"): tuple
        Shape of the images (required if they are vectorized).
    - indices (default: "all"): list of int
        List of images to draw.
    """
    num_images = images.shape[stack_dim]
    if indices == "all":
        indices = range(0, num_images)
    first_image = np.squeeze(images.take((indices[0],), axis=stack_dim))
    if shape == "optional":
        if len(first_image.shape) == 2:
            shape = first_image.shape
        else:
            raise Exception("Shape is required when providing vectorized\
images.")
    first_image.reshape(shape)
    # prepare subplot sizes
    num = len(indices)
    num_limit = 100
    if num > num_limit:
        num = num_limit
        print('Limiting to %s images.' % num_limit)
    sx = int(np.sqrt(num) + 0.99999)
    sy = int(num / sx + 0.99999)
    for k in range(0, num):
        plt.subplot(sy, sx, k + 1)
        im = np.squeeze(images.take((indices[k],),
                                    axis=stack_dim)).reshape(shape)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('image #%s' % k)


def create_image_mixtures(parameters=None, **kargs):
    r"""
    Synthesize mixtures of 4 images.

    Inputs
    ------
    - parameters (default: None): dict
        Parameters dictionary, with potential keywords provided below.
    - any keyword argument from the ones listed below.

    Keyword parameters (required)
    ------------------
    - rows (required): int
        Number of rows of the data.

    Keyword parameters (optional)
    ------------------
    - alpha_A (default: 2): float
        Generalized Gaussian shape parameter for the distribution of the
        coefficients of A (>0).
    - dB (default: 20): float
        Additive Gaussian noise SNR in dB.
    - smaller_shape (default: True): bool
        Use 128x128 images (instead of 256x256).
    - bernoulli_A (default: 1): float
        Bernoulli parameter for the activation of the coefficients of A
        (between 0 and 1).
    - multiplicative_std (default: 0): float
        Multiplicate noise level on a variable X, so that its standard deviation
        is given by multiplicative_std * X.
    """
    _default_keyword_parameters = {"bernoulli_A": 1,
                                   "alpha_A": 2,
                                   "rows": None,
                                   "dB": 20,
                                   "multiplicative_std": 0,
                                   "smaller_shape": True}
    if not parameters:
        parameters = {}
    parameters.update(kargs)
    parameters = core.tools.fill_parameters(parameters,
        _default_keyword_parameters, "create_image_mixtures")
    rows = parameters["rows"]
    columns = 256 * 256
    rank = 4
    decibel = parameters["dB"]
    alpha_a = parameters["alpha_A"]
    bern_a = parameters["bernoulli_A"]
    mult_std = parameters["multiplicative_std"]
    # create A
    reference = {}
    reference['factorization'] = Factorization([rows, columns], rank, 0)
    reference['factorization'].A = np.abs(
        core.tools.generate_2D_generalized_gaussian(rows, rank, alpha_a)) *\
        ((np.random.rand(rows, rank) < bern_a).reshape(rank, rows).T)
    # create S
    images = load_images(parameters["smaller_shape"])
    images = images.reshape([images.shape[0], np.prod(images.shape[1:])])
    reference['factorization'].S = images
    return make_reference_data_and_noise(reference, decibel, mult_std)
