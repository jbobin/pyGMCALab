# -*- coding: utf-8 -*-
import pyGMCA.bss.ngmca.base as bss
import numpy as np


# settings which are passed to the data_generator
# exactly one tuple is allowed between variables data_settings and
# algorithm_settings (below). This tuple contains the values on which to
# perform the benchmark (bench variable)
data_settings = {'rows': 24,
                 'rank': 12,
                 'dB': 20,
                 'S_tau_mad': (0, 0.5, 1, 2, 3)}

# settings which are passed to all the algorithms
# exactly one tuple is allowed between variables data_settings and
# algorithm_settings (below). This tuple contains the values on which to
# perform the benchmark
algorithm_settings = {"maximum_iteration": 200}

# dictionary of algorithms to be used for the benchmark
algorithms = {"nGMCA": bss.algos.Ngmca(),
              "nGMCA_framework": bss.algos.Framework({
                  'A_updater': bss.algos.SparseUpdater(tau_mad=0)})}


def provide_tau_mad(data_settings, reference):
    # set up tau_mad for different algoriths
    tau_mad = data_settings["S_tau_mad"]
    from pyGMCA import bss # not convenient, but works
    return {'S_updater': bss.algos.SparseUpdater(tau_mad=tau_mad),
            'S_parameters': {"tau_mad":tau_mad}}

# settings for the processing of the benchmark, including;
# - the number of repetitions for each point of the bench variable.
# - the criteria to be saved (all the criteria must be computed by
#       the bss.tools.evaluatuation function, to add a criterion, modify
#       this function)
# - data_generator: the function which creates the data
# - display: if the field is present, results will be displayed during the
#       computation. It can take arguments for the display function.
# - additional_inputs: function which returns a dictionary which will be
#       appended to the parameters. This can be useful for providing data
#       dependant settings, or more complex input parameters.
processing_settings = {"number_of_repetitions": 12,
                       "criteria": ["SDR_S", "SIR_S", "SNR_S", "SAR_S",
                                    "SDR_A", "SIR_A", "SNR_A", "SAR_A"],
                       "data_generator": bss.tools.create_realistic_nmr_mixtures,
                       "additional_inputs": provide_tau_mad,
                       "display": {}}
