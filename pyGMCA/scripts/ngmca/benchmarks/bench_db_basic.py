# -*- coding: utf-8 -*-
from pyGMCA.bss.ngmca import base as bss

# settings which are passed to the data_generator
# exactly one tuple is allowed between variables data_settings and
# algorithm_settings (below). This tuple contains the values on which to
# perform the benchmark (bench variable)
data_settings = {}
data_settings['rows'] = 100
data_settings['rank'] = 10
data_settings['columns'] = 200
data_settings['dB'] = (10, 20)
data_settings['bernoulli_S'] = 0.08
data_settings['alpha_S'] = 1


# settings which are passed to all the algorithms
# exactly one tuple is allowed between variables data_settings and
# algorithm_settings (below). This tuple contains the values on which to
# perform the benchmark
algorithm_settings = {"number_of_iterations": 10}

# dictionary of algorithms to be used for the benchmark
algorithms = {"nGMCA": bss.algos.Ngmca()}

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
processing_settings = {"number_of_repetitions": 10,
                       "criteria": ["SDR_S", "SIR_S", "SNR_S", "SAR_S",
                                    "SDR_A", "SIR_A", "SNR_A", "SAR_A"],
                       "data_generator": bss.tools.create_sparse_data,
                       "display": {}}
