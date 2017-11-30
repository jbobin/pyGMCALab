This file contains the python implementations of the algorithms used in the numerical experiments section in the report ‘Unsupervised separation of sparse sources in the presence of outliers’, C.Chenot and J.Bobin, namely: GMCA, AMCA, rGMCA, rAMCA, the minimization of the beta-divergence and the combination PCP+GMCA.

The files are organized as follows:
- the file ‘main.py’ in which the benchmark (data generation, calls of the algorithms, save and plot of the results) is performed.
- ‘parameters.py’: in which the parameters (for the data generation, benchmark, algorithms) are set.
- ‘data.py’: called to generate the data
- ‘AGMCA.py’: algorithms AMCA, GMCA, rAMCA, and rGMCA
- ‘BetaD_ICA.py’: minimization of the beta-divergence (estimation of S updated the 14/06/16).
- ‘FunBetaD.py’: functions called when minimizing the beta-divergence, used by ‘BetaD_ICA.py’
- ‘rpcaAlgo.py’: combination PCP+GMCA
- ‘errors.py’: to compute the errors made on the estimation of the mixing matrix (the package munkres is needed).

The references to the corresponding articles or previous implementations of these algorithms are presented directly in the python files.


By running the file ‘main.py’ provided, a comparison (for 1run) of these different algorithms is performed, for varying percentage of corrupted samples (9 different percentages), in the presence of 8 sources and 16 measurements of 4096 samples. The results are displayed (but not saved).

First version: 10/06/16.
Second Version 14/06/16. (Estimation of S for the minimization of the beta-divergence updated).
