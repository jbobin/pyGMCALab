This file contains a python implementation of some algorithms used in the numerical experiments section in the report ‘Blind Source Separation with outliers in transformed domains’, C.Chenot and J.Bobin, namely: GMCA, tr-rGMCA, MCA, and Outlier Pursuit.

 The files are organized as follows:
- the file ‘main.py’ in which the benchmark (data generation, calls of the algorithms, plot of the results) is performed.
- ‘parameters.py’: in which the parameters (for the data generation, benchmark, algorithms) are set.
- ‘data.py’: called to generate the data
- ‘XMCA.py’: algorithms AMCA/GMCA (Sparsity and Adaptivity for the Blind Separation of Partially Correlated Sources;J.Bobin et al), MCA (Simultaneous cartoon and texture image inpainting using morphological component analysis (MCA); M.Elad et al.)and tr-rGMCA
- ‘outliersPursuit.py’: Outliers Pursuit (Robust PCA via Outlier Pursuit; H.Xu et al)
- ‘errors.py’: to compute the errors made on the estimation of the mixing matrix (the package munkres is needed).

The references to the corresponding articles or previous implementations of these algorithms are presented directly in the python files.


By running the file ‘main.py’ provided, a comparison (for 1run) of these different algorithms is performed, for varying number of observations, sources sparsely represented in DCT and outliers in the direct domain.