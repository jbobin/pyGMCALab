<<<<<<< HEAD
# pyGMCALab
Toolbox for solving sparse matrix factorization problems
>>>>>>>

This toolbox is composed of the following submodules:

* GMCA is the building block of the pyGMCALab toolbox. This algorithm basically tackles
sparse BSS problems.

* Building upon GMCA, AMCA is an extension that specifically deals with partially correlated sources

* nGMCA allows solving sparse non-negative matrix factorization problems (sparse NMF). Sparse modelling can be done
either in the sample domain or in a transformed domain. For that purpose, a python/C wrapper for wavelets (RedWave) is also
provided as an external module (see ./redwave_toolbox).

* rGMCA copes with sparse BSS problems in the presence of outliers. Tr_rGMCA is an extension that can benefit from the morphological
diversity between the outliers and the sources.

For all these algorithms, we strongly advise the interested user to have a close look at the jupyter notebooks, which are provided in ./pyGMCA/scripts
