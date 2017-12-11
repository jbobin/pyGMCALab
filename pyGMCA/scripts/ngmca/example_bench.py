# -*- coding: utf-8 -*-

from pyGMCA.bss.ngmca.core.benchmark import Benchmark


# load the provided configuration file
#bench = Benchmark("benchmarks/bench_lambda.py")
bench = Benchmark("pyGMCA/scripts/ngmca/benchmarks/bench_db_basic.py")

# compute the benchmark and save it in the "benchmarks" (relative) folder
bench.run("benchmarks")




# if the benchmark is already computed and saved,
# it is possible to reload it by providing the relative path to the
# file.
from pyGMCA.bss.ngmca.core.benchmark import Benchmark
#bench = Benchmark("benchmarks/bench_dB_P2MC10_10nov.14_2200_172631.bch")
bench = Benchmark("pyGMCA/scripts/ngmca/benchmarks/bench_S_tau_mad_P5MC12_04déc14_1042_820818.bch")

# one can then display it with any display settings
bench.display(plottype="mean-std")
