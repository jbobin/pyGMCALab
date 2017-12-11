# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pyGMCA.bss.ngmca import proximal


# this script shows how to use the Forward backward algorithm
# this algorithms solves argmin f + lambda g
# with f differentiable and g proximable

# create data
A = np.array([[1, 2, 3], [5, 4, 3]])
Y = np.array([[0.5,  0.3, 0.3, 0.4],
              [0.07, 0.8, 0.5, 0.6]])
x0 = np.zeros([3, 4])
H = A.T.dot(A)
mu = 0.2
AtY = A.T.dot(Y)

# set up the gradient function of f
def grad(x):
    return H.dot(x) - AtY

# set up the cost function (so as to record its value at each iteration)
def cost(data):
    return 0.5 * np.linalg.norm(Y - A.dot(data['x']), 'fro')**2 +\
        mu * np.linalg.norm(data['x'].flatten(), 1)

# set up the display function, used to display the results during the iterations
def display(data):
    plt.plot(data['recording']['cost'])


parameters = {}
# set the arguments for the algorithm a list of the necessary and optional
# arguments is available in the docstrings
# gradient function of f
parameters['gradient'] = grad
# proximal operator of g
parameters['proximal'] = proximal.operators.nonnegative_soft_thresholding
parameters['lambda'] = mu
parameters['lipschitz_constant'] = np.linalg.norm(H, 2)
parameters['maximum_iteration'] = 50
parameters['initialization'] = x0
# register the cost function to be recorded
parameters['recording_functions'] = {'cost': cost}
# register the cost function to be recorded
parameters['display_function'] = display

# create the algorithm instance one can provide the parameters
# at this moment or when calling the run function.
FB = proximal.algorithms.ForwardBackward()
# Launch the algorithm with the parameters. Other parameters can be passed
# independently
res = FB.run(parameters, initialization=x0, verbose=1)
