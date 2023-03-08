# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:18:42 2021

@author: xunzhang
"""

from rsome import dro
from rsome import norm
from rsome import E
from rsome import grb_solver as grb
from rsome import msk_solver as msk
import numpy as np
import numpy.random as rd

# Model and ambiguity set parameters
I = 2
S = 50
c = np.ones(I)
d = 50 * I
p = 1 + 4*rd.rand(I)
zbar = 100 * rd.rand(I)
zhat = zbar * rd.rand(S, I)
theta = 0.01 * zbar.min()

# Modeling with RSOME
model = dro.Model(S)                        # Create a DRO model with S scenarios
z = model.rvar(I)                           # Random demand z
u = model.rvar()                            # Auxiliary random variable

fset = model.ambiguity()                    #

fset.suppset(0 <= z,
    norm(z - mu) <= u) # Define the support for each scenario
fset.exptset(E(u) <= sigma, E(z) == mu)                 # The Wasserstein metric constraint                               # An array of scenario probabilities

x = model.dvar(I)                           # Define first-stage decisions
y = model.dvar(I)                           # Define decision rule variables
y.adapt(z)                                  # y affinely adapts to z
y.adapt(u)                                  # y affinely adapts to u

model.minsup(-p@x + E(p@y), fset)           # Worst-case expectation over fset
model.st(y[0] >= 0)                            # Constraints
model.st(y[i] >= y[i-1] + x - z)                        # Constraints
model.st(x >= 0)                            # Constraints
model.st(c@x == d)                          # Constraints

model.solve(msk)                            # Solve the model by Gurobi