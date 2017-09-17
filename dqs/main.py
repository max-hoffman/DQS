import numpy as np
from scipy import integrate
from scipy.linalg import svd
import matplotlib.pyplot as plt
import pylab

from initializeSINDy import *
from sampleGenerator import generateTrainingSample

functionCount = 3
maxOrder = 3
maxElements = 3
coefficientMagnitude = 2
henkelRows = 10
dt = .001

data, xiOracle = generateTrainingSample(functionCount, maxOrder, maxElements, coefficientMagnitude)
V, dX, theta, norms = initializeSINDy(data[:, 0], henkelRows, functionCount, maxOrder, dt)



