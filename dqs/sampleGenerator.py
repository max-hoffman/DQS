import numpy as np
from math import factorial
import random as rd

def generateTrainingSample(functionCount, maxOrder, maxElements):
  rd.seed()

  randomSystem = _generateSystemString(functionCount, maxOrder, maxElements)
  xiOracle = _marshalXi(randomSystem, maxOrder, functionCount)

  # TODO : create dynamics function from the oracle
  # TODO : solve ODE to get data points

def _marshallDynamicalSystem(systemStrings):
  def dynamicalSystem(X, t=0):
    dX = np.zeros(X.shape)
    for i in range(X.shape):
      dX = 


def _generateSystemString(functionCount, maxOrder, maxElements):
  functions = []
  charRef = 97

  for functionIdx in range(functionCount):
    elementCount = rd.randint(1, maxElements)
    newFunction = np.empty(elementCount, dtype="S"+str(maxOrder))
    # print(elementCount, newFunction)
    for elementIdx in range(elementCount):
      elString = ""
      order = rd.randint(1, maxOrder)
      for i in range(order):
        new = rd.randint(0, maxOrder-1)
        elString += chr(charRef+new)
      newFunction[elementIdx] = elString
    functions.append(newFunction)
  return functions
  
def _sumOfComboWithReplacement(selectN, fromOptions):
  count = 1
  for i in range(1, selectN+1):
    count += factorial(fromOptions+i-1) / (factorial(i) * factorial(fromOptions-1))
  return int(count)


def _stringSum(string):
  "Helper function to condense function string to numerical representation"
  chars = list(string)
  ints = list(map(lambda x: (int(format(ord(x), "x"))-61), chars))
  return reduce(
    (lambda accum, curr: accum + curr), ints, 0
  )

def _indexForString(string, modes):
  "Returns the Xi index for a given string"
  order = len(string)
  ref = _sumOfComboWithReplacement(order-1, modes)
  return ref + _stringSum(string)

def _marshalXi(constraints, order, modes):
  "Converts string representation to sparse representation"
  rowCount = len(constraints)
  colCount = _sumOfComboWithReplacement(order, modes)
  xi = np.zeros((rowCount,colCount))

  for row in range(rowCount):
    for varString in constraints[row]:
      idx = _indexForString(varString, modes)
      xi[row, idx] = 1
  
  return xi.T