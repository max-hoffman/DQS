import numpy as np
from math import factorial
import random as rd

def generateTrainingSample(functionCount, maxOrder, maxElements, coefficientMagnitude):
  rd.seed()

  randomSystem = _generateSystemString(functionCount, maxOrder, maxElements, coefficientMagnitude)
  xiOracle = _marshalXi(randomSystem, maxOrder, functionCount)
  systemODE = _marshallDynamicalSystem(randomSystem, xiOracle)

  # TODO : solve ODE to get data points
  timeStart = .01
  timeStop = 100
  time = np.linspace(timeStart, timeStop, num = numberOfPoints)
  initialCond = np.zeros(functionCount)
  initialCond = map((lambda x: x + rd.randint(-coefficientMagnitude, coefficientMagnitude)), initialCond)

  data, infodict = integrate.odeint(systemODE, initialCond, time, full_output=1)
  print( infodict['message'])

  return data, xiOracle

def _marshallDynamicalSystem(systemStrings, xi):

  def computeProduct(X, values, coefficient):
    chars = list(values)
    indices = list(map(lambda x: (int(format(ord(x), "x"))-61), chars))
    product = reduce((lambda accum, idx: accum * X[idx]), indices, 1)
    return product * coefficient

  def dynamicalSystem(X, t=0):
    functionCount = len(systemStrings)
    dX = np.zeros(functionCount)
    for funcIdx in range(functionCount):
      for string in systemStrings[funcIdx]:
        vals = string["val"]
        coef = string["coefficient"]
        dX[funcIdx] += computeProduct(X, vals, coef)
  
  return dynamicalSystem

def _generateSystemString(functionCount, maxOrder, maxElements, coefficientMagnitude):
  functions = []
  charRef = 97

  for functionIdx in range(functionCount):
    elementCount = rd.randint(1, maxElements)
    newFunction = []
    # print(elementCount, newFunction)
    for elementIdx in range(elementCount):
      elString = ""
      order = rd.randint(1, maxOrder)
      for i in range(order):
        new = rd.randint(0, maxOrder-1)
        elString += chr(charRef+new)
      newFunction.append({ 'coefficient': _randomCoefficient(coefficientMagnitude), 'val': elString })
    functions.append(newFunction)
  return functions
  
def _randomCoefficient(magnitude):
  coefficient = rd.random() * rd.randint(1, magnitude)
  if rd.random() < .5:
    coefficient = -coefficient
  return coefficient

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
      val = varString["val"]
      coef = varString["coefficient"]
      idx = _indexForString(val, modes)
      xi[row, idx] = coef
  
  return xi.T
