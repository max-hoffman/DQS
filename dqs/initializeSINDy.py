from scipy.linalg import svd
import sindy

def initializeSINDy(data, henkelRows, functionCount, maxOrder, dt, harmonics=False):
  data = sindy.henkelify(data, henkelRows)
  U, s, V = svd(data, full_matrices=False)
  V=V.conj().T
  dV = sindy.fourthOrderDerivative(V, dt, functionCount)
  theta = sindy.poolData(V[2:-2,0:functionCount], functionCount, maxOrder , False)
  theta, norms = sindy.normalize(theta, theta.shape[1])

  return V, dV, theta, norms
  