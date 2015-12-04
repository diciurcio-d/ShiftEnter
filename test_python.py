import numpy as np

def sigmoidfun(x):
	return 1/(1+np.exp(-0.007*(x-800)))