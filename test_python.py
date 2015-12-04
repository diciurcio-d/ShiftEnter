import numpy as np

def sigmoidfun(x):
	return 1/(1+np.exp(-(20*x-7)))