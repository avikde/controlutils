'''
Helpful utilities
'''

import numpy as np
import scipy

Skew2 = np.array([[0, -1], [1, 0]])

def rot2(theta):
	return np.array([
		[np.cos(theta), -np.sin(theta)],
		[np.sin(theta), np.cos(theta)]
	])

def screw(twistdt):
	# Assuming SE(2)
	twistHat = np.vstack((
		np.hstack((Skew2 * twistdt[2], np.array([[twistdt[0]],[twistdt[1]]]))),
		np.zeros(3)
	))
	return scipy.linalg.expm(twistHat)
