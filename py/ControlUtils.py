'''
Helpful utilities
'''

import numpy as np
import scipy


def rot2(theta):
	return np.array([
		[np.cos(theta), -np.sin(theta)],
		[np.sin(theta), np.cos(theta)]
	])

def screw(twistdt):
	# Assuming SE(2)
	twistHat = np.vstack((
		np.hstack((np.array([[0, -twistdt[2]], [twistdt[2], 0]]), np.array([[twistdt[0]],[twistdt[1]]]))),
		np.zeros(3)
	))
	return scipy.linalg.expm(twistHat)
