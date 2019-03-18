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

def rot2lin(phiz):
	'''Linearization of rotation (small angle)'''
	return np.eye(2) + phiz * Skew2

def screw(twistdt):
	if len(twistdt) != 3:
		raise 'Only SE(2) implemented'
	# Assuming SE(2)
	twistHat = np.vstack((
		np.hstack((Skew2 * twistdt[2], np.array([[twistdt[0]],[twistdt[1]]]))),
		np.zeros(3)
	))
	return scipy.linalg.expm(twistHat)

def twistInt(q0, twistDes, N, dt):
	# Integrates up the twist. Should work regardless of SE(2) or SE(3)
	# This linear map is the "transition" matrix
	Screw = screw(twistDes * dt)
	# Apply recursively
	qtraj = np.zeros((N, len(q0)))
	qtraj[0, :] = q0
	for k in range(1, N):
		# Homegeneous coords
		qtraj[k, :] = Screw @ np.hstack((qtraj[k-1, 0:2], 1))
		# Last element is just yaw
		qtraj[k, 2] = qtraj[k-1, 2] + twistDes[2] * dt
	return qtraj
