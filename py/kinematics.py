'''
Helpful utilities
'''

import numpy as np

Skew2 = np.array([[0, -1], [1, 0]])

def rot2(theta):
	return np.array([
		[np.cos(theta), -np.sin(theta)],
		[np.sin(theta), np.cos(theta)]
	])

def rot2lin(phiz):
	'''Linearization of rotation (small angle)'''
	return np.eye(2) + phiz * Skew2

def applyTwist(q0, twistDes, dt):
	'''Integrates up the twist. Should work regardless of SE(2) or SE(3).
	'''
	if len(twistDes) != 3:
		raise('Only SE(2) implemented')
	
	qnew = np.hstack((
		# Rotate to align twist to body frame
		q0[0:2] + (rot2(q0[2]) @ twistDes[0:2]) * dt,
		# Last element is just yaw
		q0[2] + twistDes[2] * dt
	))
	return qnew

def linearizedKinematics(q, dt):
	'''Linearized kinematics of applying an SE(n) twist to a pose. (Linearized version of applyTwist)
	q = input pose
	dt = time to apply twist (should be small for accuracy)
	Returns B s.t.
	qnext = q + B(q) v
	'''
	if len(q) != 3:
		raise('Only SE(2) implemented')
	
	posColVec = np.array([[q[0]],[q[1]]])
	# Linearize a homogeneous transformation exp(t skew(V)) qhomog ~= B(q) V for the first two rows
	return dt * np.vstack((
		np.hstack((np.eye(2), Skew2 @ posColVec)),
		np.array([0, 0, 1])
	))
