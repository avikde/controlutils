'''
Helpful utilities
'''

import numpy as np
from scipy.spatial.transform import Rotation

Skew2 = np.array([[0, -1], [1, 0]])

def skew(a=None):
	# Skew of a vector
	if len(a) == 3:
		return np.array([
			[0, -a[2], a[1]],
			[a[2], 0, -a[0]],
			[-a[1], a[0], 0]
			])
	else:
		return Skew2

def rot2(theta):
	return np.array([
		[np.cos(theta), -np.sin(theta)],
		[np.sin(theta), np.cos(theta)]
	])

def rot2lin(phiz):
	'''Linearization of rotation (small angle)'''
	return np.eye(2) + phiz * Skew2

def affineKinematics(q, dt):
	'''Linearized kinematics of applying an SE(n) twist to a pose.
	q = input pose
	dt = time to apply twist (should be small for accuracy)
	Returns B s.t.
	qnext = q + B(q) v
	'''
	if len(q) != 3:
		raise('Only SE(2) implemented')
	# qnext = {q0, q1} + RotationMatrix[q2].{v0, v1} t
	return dt * np.vstack((
		np.hstack((rot2(q[2]), np.zeros((2,1)))),
		np.array([0, 0, 1])
	))

def rot(phi):
	'''Either planar, or rotation vector for 3D'''
	if len(phi) == 1:
		return rot2(phi[0])
	else:
		return Rotation.from_rotvec(phi).as_dcm()

def applyTwist(q0, twistDes, dt):
	'''Integrates up the twist. Should work regardless of SE(2) or SE(3).
	'''
	if len(twistDes) != 3:
		raise('Only SE(2) implemented')
	return q0 + affineKinematics(q0, dt) @ twistDes
