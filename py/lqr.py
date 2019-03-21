'''
Original credits: http://www.kostasalexis.com/lqr-control.html
Modifications by Avik De
'''

from __future__ import division, print_function
 
import numpy as np
import scipy.linalg
 
def lqr(A,B,Q,R, eigs=False):
	"""Solve the continuous time lqr controller.
		
	dx/dt = A x + B u
		
	cost = integral x.T*Q*x + u.T*R*u
	"""
	#ref Bertsekas, p.151

	#first, try to solve the ricatti equation
	X = scipy.linalg.solve_continuous_are(A, B, Q, R)
		
	#compute the LQR gain
	K = np.linalg.inv(R) @ (B.T @ X)
	
	if eigs:
		eigVals, eigVecs = np.linalg.eig(A - B @ K)
		return K, X, eigVals
	else:
		return K, X
 
def dlqr(A,B,Q,R, eigs=False):
	"""Solve the discrete time lqr controller.
		
		
	x[k+1] = A x[k] + B u[k]
		
	cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
	"""
	#ref Bertsekas, p.151

	#first, try to solve the ricatti equation
	X = scipy.linalg.solve_discrete_are(A, B, Q, R)
		
	#compute the LQR gain
	K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
	
	if eigs:
		eigVals, eigVecs = np.linalg.eig(A - B @ K)
		return K, X, eigVals
	else:
		return K, X

