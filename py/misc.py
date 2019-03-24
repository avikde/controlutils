import numpy as np
from . import kinematics as kin

def rectangle(xy, ang, width, height):
	'''draw a planar rectangle using matplotlib but based on the center
	'''
	from matplotlib.patches import Polygon
	
	Ryaw = kin.rot2(ang)

	# corners in body frame
	di = 0.5 * np.array([
		[width,height],
		[-width,height],
		[-width,-height],
		[width,-height]
		])

	# find which corner is lower left
	corners = np.zeros_like(di)
	for i in range(4):
		corners[i,:] = xy + Ryaw @ di[i,:]

	return Polygon(corners)

