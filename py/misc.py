import numpy as np
from . import geometry as geom

def rectangle(xy, ang, width, height):
	'''draw a planar rectangle using matplotlib but based on the center
	'''
	from matplotlib.patches import Rectangle
	# First argument is lower left (i.e. smallest x,y components), not center
	lowerLeftCorner = np.full(2, np.inf)
	Ryaw = geom.rot2(ang)

	# corners in body frame
	di = 0.5 * np.array([
		[width,height],
		[-width,height],
		[-width,-height],
		[width,-height]
		])

	# find which corner is lower left
	for i in range(4):
		corneri = xy + Ryaw @ di[i,0:2]
	# Not quite sure about what it does when "leftmost" and "lowermost" are different corners
	if corneri[1] < lowerLeftCorner[1]:
		lowerLeftCorner = corneri
	return Rectangle(lowerLeftCorner, width, height, angle=np.degrees(ang))

