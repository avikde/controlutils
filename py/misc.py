import autograd.numpy as np
from scipy.spatial.transform import Rotation
from . import kinematics as kin

def rectangle(xy, ang, width, height, rawxy=False, **patchkw):
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

    return corners if rawxy else Polygon(corners, **patchkw)


def cuboid(center, rotvec, lwh, rawxy=False, **patchkw):
    '''Vertices for a cuboid
    '''
    vertsB = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]]) @ np.diag(lwh)
    # tested: this adds to each row
    Z = Rotation.from_rotvec(rotvec).apply(vertsB) + center # this is vertsW

    if rawxy:
        return Z
    else:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # list of sides' polygons of figure
        polysides = [[Z[0], Z[1], Z[2], Z[3]],
                [Z[4], Z[5], Z[6], Z[7]],
                [Z[0], Z[1], Z[5], Z[4]],
                [Z[2], Z[3], Z[7], Z[6]],
                [Z[1], Z[2], Z[6], Z[5]],
                [Z[4], Z[7], Z[3], Z[0]]]

        return Z, Poly3DCollection(polysides, **patchkw)


