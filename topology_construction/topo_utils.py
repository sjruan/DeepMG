from tptk.common.spatial_func import SPoint
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def magnitude(vector):
    return np.sqrt(np.dot(np.array(vector),np.array(vector)))


def norm(vector):
    return np.array(vector)/magnitude(np.array(vector))


def ccw(A,B,C):
    return (C.lat-A.lat) * (B.lng-A.lng) > (B.lat-A.lat) * (C.lng-A.lng)


def is_line_line_intersected(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def line_ray_intersection_test(o, f, a, b):
    """
    :param o: ray original point SPoint
    :param f: ray from point SPoint ray: f->o
    :param a: line segment point 1 SPoint
    :param b: line segment point 2 SPoint
    :return:
    """
    o = np.array((o.lng, o.lat), dtype=np.float)
    dir = np.array(norm((o[0] - f.lng, o[1] - f.lat)), dtype=np.float)
    a = np.array((a.lng, a.lat), dtype=np.float)
    b = np.array((b.lng, b.lat), dtype=np.float)

    v1 = o - a
    v2 = b - a
    v3 = np.asarray([-dir[1], dir[0]])
    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    # t1=inf parallel
    if t1 == np.inf or t1 < 0:
        # ray has no intersection with line segment
        return None
    else:
        pt = o + t1 * dir
        # 1. t2<0, in extension of a; 2. t2 in [0,1], within ab;  3. t2>1, in extension of b
        return t2, SPoint(pt[1], pt[0])
