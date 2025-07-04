import numpy as np
from scipy.linalg import sqrtm
from shapely.geometry import Polygon

from src.utilities.utils import params_to_matrix, rot


def gwd_shape(d1, X1, d2, X2, return_parts=False):
    """
    Squared Gaussian Wasserstein Distance as defined in eq. (63)
    Compares two elliptic extended targets defined by their centers (d) and shape matrices (X)
    """
    d1, X1, d2, X2 = np.array(d1), np.array(X1), np.array(d2), np.array(X2)
    # first: perform preliminary calculations
    X1sqrt = sqrtm(X1.astype(float))
    C = sqrtm(X1sqrt @ X2 @ X1sqrt)

    # finally: construct GW distance
    d1 = np.linalg.norm(d1 - d2) ** 2
    d2 = np.trace(X1 + X2 - 2 * C)
    d = d1 + d2
    # for sgw == 0, rounding errors might cause a minimally negative result. round to avoid
    if d < 0:
        d = np.around(d, 4)
    if return_parts:
        return d, d1, d2
    else:
        return d


def gwd(m1, p1, m2, p2, return_parts=False):
    """
    Calculates the Squared Gaussian Wasserstein Distance, using the 'gwd_shape' function.
    Compares two ellipses with object center m=[x_location, y_location]
    and object extend p=[orientation, length, width].
    """
    d1 = np.array(m1).astype(float)
    d2 = np.array(m2).astype(float)
    X1 = params_to_matrix(p1)
    X2 = params_to_matrix(p2)
    return gwd_shape(d1, X1, d2, X2, return_parts=return_parts)


def gwd_full_state(s1, s2, return_parts=False):
    """
    Returns the squared Gauss Wasserstein distance for two elliptical extended object.
    Each object is parameterized by a 7D state in the form:
        [loc_x, loc_y, velocity_x, velocity_y, orientation, full_length, full_width]
    :param s1: 7D state of object 1
    :param s2: 7D state of object 2
    :param return_parts:
    :return: Squared Gauss Wasserstein distance
    """
    # split each state into center and extent information
    # use half-axis length
    s1, s2 = np.array(s1).astype(float), np.array(s2).astype(float)
    m1 = s1[:2]
    p1 = s1[4:]
    p1[1:] /= 2

    m2 = s2[:2]
    p2 = s2[4:]
    p2[1:] /= 2
    return gwd(m1, p1, m2, p2, return_parts=return_parts)


def state_to_ellipse_contour_pts(m, p, n_pts=100):
    """Given mean m and p = [theta, l, w] for an ellipse, return n_pts many sampled of the contour, uniform in angle"""
    ellipse_angle_array = np.linspace(0.0, 2.0 * np.pi, n_pts)
    pts = (m[:, None] + rot(p[0]) @ np.diag([p[1], p[2]]) @ np.array(
        [np.cos(ellipse_angle_array), np.sin(ellipse_angle_array)])).T
    return np.array(pts)


def iou_full_state(s1, s2):
    """
    Returns the Intersection over Union between two elliptical extended objects.
    Each object is parameterized by a 7D state in the form:
        [loc_x, loc_y, velocity_x, velocity_y, orientation, full_length, full_width]
    :param s1: 7D state of object 1
    :param s2: 7D state of object 2
    :return: IoU between objects
    """
    try:
        p1 = Polygon(state_to_ellipse_contour_pts(m=s1[:2], p=s1[-3:]))
        p2 = Polygon(state_to_ellipse_contour_pts(m=s2[:2], p=s2[-3:]))
        return p1.intersection(p2).area / p1.union(p2).area
    except:
        return 0


def errors_per_state_component(s1, s2, sort_axes=True, half_angles=True):
    """
    Return individual errors of the components of the state
    :param s1: 7D state of object 1
    :param s2: 7D state of object 2
    :param sort_axes: If True, both targets will be sorted s.t. the primary semi-axis comes first (orientation will be
    adapted accordingly as necessary by a 90° turn)
    :param half_angles: If True, will only consider angles in [0, pi] instead of [0, 2pi], i.e., forward vs backward
    facing ellipse won't be punished if this is set to True
    :return: 4tuple of errors for (center, orientation, first semi-axis, second semi-axis)
    """
    # split each state into center and extent information
    # use half-axis length
    s1, s2 = np.array(s1).astype(float), np.array(s2).astype(float)
    s1[-2:] = np.abs(s1[-2:])
    s2[-2:] = np.abs(s2[-2:])
    f = 1 if half_angles else 2
    if sort_axes:
        if s1[5] < s1[6]:
            s1[[5, 6]] = s1[[6, 5]]
            s1[4] += np.pi / 2
        if s2[5] < s2[6]:
            s2[[5, 6]] = s2[[6, 5]]
            s2[4] += np.pi / 2

    m1 = s1[:2]
    p1 = s1[4:]
    p1[0] = p1[0] % (f * np.pi)
    p1[1:] /= 2

    m2 = s2[:2]
    p2 = s2[4:]
    p2[0] = p2[0] % (f * np.pi)
    p2[1:] /= 2

    # compute metrics
    error_center = np.linalg.norm(m1 - m2)
    error_orientation = np.abs(p1[0] - p2[0])
    if error_orientation > np.pi * (f / 2):
        error_orientation -= np.pi * f
    error_orientation = np.abs(error_orientation) % (f * np.pi)
    error_l1 = np.abs(p1[1] - p2[1])
    error_l2 = np.abs(p1[2] - p2[2])

    return error_center, error_orientation, error_l1, error_l2
