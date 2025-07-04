"""
Contains general utility functions.
"""
import numpy as np
from scipy.special import logsumexp


def rot(theta):
    """
    Constructs a rotation matrix for given angle alpha.
    :param theta: angle of orientation
    :return: Rotation matrix in 2D around theta (2x2)
    """
    theta = np.squeeze(theta)
    r = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    if len(r.shape) == 3:
        r = np.moveaxis(r, -1, 0)
        assert r.shape == (len(theta), 2, 2)
    return r


def matrix_to_params(X):
    """Convert shape matrix X to parameter form [alpha, l1, l2] with semi-axis length"""
    assert X.shape == (2, 2), "X is not a 2x2 matrix"
    val, vec = np.linalg.eig(X)  # eigenvalue decomposition
    alpha = np.arctan2(vec[1][0], vec[0][0])  # calculate angle of orientation
    alpha = (alpha + 2 * np.pi) % (2 * np.pi)  # just in case alpha was negative
    p = [alpha, *np.sqrt(val)]
    return np.array(p)


def params_to_matrix(p):
    """
    Convert parameters [alpha, l1, l2] to shape matrix X (2x2)
    """
    X = rot(p[0]) @ np.diag(np.array(p[1:]) ** 2) @ rot(p[0]).T
    return X


def state_to_srs(extent_state):
    """Convert state parameters [theta, l, w] to square rood space shape matrix"""
    theta, l, w = extent_state
    return rot(theta) @ np.diag([l, w]) @ rot(theta).T


def srs_to_state(srs_mat):
    """convert square root space matrix to shape parameters"""
    # convert to shape matrix by squaring and then calculate parameters
    return matrix_to_params((srs_mat @ srs_mat.T).astype(float))


def vect(M):
    """
    From original MEM-EKF* paper:
    Constructs a column vector from a matrix M by stacking its column vectors
    """
    v = M.flatten(order="F")  # just use ndarray.flatten(), pass `order='F'` for column-major order
    v = np.reshape(v, (len(v), 1))  # ensure output is column vector
    return v


def lse_norm(log_likelihoods):
    """
    Use log_sum_exp trick for normalization in [0, 1]
    :param log_likelihoods: Array of log-likelihoods
    :return: Array of probabilities normalized in [0, 1] computed from log likelihoods
    """
    c = log_likelihoods.max()
    t = c + logsumexp(log_likelihoods - c)
    return np.exp(log_likelihoods - t)
