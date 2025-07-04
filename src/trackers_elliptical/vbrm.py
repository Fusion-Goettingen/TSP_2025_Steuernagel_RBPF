"""
Implements the Variational Bayes Random Matrix tracker (VBRM)

Please refer to:
    B. Tuncer and E. Özkan, "Random Matrix Based Extended Target Tracking With Orientation: A New Model and Inference,"
    in IEEE Transactions on Signal Processing, vol. 69, pp. 1910-1923, 2021, doi: 10.1109/TSP.2021.3065136.

    https://ieeexplore.ieee.org/document/9374715
"""
import numpy as np
from scipy.linalg import block_diag
from src.trackers_elliptical.abstract_tracking import AbtractEllipticalTracker
from src.utilities.utils import rot
from numpy.linalg import inv


class TrackerVBRM(AbtractEllipticalTracker):
    def __init__(self,
                 m_init,
                 p_init,
                 P_init,
                 gamma,
                 l_max,
                 Q,
                 q_theta,
                 R,
                 init_theta_var,
                 alpha_init,
                 time_delta=1):
        # state etc
        self.x = np.array(m_init).astype(float)  # kinematic state
        self.P = P_init  # kinematic covariance
        self.alpha = np.array([alpha_init, alpha_init])  # list of shape parameters
        self.beta = p_init[1:].astype(float) ** 2 * (self.alpha - 1)
        self.theta = p_init[0]  # orientation estimate
        self.theta_var = init_theta_var  # variance of orientation estimate

        # general hyperparameters
        self.gamma = gamma  # forgetting factor for prediction
        self.F = block_diag(np.array([
            [1, 0, time_delta, 0],
            [0, 1, 0, time_delta],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]), 1)
        self.Q = block_diag(Q, q_theta)  # process noise
        self.l_max = l_max  # maximum number of iterations
        self.s = 0.25  # scaling factor
        self.H = np.block([np.eye(2), np.zeros((2, 2))])  # measurement matrix
        self.R = R  # meas noise cov. matrix

    def predict(self):
        assert self.x is not None, "x is None - did you predict before first update?"
        self.x, self.P, self.alpha, self.beta, self.theta, self.theta_var = predict_vbrm(self.x,
                                                                                         self.P,
                                                                                         self.alpha,
                                                                                         self.beta,
                                                                                         self.theta,
                                                                                         self.theta_var,
                                                                                         self.gamma,
                                                                                         self.F,
                                                                                         self.Q)
        return self.get_state()

    def update(self, measurements: np.ndarray):
        self.x, self.P, self.alpha, self.beta, self.theta, self.theta_var = update_vbrm(self.x,
                                                                                        self.P,
                                                                                        self.alpha,
                                                                                        self.beta,
                                                                                        self.theta,
                                                                                        self.theta_var,
                                                                                        measurements,
                                                                                        self.l_max,
                                                                                        self.s,
                                                                                        self.H,
                                                                                        self.R)

        return self.get_state()

    def get_state(self):
        # c.f.:
        #   https://github.com/Metu-Sensor-Fusion-Lab/Random-Matrix-Based-Extended-Target-Tracking-With-Orientation/blob/main/demo.m#L62
        semi_axis_lengths = np.sqrt(np.array(self.beta) / (np.array(self.alpha) - 1))
        state = [
            *self.x,
            self.theta,
            semi_axis_lengths[0] * 2,  # return full axis
            semi_axis_lengths[1] * 2,  # return full axis
        ]
        return np.array(state)

    def get_state_and_cov(self):
        raise NotImplementedError

    def set_R(self, R):
        assert R.shape == self.R.shape
        self.R = R


def update_vbrm(x_minus, P_minus, alpha_list_minus, beta_list_minus, theta_minus, theta_var_minus, Y, l_max, s,
                H, R):
    """
    Perform an update for the variational bayes random matrix tracker.

    All formulas according to:
    B. Tuncer and E. Özkan, "Random Matrix Based Extended Target Tracking With Orientation: A New Model and Inference,"
    in IEEE Transactions on Signal Processing, vol. 69, pp. 1910-1923, 2021, doi: 10.1109/TSP.2021.3065136.

    https://ieeexplore.ieee.org/document/9374715

    :param x_minus: Prior estimate of the state
    :param P_minus: Prior covariance of the state
    :param alpha_list_minus: Prior list of IG shape parameters. Defines the extent estimate together with the list of
    betas.
    :param beta_list_minus: Prior list of IG scale parameters. Defines the extent estimate together with the list of
    alphas.
    :param theta_minus: Prior estimate of the object orientation in radians
    :param theta_var_minus: Prior variance (/covariance) of the orientation theta. In the paper denoted as capital theta
    :param Y: Measurements used to update
    :param l_max: Number of iterations to run for.
    :param s: scaling parameter
    :param H: measurement matrix
    :param R: measurement noise
    :return: x_plus, P_plus, alpha_list_plus, beta_list_plus, theta_plus, theta_var_plus: The posterior state estimates
    and their corresponding variance where applicable.
    """
    # === PREPARE
    # assert correctness of parameters
    assert l_max > 0, "Number of Iterations given as {}, needs to be >0".format(l_max)
    assert len(alpha_list_minus) == len(beta_list_minus), \
        "Different number of alphas ({}) and betas ({}) provided".format(len(alpha_list_minus),
                                                                         len(beta_list_minus))

    # set up extractable variables
    Y = np.array(Y)
    n_y = len(alpha_list_minus)
    m_k = len(Y)

    # === INITIALIZATION
    x_iterations = [x_minus]
    P_iterations = [P_minus]
    alpha_list_iterations = [alpha_list_minus]
    beta_list_iterations = [beta_list_minus]
    theta_iterations = [theta_minus]
    theta_var_iterations = [theta_var_minus]
    z_iterations = [Y]
    # to initial sigma, use eq. 34:
    sigma = exp_qX_sX(s=s, alpha_list=alpha_list_minus, beta_list=beta_list_minus)
    sigma_iterations = [sigma]

    # === ITERATIONS
    for l in range(l_max):
        # ----------
        # 1. x and P
        #   using eq.17a and 17b
        #   based on eq36 for the expectation over qX and qTheta, which uses 33c for E(qX)[sXk^-1]
        #   z_bar is given right below 15d, and the expectation over Z necessary to calculate it is in eq33b
        z_j_list = [z for z in z_iterations[-1]]  # eq 33b
        z_bar = np.average(z_j_list, axis=0).reshape((2, 1))  # given below 15d

        # calculate Exp_qX_qTheta[(T_theta @ X @ T_theta.T)^-1] according to eq. 36
        exp_realization = exp_qX_qT(theta_iterations[-1], theta_var_iterations[-1],
                                    alpha_list_iterations[-1], beta_list_iterations[-1], s)
        P_next = inv(inv(P_minus) + m_k * H.T @ exp_realization @ H)
        x_next = P_next @ (inv(P_minus) @ x_minus.reshape((4, 1)) + m_k * H.T @ exp_realization @ z_bar)

        # ----------
        # 2. theta and theta_var
        #   using eq 32a and 32b
        #   which rely on 32c and 32d for delta and delta_var (capital delta)
        #   these rely on expectations 33c and 33d
        # use 33c for sX_k^-1 (bar)
        n_y = len(alpha_list_iterations[-1])
        sX_bar = np.diag([alpha_list_iterations[-1][i] / (s * beta_list_iterations[-1][i]) for i in range(n_y)])

        # rotation matrix around theta (T) and its derivative w.r.t theta (T_dash)
        T = rot(theta_iterations[-1])
        T_dash = rot_deriv(theta_iterations[-1])

        # fixing to float orientation for this application
        delta = 0
        delta_var = 0
        innov_bar_sum = np.zeros((2, 2))
        for j in range(m_k):
            # use 33d for innov_bar (for current j)
            innov = np.array(z_iterations[-1][j]) - H @ x_iterations[-1]
            innov = innov.reshape((-1, 1))
            innov_bar = H @ P_iterations[-1] @ H.T + sigma_iterations[-1] + innov @ innov.T
            innov_bar_sum += innov_bar

            # first part of delta calculation
            delta += np.trace(sX_bar @ T_dash.T @ innov_bar @ T_dash * theta_iterations[-1])

            # second part of delta calculation
            delta -= np.trace(sX_bar @ T.T @ innov_bar @ T_dash)

            # capital delta
            delta_var += np.trace(sX_bar @ T_dash.T @ innov_bar @ T_dash)

        theta_var_next = (theta_var_minus ** -1 + delta_var) ** -1
        theta_next = theta_var_next * (theta_var_minus ** -1 * theta_minus + delta)
        theta_next = np.squeeze(theta_next)

        # ----------
        # 3. alpha, beta
        # c.f.:
        # https://github.com/Metu-Sensor-Fusion-Lab/Random-Matrix-Based-Extended-Target-Tracking-With-Orientation/blob/main/UpdateVB.m
        alpha_list_next = np.array(alpha_list_iterations[0]) + m_k * 0.5
        LsumLbar = ComputeLXL(-theta_iterations[-1], np.squeeze(theta_var_iterations[-1]),
                              (1 / (2 * s)) * innov_bar_sum)
        beta_list_next = np.array(beta_list_iterations[0]) + np.diag(LsumLbar.reshape((2, 2)))

        # ----------
        # 4. z and sigma
        #   based on 25a and 25b
        #   E(qX, qTheta) is given in eq. 36, and E(qX)[xk] is given in 33a
        exp_oriented_shape_inv = exp_qX_qT(theta_iterations[-1], theta_var_iterations[-1],
                                           alpha_list_iterations[-1], beta_list_iterations[-1], s)
        sigma_next = inv(exp_oriented_shape_inv + inv(R))
        z_next = []
        for j in range(len(Y)):
            next_individual_z = sigma_next @ (exp_oriented_shape_inv @ H @ x_iterations[-1] + inv(R) @ Y[j, :])
            z_next.append(next_individual_z)

        # ----------
        # X. append everything to the iteration lists again
        x_iterations.append(x_next.reshape((4,)))
        P_iterations.append(P_next)
        theta_iterations.append(theta_next)
        theta_var_iterations.append(theta_var_next)
        alpha_list_iterations.append(alpha_list_next)
        beta_list_iterations.append(beta_list_next)
        sigma_iterations.append(sigma_next)
        z_iterations.append(np.array(z_next))

        # === FINAL ESTIMATES
        x_plus = x_iterations[-1]
        P_plus = P_iterations[-1]
        alpha_list_plus = alpha_list_iterations[-1]
        beta_list_plus = beta_list_iterations[-1]
        theta_plus = theta_iterations[-1]
        theta_var_plus = theta_var_iterations[-1]

    return x_plus, P_plus, alpha_list_plus, beta_list_plus, theta_plus, theta_var_plus


def predict_vbrm(x_minus, P_minus, alpha_list_minus, beta_list_minus, theta_minus, theta_var_minus, gamma, F, Q):
    """
    Perform a predict step for the variational bayes random matrix tracker.

    All formulas according to:
    B. Tuncer and E. Özkan, "Random Matrix Based Extended Target Tracking With Orientation: A New Model and Inference,"
    in IEEE Transactions on Signal Processing, vol. 69, pp. 1910-1923, 2021, doi: 10.1109/TSP.2021.3065136.

    https://ieeexplore.ieee.org/document/9374715

    :param x_minus: Prior estimate of the state
    :param P_minus: Prior covariance of the state
    :param alpha_list_minus: Prior list of IG shape parameters. Defines the extent estimate together with the list of
    betas.
    :param beta_list_minus: Prior list of IG scale parameters. Defines the extent estimate together with the list of
    alphas.
    :param theta_minus: Prior estimate of the object orientation in radians
    :param theta_var_minus: Prior variance (/covariance) of the orientation theta. In the paper denoted as capital theta
    :param gamma: Forgetting factor gamma
    :param F: Motion model
    :param Q: Process Noise
    :return: x_plus, P_plus, alpha_list_plus, beta_list_plus, theta_plus, theta_var_plus: Predicted state estimates
    and their corresponding variance where applicable.
    """
    # numpy-fy as necessary
    P_minus = np.array(P_minus)
    F, Q = np.array(F), np.array(Q)

    # construct extended state space (i.e. including orientation)
    x_ext = np.hstack([x_minus, theta_minus])
    P_ext = block_diag(P_minus, theta_var_minus)

    # assertions on object shape
    assert F.shape == (len(x_ext), len(x_ext)), \
        "F shape is {}x{}, which does not match extended state of shape".format(F.shape[0], F.shape[1], len(x_ext))
    assert Q.shape == F.shape, "F({}x{}) and Q({}x{}) have different shapes".format(F.shape[0], F.shape[1],
                                                                                    Q.shape[0], Q.shape[1])
    assert Q.shape == P_ext.shape, \
        "F({}x{}) and extended P({}x{}) have different shapes".format(F.shape[0], F.shape[1],
                                                                      P_ext.shape[0], P_ext.shape[1])

    # update x/P (extended)
    x_ext_plus = F @ x_ext
    P_ext_plus = F @ P_ext @ F.T + Q

    # extract x/P/theta/theta_var again
    x_plus = x_ext_plus[:len(x_minus)]
    theta_plus = x_ext_plus[len(x_minus):]
    if isinstance(theta_minus, float) and not isinstance(theta_plus, float) and len(theta_plus) == 1:
        # theta was just orientation
        # simply extract from list to preserve data type as given in input
        theta_plus = float(theta_plus[0])
    P_plus = P_ext_plus[:P_minus.shape[0], :P_minus.shape[1]]
    theta_var_plus = P_ext_plus[P_minus.shape[0]:, P_minus.shape[1]:]
    if isinstance(theta_var_minus, float) and not isinstance(theta_var_plus, float) and len(theta_var_plus) == 1:
        # as above: preserve data type of variance of theta in case it was not given as an array
        theta_var_plus = float(theta_var_plus[0, 0])

    # update state space
    alpha_list_plus = [gamma * alpha for alpha in alpha_list_minus]
    beta_list_plus = [gamma * beta for beta in beta_list_minus]

    return x_plus, P_plus, alpha_list_plus, beta_list_plus, theta_plus, theta_var_plus


def exp_qX_sX(s, alpha_list, beta_list):
    """
    Calculate E_(qX)[sX] as defined in eq. 34

    :param s: scale parameter
    :param alpha_list: list of IG alphas
    :param beta_list: list of IG betas
    :return: Estimated shape matrix
    """
    # prepare
    assert len(alpha_list) == len(beta_list), \
        "{} alphas but {} betas passed to expectation of shape matrix".format(len(alpha_list), len(beta_list))
    n_y = len(alpha_list)  # = len(beta_list) due to assertion above

    # calculate diagonal entries
    diag_entries = [s * beta_list[i] / (alpha_list[i] - 1) for i in range(n_y)]

    # finalize result and return
    X_hat = np.diag(diag_entries)

    return X_hat


def exp_qX_qT(theta, theta_var, alpha_list, beta_list, s):
    """
    Calculates the expectation Exp_qX_qTheta[(sT_theta @ X @ T_theta.T)^-1] according to eq. 36

    :param theta: Estimated orientation
    :param theta_var: variante of orientation estimate
    :param alpha_list: list of IG alphas
    :param beta_list: list of IG betas
    :param s: scale parameter
    :return: Exp_qX_qTheta[(T_theta @ X @ T_theta.T)^-1]
    """
    # E(qX)[sXk^-1] - as defined in 33c
    n_y = len(alpha_list)
    E_shape_inv = np.diag([alpha_list[i] / (s * beta_list[i]) for i in range(n_y)])

    # T_theta
    T_theta = rot(theta)

    # put together
    res = (1 - np.exp(-2 * theta_var)) * (np.trace(E_shape_inv) / 2) * np.eye(2)

    res += np.exp(-2 * theta_var) * (T_theta @ E_shape_inv @ T_theta.T)

    return res


def exp_qX_qT_qZ(x, theta, theta_var, z, sigma, H, P):
    """
    Calculate the expectation given in 33e.

    :param x: estimated state
    :param theta: Estimated orientation
    :param theta_var: variance of orientation estimate
    :param z: single measurement
    :param sigma:
    :param H: measurement matrix
    :return: Expectation realization according to eq. 33e
    """
    theta = -theta  # (35) is for E[T@M@T^T], (33e) is using T^T@M@T -> flip theta to go from T to T^T and vice versa
    # calculate K according to to eq. 35e
    K = np.array([
        1 + np.cos(2 * theta) * np.exp(-2 * theta_var),
        1 - np.cos(2 * theta) * np.exp(-2 * theta_var),
        np.sin(2 * theta) * np.exp(-2 * theta_var)
    ]).reshape((3, 1))

    # calculate M to make use of eq. 35a
    innov = np.array(z - H @ x).reshape((2, 1))
    M = innov @ innov.T + H @ P @ H.T + sigma
    m11, m12, m21, m22 = M.flatten()

    # Build result
    # a-d represent the results from equations 35a-d
    a = np.array([m11, m22, -(m12 + m21)]).reshape((1, 3)) @ K
    b = np.array([m12, -m21, m11 - m22]).reshape((1, 3)) @ K
    c = np.array([m21, -m12, m11 - m22]).reshape((1, 3)) @ K
    d = np.array([m22, m11, m12 + m21]).reshape((1, 3)) @ K

    EqT = np.block([
        [a, b],
        [c, d]
    ])

    assert EqT.shape == (2, 2), "Result from equation 33e is of shape 2x2, something went wrong."
    return EqT


def rot_deriv(theta):
    """
    Returns the derivative of rot(theta) w.r.t theta
    :param theta: angle of orientation
    :return: Derivative of rot(theta) w.r.t. theta (2x2)
    """
    r = np.array([
        [-np.sin(theta), -np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]
    ])
    return r.reshape((2, 2))


def ComputeLXL(theta_bar, Sigma_theta_k, X):
    """
    https://github.com/Metu-Sensor-Fusion-Lab/Random-Matrix-Based-Extended-Target-Tracking-With-Orientation/blob/main/UpdateVB.m
    """
    a = X[0, 0]
    b = X[0, 1]
    c = X[1, 0]
    d = X[1, 1]

    LXL_bar = np.zeros((2, 2))
    LXL_bar[0, 0] = a * (1 + np.cos(2 * theta_bar) * np.exp(-2 * Sigma_theta_k)) + \
                    d * (1 - np.cos(2 * theta_bar) * np.exp(-2 * Sigma_theta_k)) - \
                    (c + b) * (np.sin(2 * theta_bar) * np.exp(-2 * Sigma_theta_k))

    LXL_bar[0, 1] = b * (1 + np.cos(2 * theta_bar) * np.exp(-2 * Sigma_theta_k)) - \
                    c * (1 - np.cos(2 * theta_bar) * np.exp(-2 * Sigma_theta_k)) + \
                    (a - d) * (np.sin(2 * theta_bar) * np.exp(-2 * Sigma_theta_k))

    LXL_bar[1, 0] = c * (1 + np.cos(2 * theta_bar) * np.exp(-2 * Sigma_theta_k)) - \
                    b * (1 - np.cos(2 * theta_bar) * np.exp(-2 * Sigma_theta_k)) + \
                    (a - d) * (np.sin(2 * theta_bar) * np.exp(-2 * Sigma_theta_k))

    LXL_bar[1, 1] = d * (1 + np.cos(2 * theta_bar) * np.exp(-2 * Sigma_theta_k)) + \
                    a * (1 - np.cos(2 * theta_bar) * np.exp(-2 * Sigma_theta_k)) + \
                    (c + b) * (np.sin(2 * theta_bar) * np.exp(-2 * Sigma_theta_k))

    LXL_bar = 0.5 * LXL_bar

    return LXL_bar
