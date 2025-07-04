"""
Implements the MEM-RBPF
https://github.com/Fusion-Goettingen/TSP_2025_Steuernagel_RBPF

S. Steuernagel and M. Baum,
"Extended Object Tracking by Rao-Blackwellized Particle Filtering for Orientation Estimation,"
in IEEE Transactions on Signal Processing, doi: 10.1109/TSP.2025.3574689
"""
import numpy as np
from filterpy.monte_carlo import systematic_resample, stratified_resample, multinomial_resample, residual_resample

from src.utilities.utils import lse_norm
from src.trackers_elliptical.abstract_tracking import AbtractEllipticalTracker
from src.utilities.utils import rot
from src.utilities.utils import state_to_srs, srs_to_state


class MEMRBPF(AbtractEllipticalTracker):
    _c = 0.25  # scaling factor for ellipses

    def __init__(self,
                 m_init,
                 p_init,
                 P_kinematic_init,
                 P_shape_init,
                 R,
                 Q_kinematic,
                 Q_shape,
                 n_particles=100,
                 resampling_var: float = None,
                 rng: np.random.Generator = None,
                 time_step_length=1,
                 resampling_mode="multinomial",
                 ):
        self.rng = rng
        self.R = np.array(R)

        # Bound angles in [0, self.max_angles] - choose either pi or 2pi
        self.max_angle = 2 * np.pi
        self._n_observations = 0

        # define resampling function
        self.resampling_mode = str(resampling_mode).lower()
        if self.resampling_mode == "systematic":
            self._sample_particle_indices = systematic_resample
        elif self.resampling_mode == "stratified":
            self._sample_particle_indices = stratified_resample
        elif self.resampling_mode == "multinomial":
            self._sample_particle_indices = multinomial_resample
        elif self.resampling_mode == "residual":
            self._sample_particle_indices = residual_resample
        else:
            raise NotImplementedError(f"Resampling mode {resampling_mode} not supported!")

        # init kinematic KF
        self.x = np.array(m_init).astype(float)
        self.P = P_kinematic_init.astype(float)
        self.H = np.block([np.eye(2), np.zeros((2, 2))])
        self.F = np.array([
            [1, 0, time_step_length, 0],
            [0, 1, 0, time_step_length],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.Q = Q_kinematic

        # init shape PF
        self.resampling_var = Q_shape[0, 0] if resampling_var is None else resampling_var
        self.n_particles = n_particles

        # init weights
        self.weights = np.full(shape=(self.n_particles,), fill_value=1 / self.n_particles)
        self.log_weights = np.log(self.weights)

        # init angles
        # sample from prior
        self.theta = self.rng.normal(loc=p_init[0], scale=np.sqrt(P_shape_init[0, 0]), size=self.n_particles)
        self.theta = self.theta % self.max_angle

        # init shape filters
        # initial semi-axis = p_shape init
        self.axis = np.tile(p_init[1:], self.n_particles).reshape((self.n_particles, 2)).astype(float)
        # initial cov according to P_shape_init
        init_cov = P_shape_init[1:, 1:]
        self.covs = np.repeat(init_cov[np.newaxis, :, :], self.n_particles, axis=0).astype(float)
        self.F_shape = np.eye(2)  # axis transition matrix
        self.Q_shape = Q_shape[1:, 1:]  # axis process noise

    def kinematic_update(self, z, R):
        """
        Update the kinematic state

        :param z: kinematic pseudo-measurement
        :param R: covariance of the kinematic pseudo meas
        """
        # compute standard Kalman update
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        self.x += K @ (z - self.H @ self.x)
        self.P -= K @ self.H @ self.P

    def update(self, measurements: np.ndarray):
        """
        Perform the measurement update step of the filter

        :param measurements: Nx2 array of measurements for the current time step
        """
        # Kinematic KF update
        # determine shape matrix X if any number of measurements has been received so far
        if self._n_observations > 0:
            shape = self.get_shape_point_estimate()
            _rmat = rot(shape[0])
            X = self._c * _rmat @ np.diag((shape[1:]) ** 2) @ _rmat.T
        else:
            X = np.zeros((2, 2))
        # perform kinematic update using mean of measurements and its cov
        self.kinematic_update(
            z=np.average(measurements, axis=0),
            R=(self.R + X) / len(measurements)  # n measurements used for mean -> divide by n
        )

        # PF update
        # 1) Predict shape Kalman filters
        ...  # this was done in the predict/update cycle

        # 2) Sample from importance distribution
        angles_sampled_values = self.rng.normal(loc=np.zeros(self.n_particles),
                                                scale=np.sqrt(self.resampling_var))

        self.theta = (self.theta + angles_sampled_values) % self.max_angle

        # 3) Update weights
        # first: pre-compute the centered and axis-aligned measurements Z_tilde_stack
        m = (self.H @ self.x).reshape((1, 2))
        Z_centered = measurements - m  # center measurements on estimated mean
        rmat_stack = rot(-self.theta)  # Px2x2 array of rotation matrices (note '-theta' is required to align!)
        # rotate each meas. in the Mx2 array of measurements by the P rotation matrices, yielding a Px2x2 array
        Z_tilde_stack = np.einsum('nab,mb->nma',
                                  rmat_stack, Z_centered)

        # second: compute weights (except in first step)
        # first: compute the covariance matrices for the marginal measurement likelihood
        marginal_covs = np.zeros((self.n_particles, 2, 2))
        # diagonal entries based on scaled squared semi-axis lengths
        marginal_covs[:, 0, 0] = self._c * self.axis[:, 0] ** 2
        marginal_covs[:, 1, 1] = self._c * self.axis[:, 1] ** 2
        # rotate the shape (note transpose of rot. mat is encoded by switching in einsum)
        marginal_covs = np.einsum('pij,pjk,plk->pil',
                                  rot(self.theta), marginal_covs, rot(self.theta))
        # add the meas noise
        marginal_covs += self.R
        # add the rotated and scaled shape cov. (note transpose of rot. mat is encoded by switching in einsum)
        marginal_covs += np.einsum('pij,pjk,plk->pil',
                                   rot(self.theta), self._c * self.covs, rot(self.theta))
        # pre-compute the inverse matrices
        marginal_invs = np.linalg.inv(marginal_covs)

        # === Compute the log_likelihoods
        # note that the measurements used here are zero centered
        # log(gaussian pdf) consists of 3 parts
        #   - the first is constant so can be discarded (weights will be normalized anyway)
        #   - the second can be quickly computed using slogdet for stability
        sign, logabsdet = np.linalg.slogdet(marginal_covs)  # compute log of determinants of covariances
        logdet = (sign * logabsdet) * -0.5  # ^-0.5 becomes *-0.5 in log space
        #   - the third is exp(-0.5 * x^T @ C^-1 @ x) (for zero mean Gaussian)
        #   since we are in logspace, we discard the exp
        #   and then just compute a PxM array of likelihoods
        dist = -0.5 * np.einsum('pma,pab,pmb->pm',
                                np.repeat(Z_centered[np.newaxis, :, :], self.n_particles, axis=0),
                                marginal_invs,
                                np.repeat(Z_centered[np.newaxis, :, :], self.n_particles, axis=0))
        # add the logdet part, and sum over the measurements so that an array of shape (P,) remains for the lls
        ll = np.sum(logdet[:, np.newaxis] + dist, axis=1)

        # finally: normalize the weights and move out of logspace
        self.weights = lse_norm(ll)

        # 4) Update the shape Kalman filters
        # update the MEM Kalman filters for the semi-axis
        self._update_axis(Z_tilde_stack)

        # 5) resample
        self.resample()

        # Optional Check for negative axis somewhere
        # n_below_zero = np.sum(self.axis < 0)
        # point_est_below_zero = np.sum(self.get_shape_point_estimate()[1:] < 0)
        # if n_below_zero > 0 or point_est_below_zero > 0:
        #     print(f"Particle axis <0: {n_below_zero}-point est. {np.around(self.get_shape_point_estimate()[1:], 2)}")

        # ---
        self._n_observations += len(measurements)

    def _update_axis(self, Z_tilde_stack):
        # sequential update for each of the measurements
        for i in range(Z_tilde_stack.shape[1]):
            # create a Px2 list of the next measurement aligned using the respective particle orientation
            z_tilde = Z_tilde_stack[:, i, :]

            # create squared pseudo_meas (Px2)
            j = z_tilde ** 2

            # generate local frame uncertainty for each particle (rotate meas noise)
            # note that we use rot(-t).T = rot(t)
            A = np.einsum("pab, bc, pcd->pad",
                          rot(-self.theta), self.R, rot(self.theta))

            # generate moments
            E_j = np.diagonal(A, axis1=1, axis2=2) + \
                  self._c * (np.diagonal(self.covs, axis1=1, axis2=2) + self.axis ** 2)
            # pseudo meas cov
            U_jj = np.zeros((self.n_particles, 2, 2))  # Px2x2
            U_jj[:, 0, 0] = 2 * E_j[:, 0] ** 2
            U_jj[:, 1, 1] = 2 * E_j[:, 1] ** 2
            U_jj[:, 0, 1] = 2 * A[:, 0, 1] ** 2
            U_jj[:, 1, 0] = 2 * A[:, 1, 0] ** 2

            # pseudo_meas-state cross covariance
            U_jl = np.zeros((self.n_particles, 2, 2))  # Px2x2
            U_jl[:, 0, 0] = 2 * self._c * self.axis[:, 0] * self.covs[:, 0, 0]
            U_jl[:, 1, 1] = 2 * self._c * self.axis[:, 1] * self.covs[:, 1, 1]

            # pre-compute inverse of U_jj
            # note: linalg applies to the last 2 dims -> results is Px2x2 as desired
            inv_C_YY = np.linalg.inv(U_jj)

            # compute Kalman update for all particles using einsum
            self.axis += np.einsum('pab,pbc,pc->pa',
                                   U_jl, inv_C_YY, j - E_j)
            # U_jl transpose is encoded by using pdc instead of pcd
            self.covs -= np.einsum('pab,pbc,pdc->pad',
                                   U_jl, inv_C_YY, U_jl)

    def predict(self):
        """
        Perform the predict (time update) step of the filter
        """
        # predict kinematics
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # predict shape (note: axis is Nx2 hence the counterintuitive way of writing this)
        self.axis = self.axis @ self.F_shape.T
        # apply F_shape along whole axis and add Q (F @ C_i @ F.T + Q for each i in 0...n_particles-1)
        self.covs = (np.einsum('ab,nbc,cd->nad',
                               self.F_shape, self.covs, self.F_shape.T) + self.Q_shape[np.newaxis, :, :])

    def resample(self):
        """Perform resampling of the particles"""
        # pick n_particles-many random rows from the particle array, with replacement, with prob. according to weights
        sampled_indices = self._sample_particle_indices(self.weights)

        self.theta = self.theta[sampled_indices]
        self.axis = self.axis[sampled_indices]
        self.covs = self.covs[sampled_indices]

        self.weights = np.full(shape=(self.n_particles,), fill_value=1 / self.n_particles)
        self.log_weights = np.log(self.weights)

    def get_state(self) -> np.ndarray:
        """Get the current overall state estimate as ndarray of shape (7,)"""
        # kinematic state is simply the KF estimate
        kinematics = self.x
        extent = self.get_shape_point_estimate()
        # get_state is supposed to return full axis, double semi-axis estimate
        extent[1:] *= 2
        return np.array([*kinematics, *extent])

    def get_shape_point_estimate(self):
        """Compute the current point estimate of the shape as a 3d vector (theta, l, w)"""
        # get average extent in SquareRootSpace
        extents = np.array(
            np.hstack([self.theta.reshape((-1, 1)), self.axis])
        ).astype(float)
        srs_stack = np.array([state_to_srs(extents[i]) for i in range(len(extents))])
        srs_avg = np.average(srs_stack, axis=0, weights=self.weights)
        avg_extent = srs_to_state(srs_avg)

        # ensure consistency of semi-axis length (major-first notation, no impact on estimated ellipse)
        if avg_extent[2] > avg_extent[1]:
            avg_extent = avg_extent[0] + np.pi / 2, avg_extent[2], avg_extent[1]
        return np.array(avg_extent)

    def get_state_and_cov(self):
        cov = np.zeros((7, 7))
        cov[:4, :4] = self.P
        cov[4:, 4:] = ...
        raise NotImplementedError

    def set_R(self, R):
        """
        Update the measurement noise R
        :param R: new measurement noise
        """
        R = np.array(R)
        assert R.shape == self.R.shape
        self.R = R

    def get_state_array(self, with_weight=False):
        """
        Return all states as (n_particles, 7) array, or as (n_particles, 8) if the weight is to be appended
        :param with_weight: boolean indicating whether to return the particle weight too
        :return: array of particles (Px7 or Px8, depending on with_weight parameter)
        """
        kinematics = self.x
        if with_weight:
            return np.array([
                [
                    *kinematics,
                    self.theta[i],
                    *self.axis[i, :],
                    self.weights[i]
                ]
                for i in range(self.n_particles)
            ]).astype(float)
        else:
            return np.array([
                [
                    *kinematics,
                    self.theta[i],
                    *self.axis[i, :],
                ]
                for i in range(self.n_particles)
            ]).astype(float)
