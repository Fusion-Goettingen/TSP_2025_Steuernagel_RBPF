import numpy as np
import re
from scipy.stats import foldnorm

from src.evaluation.scenario_definitions import SETTINGS
from src.trackers_elliptical.principal_axes_kf import PrincipalAxesKalmanFilter
from src.trackers_elliptical.memekf import TrackerMEMEKF
from src.trackers_elliptical.mem_rbpf import MEMRBPF
from src.trackers_elliptical.vbrm import TrackerVBRM


class EvaluationManager:
    """
    An instance of this class can be used to easily get parameterized trackers for the same settings.
    """
    def __init__(self, mode, seed):
        try:
            self.settings = SETTINGS[mode]
        except KeyError:
            raise ValueError(f"Unknown mode '{mode}'!")

        # =====
        self.mode = mode
        self.default_n_particles = 50
        self.default_resampling_var = None
        self.seed = seed

    def get_settings(self):
        return self.settings

    def get_tracker_dict(self,
                         tracker_names,
                         settings=None):
        if settings is None:
            settings = self.get_settings()

        if type(tracker_names) == str:
            tracker_names = [tracker_names]

        trackers = {}  # dict of trackers to be filled

        # folded normal matching for variance along diagonal of Q_shape
        Q_shape = np.array(settings["Q_shape"])  # avoid overwriting reference!
        for i in [1, 2]:
            if Q_shape[i, i] > 0:
                Q_shape[i, i] = foldnorm(
                    c=0,  # c=|mean|/stddev = 0
                    loc=0,  # zero-mean process noise
                    scale=np.sqrt(Q_shape[i, i])  # scale = std. dev. of original Gaussian
                ).var()  # get variance

        # MEM-EKF*
        kwargs_memekf = dict(
            m_init=settings["m_init"],
            p_init=settings["p_init"],
            C_m_init=settings["P_kinematic"],
            C_p_init=settings["P_shape"],
            R=settings["R"],
            Q=settings["Q_kinematic"],
            Q_extent=Q_shape
        )

        # PAKF
        kwargs_pakf = dict(
            m_init=settings["m_init"],
            p_init=settings["p_init"],
            C_x_init=settings["P_kinematic"],
            C_p_init=settings["P_shape"],
            R=settings["R"],
            R_p=None,
            Q=settings["Q_kinematic"],
            Q_extent=Q_shape
        )

        # MEM-RBPF basis
        kwargs_memrbpf = dict(
            m_init=settings["m_init"],
            p_init=settings["p_init"],
            P_kinematic_init=settings["P_kinematic"],
            P_shape_init=settings["P_shape"],
            R=settings["R"],
            Q_kinematic=settings["Q_kinematic"],
            Q_shape=Q_shape,
            resampling_var=self.default_resampling_var,
        )

        # === REFERENCE METHODS
        if "MEM-EKF*" in tracker_names:
            trackers["MEM-EKF*"] = {"instance": TrackerMEMEKF(**kwargs_memekf),
                                    "color": 'C0'
                                    }

        if "VBRM" in tracker_names:
            trackers["VBRM"] = {
                "instance": TrackerVBRM(
                    m_init=settings["m_init"],
                    p_init=settings["p_init"],
                    P_init=settings["P_kinematic"],
                    gamma=0.98 if settings["fix_size"] else 0.9,
                    l_max=10,
                    Q=settings["Q_kinematic"],
                    R=settings["R"],
                    init_theta_var=1,
                    q_theta=0.15,
                    alpha_init=2,
                ),
                "color": 'C1',
            }

        if "PAKF" in tracker_names:
            trackers["PAKF"] = {
                "instance": PrincipalAxesKalmanFilter(**kwargs_pakf),
                "color": 'C3',
            }

        # === DEFAULT MEM-RBPF
        if "MEM-RBPF" in tracker_names:
            trackers["MEM-RBPF"] = {
                "instance": MEMRBPF(
                    n_particles=self.default_n_particles,
                    rng=np.random.default_rng(self.seed),
                    resampling_mode="systematic",
                    **kwargs_memrbpf
                ),
                "color": "C2"
            }

        # === 'MEM-RBPF #X' to get for different n_particles
        for s in tracker_names:
            pattern = r'^MEM-RBPF #(\d+)$'  # match strings such as 'MEM-RPBF #42' (for 42 particles)
            if re.match(pattern, s) is not None:
                n_p = int(re.match(pattern, s).group(1))  # number of particles
                trackers[f"MEM-RBPF [P={n_p}]"] = {
                    "instance": MEMRBPF(
                        n_particles=n_p,
                        rng=np.random.default_rng(self.seed),
                        resampling_mode="systematic",
                        **kwargs_memrbpf
                    ),
                    "color": np.random.default_rng(1).random(3)  # get 3 random floats for the color from a fixed seed
                }

        # ==== VARIANTS OF RESAMPLING
        if "MEM-RBPF [multinomial]" in tracker_names:
            trackers["MEM-RBPF [multinomial]"] = {
                "instance": MEMRBPF(
                    n_particles=self.default_n_particles,
                    rng=np.random.default_rng(self.seed),
                    resampling_mode="multinomial",
                    **kwargs_memrbpf
                ),
                "color": "C1"
            }
        if "MEM-RBPF [stratified]" in tracker_names:
            trackers["MEM-RBPF [stratified]"] = {
                "instance": MEMRBPF(
                    n_particles=self.default_n_particles,
                    rng=np.random.default_rng(self.seed),
                    resampling_mode="stratified",
                    **kwargs_memrbpf
                ),
                "color": "C2"
            }
        if "MEM-RBPF [systematic]" in tracker_names:
            trackers["MEM-RBPF [systematic]"] = {
                "instance": MEMRBPF(
                    n_particles=self.default_n_particles,
                    rng=np.random.default_rng(self.seed),
                    resampling_mode="systematic",
                    **kwargs_memrbpf
                ),
                "color": "C3"
            }
        if "MEM-RBPF [residual]" in tracker_names:
            trackers["MEM-RBPF [residual]"] = {
                "instance": MEMRBPF(
                    n_particles=self.default_n_particles,
                    rng=np.random.default_rng(self.seed),
                    resampling_mode="residual",
                    **kwargs_memrbpf
                ),
                "color": "C4"
            }

        # return everything
        return trackers
