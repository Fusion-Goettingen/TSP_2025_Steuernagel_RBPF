import numpy as np
from src.utilities.utils import rot

# Define the process noise covariance matrices
_Q_KINEMATIC_CV = np.diag([0.25, 0.25, 2.5, 2.5])
_Q_KINEMATIC_CV_MOTION = np.diag([5, 5, 2.5, 2.5])
_Q_KINEMATIC_CV_MOTIONVELO = np.diag([5, 5, 7.5, 7.5])

# Define the two types of trajectories:
#   "butterfly" (named after the shape of the trajectory)
#   "nimitz" (named after the type of object tracked in one of the first papers using the trajectory)
#   both additionally come with a "variable" variant with changing shape
_BUTTERFLY = dict(
    trajectory_type='butterfly',
    m_init=np.array([0, 0, 6.7, 6.7]),
    p_init=np.array([np.pi / 4, 5, 2]),
    P_kinematic=np.diag([2, 2, 0.5, 0.5]),
    P_shape=np.diag([0.5, 1, 1]),
    Q_kinematic=np.diag([1, 1, 2, 2]),
    Q_shape=np.diag([0.1, 0, 0]),
)

_BUTTERFLY_VARIABLE = dict(
    trajectory_type='butterfly',
    m_init=np.array([0, 0, 6.7, 6.7]),
    p_init=np.array([np.pi / 4, 5, 2]),
    P_kinematic=np.diag([2, 2, 0.5, 0.5]),
    P_shape=np.diag([0.5, 1, 1]),
    Q_kinematic=np.diag([1, 1, 2, 2]),
    Q_shape=np.diag([0.1, 0.15, 0.15]),
)

_NIMITZ = dict(
    trajectory_type='nimitz',
    m_init=np.array([0, 0, 10., -10.]),
    p_init=np.array([-np.pi / 4, 5, 2]),
    P_kinematic=np.diag([2, 2, 0.5, 0.5]),
    P_shape=np.diag([0.5, 1, 1]),
    Q_kinematic=np.diag([1, 1, 2, 2]),
    Q_shape=np.diag([0.1, 0, 0]),
)

_NIMITZ_VARIABLE = dict(
    trajectory_type='nimitz',
    m_init=np.array([0, 0, 10., -10.]),
    p_init=np.array([-np.pi / 4, 5, 2]),
    P_kinematic=np.diag([2, 2, 0.5, 0.5]),
    P_shape=np.diag([0.5, 1, 1]),
    Q_kinematic=np.diag([1, 1, 2, 2]),
    Q_shape=np.diag([0.1, 0.15, 0.15]),
)

# Define the measurement noise options
_HARD_ANISOTROPIC = dict(
    measurement_lambda=6,
    R=rot(np.pi / 4) @ np.diag([3, 1]) @ rot(np.pi / 4).T,
    min_measurements=1,
)
_EASY_ANISOTROPIC = dict(
    measurement_lambda=12,
    R=rot(np.pi / 4) @ np.diag([1.5, 2 / 3]) @ rot(np.pi / 4).T,
    min_measurements=1,
)

# Consolidate everything into a dict of settings dicts
SETTINGS = {
    "easy-random-butterfly": dict(
        rotation_mode='random',
        **_BUTTERFLY,
        **_EASY_ANISOTROPIC,
        fix_size=np.mean(np.diag(_BUTTERFLY["Q_shape"])[1:]) < 1e-3,
        approximate_measurements_with_gaussian=False,
    ),
    "hard-random-butterfly": dict(
        rotation_mode='random',
        **_BUTTERFLY,
        **_HARD_ANISOTROPIC,
        fix_size=np.mean(np.diag(_BUTTERFLY["Q_shape"])[1:]) < 1e-3,
        approximate_measurements_with_gaussian=False,
    ),
    "easy-random-nimitz": dict(
        rotation_mode='random',
        **_NIMITZ,
        **_EASY_ANISOTROPIC,
        fix_size=np.mean(np.diag(_BUTTERFLY["Q_shape"])[1:]) < 1e-3,
        approximate_measurements_with_gaussian=False,
    ),
    "easy-random-nimitz-b": dict(
        rotation_mode='random',
        **_NIMITZ,
        **_EASY_ANISOTROPIC,
        fix_size=np.mean(np.diag(_BUTTERFLY["Q_shape"])[1:]) < 1e-3,
        approximate_measurements_with_gaussian=False,
    ),
    "hard-random-nimitz": dict(
        rotation_mode='random',
        **_NIMITZ,
        **_HARD_ANISOTROPIC,
        fix_size=np.mean(np.diag(_BUTTERFLY["Q_shape"])[1:]) < 1e-3,
        approximate_measurements_with_gaussian=False,
    ),
    "easy-random-butterfly-variable": dict(
        rotation_mode='random',
        **_BUTTERFLY_VARIABLE,
        **_EASY_ANISOTROPIC,
        fix_size=np.mean(np.diag(_BUTTERFLY_VARIABLE["Q_shape"])[1:]) < 1e-3,
        approximate_measurements_with_gaussian=False,
    ),
    "easy-random-nimitz-variable": dict(
        rotation_mode='random',
        **_NIMITZ_VARIABLE,
        **_EASY_ANISOTROPIC,
        fix_size=np.mean(np.diag(_BUTTERFLY_VARIABLE["Q_shape"])[1:]) < 1e-3,
        approximate_measurements_with_gaussian=False,
    ),
    "hard-random-butterfly-variable": dict(
        rotation_mode='random',
        **_BUTTERFLY_VARIABLE,
        **_HARD_ANISOTROPIC,
        fix_size=np.mean(np.diag(_BUTTERFLY_VARIABLE["Q_shape"])[1:]) < 1e-3,
        approximate_measurements_with_gaussian=False,
    ),
    "hard-random-nimitz-variable": dict(
        rotation_mode='random',
        **_NIMITZ_VARIABLE,
        **_HARD_ANISOTROPIC,
        fix_size=np.mean(np.diag(_BUTTERFLY_VARIABLE["Q_shape"])[1:]) < 1e-3,
        approximate_measurements_with_gaussian=False,
    ),

    "easy-random-cv": dict(
        trajectory_type='cv',
        rotation_mode='random',
        **_EASY_ANISOTROPIC,
        m_init=np.array([0, 0, 10, 0]),
        p_init=np.array([0, 6, 3]),
        P_kinematic=np.diag([2, 2, 1, 1]),
        P_shape=np.diag([0.5, 1, 1]),
        Q_kinematic=_Q_KINEMATIC_CV,
        Q_shape=np.diag([0.1, 0, 0]),
        fix_size=True,
        approximate_measurements_with_gaussian=False,
    ),
    "hard-random-cv": dict(
        trajectory_type='cv',
        rotation_mode='random',
        **_HARD_ANISOTROPIC,
        m_init=np.array([0, 0, 10, 0]),
        p_init=np.array([0, 6, 3]),
        P_kinematic=np.diag([2, 2, 1, 1]),
        P_shape=np.diag([0.5, 1, 1]),
        Q_kinematic=_Q_KINEMATIC_CV,
        Q_shape=np.diag([0.1, 0, 0]),
        fix_size=True,
        approximate_measurements_with_gaussian=False,
    ),
    "easy-random-cv-motion": dict(
        trajectory_type='cv',
        rotation_mode='random',
        **_EASY_ANISOTROPIC,
        m_init=np.array([0, 0, 10, 0]),
        p_init=np.array([0, 6, 3]),
        P_kinematic=np.diag([2, 2, 1, 1]),
        P_shape=np.diag([0.5, 1, 1]),
        Q_kinematic=_Q_KINEMATIC_CV_MOTION,
        Q_shape=np.diag([0.1, 0, 0]),
        fix_size=True,
        approximate_measurements_with_gaussian=False,
    ),
    "easy-random-cv-motionvelo": dict(
        trajectory_type='cv',
        rotation_mode='random',
        **_EASY_ANISOTROPIC,
        m_init=np.array([0, 0, 10, 0]),
        p_init=np.array([0, 6, 3]),
        P_kinematic=np.diag([2, 2, 1, 1]),
        P_shape=np.diag([0.5, 1, 1]),
        Q_kinematic=_Q_KINEMATIC_CV_MOTIONVELO,
        Q_shape=np.diag([0.1, 0, 0]),
        fix_size=True,
        approximate_measurements_with_gaussian=False,
    ),

    "hard-random-cv-motion": dict(
        trajectory_type='cv',
        rotation_mode='random',
        **_HARD_ANISOTROPIC,
        m_init=np.array([0, 0, 10, 0]),
        p_init=np.array([0, 6, 3]),
        P_kinematic=np.diag([2, 2, 1, 1]),
        P_shape=np.diag([0.5, 1, 1]),
        Q_kinematic=_Q_KINEMATIC_CV_MOTION,
        Q_shape=np.diag([0.1, 0, 0]),
        fix_size=True,
        approximate_measurements_with_gaussian=False,
    ),
    "hard-random-cv-motionvelo": dict(
        trajectory_type='cv',
        rotation_mode='random',
        **_HARD_ANISOTROPIC,
        m_init=np.array([0, 0, 10, 0]),
        p_init=np.array([0, 6, 3]),
        P_kinematic=np.diag([2, 2, 1, 1]),
        P_shape=np.diag([0.5, 1, 1]),
        Q_kinematic=_Q_KINEMATIC_CV_MOTIONVELO,
        Q_shape=np.diag([0.1, 0, 0]),
        fix_size=True,
        approximate_measurements_with_gaussian=False,
    ),
}
