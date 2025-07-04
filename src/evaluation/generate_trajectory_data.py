import numpy as np

from src.evaluation.elliptical_objects import get_ellipse_measurements, approximate_ellipse_measurements
from src.utilities.utils import rot


def get_cv_kinematics(rng: np.random.Generator,
                      m_init,
                      P_init,
                      Q_kin,
                      n_steps=20,
                      stationary=False
                      ):
    m = np.zeros((n_steps, 4))
    m[0, :] = rng.multivariate_normal(m_init, P_init)

    if stationary:
        F = np.eye(4)
    else:
        F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    for step_ix in range(1, n_steps):
        m[step_ix, :] = F @ m[step_ix - 1, :] + rng.multivariate_normal(mean=np.zeros(4), cov=Q_kin)

    return m


def get_butterfly_kinematics(rng: np.random.Generator,
                             m_init,
                             P_init,
                             Q_kin,
                             ):
    """speed should be roughly max(len, width) / 1.5 on both x and y"""
    F_cv = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    steps_in_turn = 6
    steps_in_long_line = 12
    steps_in_short_line = 5

    def perform_linear_step(state_to_update):
        return F_cv @ state_to_update

    def perform_rotation_step_right(state_to_update):
        rotation_angle = (np.pi / 4 + np.pi / 2) / steps_in_turn
        rmat = rot(-rotation_angle)
        state_to_update[2:4] = rmat @ state_to_update[2:4]
        return perform_linear_step(state_to_update)

    def perform_rotation_step_left(state_to_update):
        rotation_angle = (np.pi / 4 + np.pi / 2) / steps_in_turn
        rmat = rot(rotation_angle)
        state_to_update[2:4] = rmat @ state_to_update[2:4]
        return perform_linear_step(state_to_update)

    update_function_list = [
        *[perform_linear_step] * steps_in_long_line,  # long path
        *[perform_rotation_step_right] * steps_in_turn,  # first turn
        *[perform_linear_step] * steps_in_short_line,  # down again
        *[perform_rotation_step_right] * steps_in_turn,  # second turn
        *[perform_linear_step] * steps_in_long_line,  # long path
        *[perform_rotation_step_left] * steps_in_turn,  # third turn
        *[perform_linear_step] * steps_in_short_line,  # down again
        *[perform_rotation_step_left] * (steps_in_turn - 1),  # fourth turn
    ]

    kinematics = [rng.multivariate_normal(m_init, P_init)]
    for step_ix, update_function_list in enumerate(update_function_list):
        kinematics.append(update_function_list(kinematics[-1]))

    return np.vstack(kinematics)


def get_nimitz_kinematics(rng: np.random.Generator,
                          m_init,
                          P_init,
                          Q_kin,
                          ):
    F_cv = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    list_delta_theta = [
        *[0] * 6,  # start by going down-right for some time
        *[(np.pi / 4) / 3] * 3,  # rotate to turn right over a few steps
        *[0] * 8,  # go right a bit
        *[(np.pi / 2) / 5] * 5,  # sharper turn upwards
        *[0] * 4,
        *[(np.pi / 2) / 5] * 5,  # turn left
        *[0] * 11,
    ]

    kinematics = [rng.multivariate_normal(m_init, P_init)]
    for step_ix, delta_theta in enumerate(list_delta_theta):
        new_m = F_cv @ kinematics[-1]
        new_m[2:] = rot(delta_theta) @ new_m[2:]
        kinematics.append(new_m)

    return np.array(kinematics, dtype=object).astype(float)


def get_shape_from_kinematics(kinematics: np.ndarray,
                              object_length,
                              object_width,
                              rotation_mode: str,
                              fix_size: bool,
                              Q_shape: np.ndarray,
                              rng: np.random.Generator):
    n_steps = kinematics.shape[0]

    extents = np.zeros((n_steps, 3))

    # ORIENTATION
    def set_orientation_to_velo(extents):
        extents[0, 0] = np.arctan2(kinematics[0, 3], kinematics[0, 2]) + rng.normal(loc=0, scale=np.sqrt(Q_shape[0, 0]))
        for step_ix in range(1, n_steps):  # apply to every step
            effective_dynamics = kinematics[step_ix] - kinematics[step_ix - 1]
            velocity_component = np.arctan2(effective_dynamics[1], effective_dynamics[0])
            extents[step_ix, 0] = velocity_component
        extents[:, 0] = extents[:, 0] % (2 * np.pi)  # ensure everything is in [0, 2pi]
        return extents

    if rotation_mode == "velocity" or rotation_mode == 'velo':
        # fix rotation to velocity
        extents = set_orientation_to_velo(extents)
    elif rotation_mode == 'spinning' or rotation_mode == 'spin':
        extents[0, 0] = np.arctan2(kinematics[0, 3], kinematics[0, 2])  # init orientation = velocity
        spin_rate = np.pi / 16  # rate of change of orientation
        for step_ix in range(1, n_steps):  # apply to every step
            extents[step_ix, 0] = extents[step_ix - 1, 0] + spin_rate
        extents[:, 0] = extents[:, 0] % (2 * np.pi)  # ensure everything is in [0, 2pi]
    elif rotation_mode == 'random':
        extents[0, 0] = np.arctan2(kinematics[0, 3], kinematics[0, 2])  # init orientation = velocity
        for step_ix in range(1, n_steps):  # apply to every step
            extents[step_ix, 0] = rng.normal(loc=extents[step_ix - 1, 0], scale=np.sqrt(Q_shape[0, 0]))  # must use std!
        extents[:, 0] = extents[:, 0] % (2 * np.pi)  # ensure everything is in [0, 2pi]
    elif rotation_mode == 'velo+rand':
        extents = set_orientation_to_velo(extents)
        # add additional random noise to every time step
        noise = np.cumsum(rng.normal(loc=0, scale=np.sqrt(Q_shape[0, 0]), size=n_steps))
        extents[:, 0] += noise
        extents[:, 0] = extents[:, 0] % (2 * np.pi)  # ensure everything is in [0, 2pi]
    else:
        raise NotImplementedError(f"Rotation mode {rotation_mode} not implemented.")

    # SIZE
    if fix_size:
        extents[:, 1] = object_length
        extents[:, 2] = object_width
    else:
        extents[0, 1] = object_length
        extents[0, 2] = object_width

        for step_ix in range(1, n_steps):
            extents[step_ix, 1:] = rng.multivariate_normal(
                mean=extents[step_ix - 1, 1:], cov=Q_shape[1:, 1:]
            )

    extents[:, 1:] = np.abs(extents[:, 1:])  # ensure pos. axis
    return extents


def get_trajectory_data(
        trajectory_type,
        rotation_mode,
        rng: np.random.Generator,
        measurement_lambda,
        min_measurements,
        m_init,
        p_init,
        P_kinematic,
        P_shape,
        Q_kinematic,
        Q_shape,
        R,
        fix_size,
        approximate_measurements_with_gaussian
):
    if approximate_measurements_with_gaussian:
        get_measurements = approximate_ellipse_measurements
    else:
        get_measurements = get_ellipse_measurements

    # 1) GET KINEMATICS
    if trajectory_type == 'cv':
        kinematics = get_cv_kinematics(rng,
                                       m_init,
                                       P_kinematic,
                                       Q_kinematic)
    elif trajectory_type == 'butterfly':
        kinematics = get_butterfly_kinematics(rng, m_init, P_kinematic, Q_kinematic)
    elif trajectory_type == 'nimitz':
        kinematics = get_nimitz_kinematics(rng, m_init, P_kinematic, Q_kinematic)
    elif trajectory_type == 'stationary':
        kinematics = get_cv_kinematics(rng,
                                       m_init,
                                       P_kinematic,
                                       Q_kinematic,
                                       stationary=True)
    else:
        raise NotImplementedError(f"Trajectory type {trajectory_type} is not implemented.")

    # 2) GET SHAPE
    object_length, object_width = rng.multivariate_normal(mean=p_init[1:], cov=P_shape[1:, 1:]) * 2
    shape = get_shape_from_kinematics(kinematics,
                                      object_length,
                                      object_width,
                                      rotation_mode,
                                      fix_size,
                                      Q_shape,
                                      rng)

    true_states = np.hstack([kinematics, shape])

    # 3) SAMPLE MEASUREMENTS
    measurements = []
    for step_ix in range(len(true_states)):
        current_state = true_states[step_ix, :]
        measurements.append(
            get_measurements(loc=current_state[:2],
                             length=current_state[5],
                             width=current_state[6],
                             theta=current_state[4],
                             R=R,
                             n_measurements=np.max([min_measurements, rng.poisson(lam=measurement_lambda)]),
                             internal_RNG=rng)[1]
        )

    return true_states, measurements
