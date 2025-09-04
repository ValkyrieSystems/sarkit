import numpy as np
import numpy.typing as npt

import sarkit.wgs84


def compute_ric_basis_vectors(p_ric: npt.ArrayLike, v_ric: npt.ArrayLike):
    """Compute the orientation of an RIC coordinate frame relative to the input coordinate frame.

    Parameters
    ----------
    p_ric, v_ric : array_like
        Position and velocity in an input coordinate frame

    Returns
    -------
    u_r, u_i, u_c : (3,) ndarray
        Radial, in-track, and cross-track unit vectors that specify the RIC frame
    """

    u_r = p_ric / np.linalg.norm(p_ric, keepdims=True, axis=-1)
    c = np.cross(u_r, np.asarray(v_ric))
    u_c = c / np.linalg.norm(c, keepdims=True, axis=-1)
    u_i = np.cross(u_c, u_r)
    return u_r, u_i, u_c


def _compute_t_ecef_ricf(p_ecef, v_ecef):
    return np.stack(compute_ric_basis_vectors(p_ecef, v_ecef), axis=1)


def _compute_t_ecef_rici(p_ecef, v_ecef):
    v_eci = v_ecef + np.cross(
        [0, 0, sarkit.wgs84.NOMINAL_MEAN_ANGULAR_VELOCITY], p_ecef
    )
    return np.stack(compute_ric_basis_vectors(p_ecef, v_eci), axis=1)


def _compute_ricf_rotation_matrix(p_ecef, v_ecef):
    t_ecef_ricf = _compute_t_ecef_ricf(p_ecef, v_ecef)
    return np.block(
        [
            [t_ecef_ricf, np.zeros_like(t_ecef_ricf)],
            [np.zeros_like(t_ecef_ricf), t_ecef_ricf],
        ]
    )


def _compute_rici_rotation_matrix(p_ecef, v_ecef):
    t_ecef_rici = _compute_t_ecef_rici(p_ecef, v_ecef)
    omega_3 = np.array(
        [
            [0, sarkit.wgs84.NOMINAL_MEAN_ANGULAR_VELOCITY, 0],
            [-sarkit.wgs84.NOMINAL_MEAN_ANGULAR_VELOCITY, 0, 0],
            [0, 0, 0],
        ]
    )
    return np.block(
        [
            [t_ecef_rici, np.zeros_like(t_ecef_rici)],
            [omega_3 @ t_ecef_rici, t_ecef_rici],
        ]
    )


def compute_ecef_pv_transformation(p_ecef, v_ecef, frame):
    """Return the transformation matrix from ``frame`` to ECEF.

    Parameters
    ----------
    p_ecef, v_ecef : (3,) array_like
        Position and velocity in ECEF coordinates
    frame : {'ECF', 'RICF', 'RICI'}
        Name of coordinate frame

    Returns
    -------
    (6, 6) ndarray
        transformation matrix from ``frame`` to ECEF
    """
    if frame == "ECF":
        return np.eye(6)
    if frame == "RICF":
        return _compute_ricf_rotation_matrix(p_ecef, v_ecef)
    if frame == "RICI":
        return _compute_rici_rotation_matrix(p_ecef, v_ecef)
    raise ValueError(frame)
