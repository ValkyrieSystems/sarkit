import numpy as np
import numpy.typing as npt

import sarkit._constants
import sarkit.wgs84

from . import _params as params
from . import _sensitivity

C = sarkit._constants.speed_of_light


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


def compute_composite_error_no_apo_mono(
    proj_set_0: params.ProjectionSetsMono,
    sens_mat: _sensitivity.SensitivityMatricesMono,
    errorstat_params: params.ErrorStatParams,
) -> np.ndarray | None:
    """Compute composite COA slant plane error covariance for monostatic with no APOs.

    Parameters
    ----------
    proj_set_0 : ProjectionSetsMono
        Monostatic COA projection set for a projection pair: IL0 and PT0 that do not have APOs applied
    sens_mat : SensitivityMatricesMono
        Monostatic sensitivity matrices for projection pair: IL0 and PT0
    errorstat_params : ErrorStatParams
        IPDD Error Statistics Parameters

    Returns
    -------
    ndarray or None
        2x2 composite COA slant plane error covariance matrix if ErrorStatistics are provided, otherwise ``None``.
    """
    if errorstat_params.C_AIF_APV is None and errorstat_params.C_SCP_RGAZ is None:
        return None

    if errorstat_params.C_AIF_APV is None:
        # 12.3.1
        return errorstat_params.C_SCP_RGAZ

    # 12.2.2
    # (1)
    m_rrdot_apv = np.concatenate(
        (sens_mat.M_RRdot_delta_ARP, sens_mat.M_RRdot_delta_VARP), axis=1
    )

    # (2)
    t_ecef_aif = compute_ecef_pv_transformation(
        proj_set_0.ARP_COA, proj_set_0.VARP_COA, errorstat_params.AIF
    )

    # 12.3.2
    # (1)
    c_apv_rrdot = (
        m_rrdot_apv
        @ t_ecef_aif
        @ errorstat_params.C_AIF_APV
        @ (m_rrdot_apv @ t_ecef_aif).T
    )

    # (2)
    c_rb_rrdot = np.array([[errorstat_params.VAR_RB, 0], [0, 0]])

    # (3)
    crsf = -proj_set_0.R_COA
    crdotsf = -proj_set_0.Rdot_COA
    c_clk_sf_rrdot = errorstat_params.VAR_CLK_SF * np.array(
        [
            [crsf**2, crsf * crdotsf],
            [crsf * crdotsf, crdotsf**2],
        ]
    )

    # (4)
    c_trop_rrdot = np.array([[errorstat_params.VAR_TROP, 0], [0, 0]])

    # (5)
    c_iono_rrdot = np.array([[errorstat_params.VAR_IONO, 0], [0, 0]])

    # (6)
    c_ui_rrdot = sens_mat.M_RRdot_IL @ errorstat_params.C_UI @ sens_mat.M_RRdot_IL.T

    # (7)
    c_ilpt_rrdot = (
        c_apv_rrdot
        + c_rb_rrdot
        + c_clk_sf_rrdot
        + c_trop_rrdot
        + c_iono_rrdot
        + c_ui_rrdot
    )

    # (8)
    m_rgaz_rrdot = -sens_mat.M_SPXY_RRdot  # 12.2.1 (3)
    c_ilpt_rgaz = m_rgaz_rrdot @ c_ilpt_rrdot @ m_rgaz_rrdot.T

    return c_ilpt_rgaz


def compute_composite_error_apo_mono(
    sens_mat: _sensitivity.SensitivityMatricesMono,
    apoerrors: params.ApoErrorParams,
):
    """Compute composite COA slant plane error covariance for monostatic with APOs.

    Parameters
    ----------
    sens_mat : SensitivityMatricesMono
        Monostatic sensitivity matrices for projection pair: IL0 and PT0
    apoerrors : ApoErrorParams
        IPDD APO Error Parameters

    Returns
    -------
    ndarray or None
        2x2 composite COA slant plane error covariance matrix if APO ErrorStatistics are provided, otherwise ``None``.
    """
    has_components = apoerrors.C_APOM is not None
    if apoerrors.C_SCPAPO_RGAZ is not None and not has_components:
        # 12.3.3
        return apoerrors.C_SCPAPO_RGAZ

    if has_components:
        # 12.3.4
        # (1)
        m_rrdot_xrt = (C / 2) * np.array([[-1, 1], [0, 0]])

        # (2)
        m_rrdot_apom = np.concatenate(
            (sens_mat.M_RRdot_delta_ARP, sens_mat.M_RRdot_delta_VARP, m_rrdot_xrt),
            axis=1,
        )

        # (3)
        c_ilpt_rrdot = m_rrdot_apom @ apoerrors.C_APOM @ m_rrdot_apom.T

        # (4)
        m_rgaz_rrdot = -sens_mat.M_SPXY_RRdot  # 12.2.1 (3)
        c_ilpt_rgaz = m_rgaz_rrdot @ c_ilpt_rrdot @ m_rgaz_rrdot.T

        return c_ilpt_rgaz
    return None
