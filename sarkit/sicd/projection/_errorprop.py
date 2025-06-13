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
    comps = errorstat_params.component_mono
    if comps is None and errorstat_params.C_SCP_RGAZ is None:
        return None

    if comps is None:
        # 12.3.1
        return errorstat_params.C_SCP_RGAZ

    # 12.2.2
    # (1)
    m_rrdot_apv = np.concatenate(
        (sens_mat.M_RRdot_delta_ARP, sens_mat.M_RRdot_delta_VARP), axis=1
    )

    # (2)
    t_ecef_aif = compute_ecef_pv_transformation(
        proj_set_0.ARP_COA, proj_set_0.VARP_COA, comps.AIF
    )

    # 12.3.2
    # (1)
    c_apv_rrdot = (
        m_rrdot_apv @ t_ecef_aif @ comps.C_AIF_APV @ (m_rrdot_apv @ t_ecef_aif).T
    )

    # (2)
    c_rb_rrdot = np.array([[comps.VAR_RB, 0], [0, 0]])

    # (3)
    crsf = -proj_set_0.R_COA
    crdotsf = -proj_set_0.Rdot_COA
    c_clk_sf_rrdot = comps.VAR_CLK_SF * np.array(
        [
            [crsf**2, crsf * crdotsf],
            [crsf * crdotsf, crdotsf**2],
        ]
    )

    # (4)
    c_trop_rrdot = np.array([[comps.VAR_TROP, 0], [0, 0]])

    # (5)
    c_iono_rrdot = np.array([[comps.VAR_IONO, 0], [0, 0]])

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


def _proj_sens_params_bi(proj_set_0, sens_mat, t_scp_coa):
    """Relevant portions of 12.2 for bistatic"""
    m_rgaz_rrdot = -sens_mat.M_SPXY_RRdot  # 12.2.1 (3)

    # 12.2.3
    # (1)
    m_rrdot_xpv = np.concatenate(
        (sens_mat.M_RRdot_delta_Xmt, sens_mat.M_RRdot_delta_VXmt), axis=1
    )

    # (2)
    m_rrdot_rpv = np.concatenate(
        (sens_mat.M_RRdot_delta_Rcv, sens_mat.M_RRdot_delta_VRcv), axis=1
    )

    # 12.2.3
    # (5)
    m_rrdot_xtf = (C / 2) * np.array([[-1, proj_set_0.tx_COA - t_scp_coa], [0, 1]])
    m_rrdot_rtf = (C / 2) * np.array([[+1, -(proj_set_0.tr_COA - t_scp_coa)], [0, -1]])
    return m_rgaz_rrdot, m_rrdot_xpv, m_rrdot_rpv, m_rrdot_xtf, m_rrdot_rtf


def compute_composite_error_no_apo_bi(
    proj_set_0: params.ProjectionSetsBi,
    sens_mat: _sensitivity.SensitivityMatricesBi,
    errorstat_params: params.ErrorStatParams,
    t_scp_coa: float,
) -> np.ndarray | None:
    """Compute composite COA slant plane error covariance for bistatic with no APOs.

    Parameters
    ----------
    proj_set_0 : ProjectionSetsBi
        Bistatic COA projection set for a projection pair: IL0 and PT0 that do not have APOs applied
    sens_mat : SensitivityMatricesBi
        Bistatic sensitivity matrices for projection pair: IL0 and PT0
    errorstat_params : ErrorStatParams
        IPDD Error Statistics Parameters
    t_scp_coa : float
        SCP pixel center of aperture time in seconds since CollectStart

    Returns
    -------
    ndarray or None
        2x2 composite COA slant plane error covariance matrix if ErrorStatistics are provided, otherwise ``None``.
    """
    m_rgaz_rrdot, m_rrdot_xpv, m_rrdot_rpv, m_rrdot_xtf, m_rrdot_rtf = (
        _proj_sens_params_bi(
            proj_set_0,
            sens_mat,
            t_scp_coa,
        )
    )
    comps = errorstat_params.component_bi
    if comps is None and errorstat_params.C_SCP_RRdot is None:
        return None

    if comps is None:
        # 12.4.1
        c_ilpt_rrdot = errorstat_params.C_SCP_RRdot
        return m_rgaz_rrdot @ c_ilpt_rrdot @ m_rgaz_rrdot.T

    # 12.2.3
    # (3)
    t_ecef_xif = compute_ecef_pv_transformation(
        proj_set_0.Xmt_COA, proj_set_0.VXmt_COA, comps.XIF
    )

    # (4)
    t_ecef_rif = compute_ecef_pv_transformation(
        proj_set_0.Rcv_COA, proj_set_0.VRcv_COA, comps.RIF
    )

    # 12.4.2
    # (1)
    c_ecef_xpv = t_ecef_xif @ comps.C_XIF_XPV @ t_ecef_xif.T

    # (2)
    c_ecef_rpv = t_ecef_rif @ comps.C_RIF_RPV @ t_ecef_rif.T

    # (3)
    cc_ecef_xpv_rpv = t_ecef_xif @ comps.CC_XIF_RIF_XPV_RPV @ t_ecef_rif.T

    # (4)
    c_ecef_xpv_rpv = np.block(
        [
            [c_ecef_xpv, cc_ecef_xpv_rpv],
            [cc_ecef_xpv_rpv.T, c_ecef_rpv],
        ]
    )

    # (5)
    m_rrdot_xpv_rpv = np.concatenate((m_rrdot_xpv, m_rrdot_rpv), axis=1)

    # (6)
    c_xpv_rpv_rrdot = m_rrdot_xpv_rpv @ c_ecef_xpv_rpv @ m_rrdot_xpv_rpv.T

    # (7) - no op
    # (8)
    m_rrdot_xrtf = np.concatenate((m_rrdot_xtf, m_rrdot_rtf), axis=1)
    c_xrtf_rrdot = m_rrdot_xrtf @ comps.C_XRTF @ m_rrdot_xrtf.T

    # (9)
    m_rrdot_atm = (C / 2) * np.array([[1, 1], [0, 0]])  # 12.2.3 (6)
    c_atm_rrdot = m_rrdot_atm @ comps.C_ATM @ m_rrdot_atm.T

    # (10)
    c_ui_rrdot = sens_mat.M_RRdot_IL @ errorstat_params.C_UI @ sens_mat.M_RRdot_IL.T

    # (11)
    c_ilpt_rrdot = c_xpv_rpv_rrdot + c_xrtf_rrdot + c_atm_rrdot + c_ui_rrdot

    # (12)
    c_ilpt_rgaz = m_rgaz_rrdot @ c_ilpt_rrdot @ m_rgaz_rrdot.T

    return c_ilpt_rgaz


def compute_composite_error_apo_bi(
    proj_set_0: params.ProjectionSetsBi,
    sens_mat: _sensitivity.SensitivityMatricesBi,
    apoerrors: params.ApoErrorParams,
    t_scp_coa: float,
):
    """Compute composite COA slant plane error covariance for bistatic with APOs.

    Parameters
    ----------
    proj_set_0 : ProjectionSetsBi
        Bistatic COA projection set for a projection pair: IL0 and PT0 that do have APOs applied
    sens_mat : SensitivityMatricesBi
        Bistatic sensitivity matrices for projection pair: IL0 and PT0
    apoerrors : ApoErrorParams
        IPDD APO Error Parameters
    t_scp_coa : float
        SCP pixel center of aperture time in seconds since CollectStart

    Returns
    -------
    ndarray or None
        2x2 composite COA slant plane error covariance matrix if APO ErrorStatistics are provided, otherwise ``None``.
    """
    m_rgaz_rrdot, m_rrdot_xpv, m_rrdot_rpv, m_rrdot_xtf, m_rrdot_rtf = (
        _proj_sens_params_bi(
            proj_set_0,
            sens_mat,
            t_scp_coa,
        )
    )
    has_components = apoerrors.C_APOXR is not None
    if apoerrors.C_SCPAPO_RRdot is not None and not has_components:
        # 12.4.3
        c_ilpt_rrdot = apoerrors.C_SCPAPO_RRdot
        return m_rgaz_rrdot @ c_ilpt_rrdot @ m_rgaz_rrdot.T

    if has_components:
        # 12.4.4
        # (1)
        m_rrdot_apoxr = np.concatenate(
            [m_rrdot_xpv, m_rrdot_xtf, m_rrdot_rpv, m_rrdot_rtf], axis=1
        )

        # (2)
        c_ilpt_rrdot = m_rrdot_apoxr @ apoerrors.C_APOXR @ m_rrdot_apoxr.T

        # (3)
        c_ilpt_rgaz = m_rgaz_rrdot @ c_ilpt_rrdot @ m_rgaz_rrdot.T

        return c_ilpt_rgaz
    return None


def compute_i2s_error(
    c_ilpt_rgaz: npt.ArrayLike,
    c_il_sel: npt.ArrayLike,
    var_hae: float,
    sens_mats: _sensitivity.SensitivityMatricesLike,
) -> np.ndarray:
    """Compute the image-to-scene projection error statistics for a projection pair: IL0 and PT0

    Parameters
    ----------
    c_ilpt_rgaz : (2, 2) array_like
        Predicted error covariance for composite RGAZ image error
    c_il_sel : (2, 2) array_like
        Predicted error covariance for image location
    var_hae : float
        Surface height error variance
    sens_mats : SensitivityMatricesLike
        Sensitivity matrices for projection pair: IL0 and PT0

    Returns
    -------
    (3, 3) ndarray
        image-to-scene projection error covariance matrix in ECEF
    """
    c_ilpt_rgaz = np.asarray(c_ilpt_rgaz)
    c_il_sel = np.asarray(c_il_sel)

    # 12.5
    # (1)
    mil_spxy_rgaz = -np.eye(2)  # 12.2.1 (4)
    c_rgaz_gpxy = (
        sens_mats.M_GPXY_SPXY
        @ mil_spxy_rgaz
        @ c_ilpt_rgaz
        @ (sens_mats.M_GPXY_SPXY @ mil_spxy_rgaz).T
    )
    c_rgaz_pt = sens_mats.M_PT_GPXY @ c_rgaz_gpxy @ sens_mats.M_PT_GPXY.T

    # (2)
    c_il_sel_gpxy = sens_mats.M_GPXY_IL @ c_il_sel @ sens_mats.M_GPXY_IL.T
    c_il_sel_pt = sens_mats.M_PT_GPXY @ c_il_sel_gpxy @ sens_mats.M_PT_GPXY.T

    # (3) unused
    # (4)
    c_hae_pt = sens_mats.MIL_PT_HAE @ sens_mats.MIL_PT_HAE.T * var_hae

    # (5)
    c_pt = c_rgaz_pt + c_il_sel_pt + c_hae_pt

    return c_pt
