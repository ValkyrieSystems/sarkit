import dataclasses
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

import sarkit._constants
import sarkit.wgs84
from sarkit.sicd.projection import _calc
from sarkit.sicd.projection import _params as params

C = sarkit._constants.c

# TODO: encouraged to migrate to type statements instead of TypeAlias in python 3.12
SensitivityMatricesLike: TypeAlias = "SensitivityMatricesMono | SensitivityMatricesBi"


@dataclasses.dataclass(kw_only=True)
class SlantPlaneSensitivityMatrices:
    """Sensitivity matrices that relate changes in slant plane projection point. (Table 11-2)"""

    M_SPXY_PT: np.ndarray
    M_SPXY_GPXY: np.ndarray
    M_GPXY_SPXY: np.ndarray
    M_PT_GPXY: np.ndarray
    MIL_PT_HAE: np.ndarray
    M_RRdot_SPXY: np.ndarray
    M_SPXY_RRdot: np.ndarray

    def __post_init__(self):
        assert self.M_SPXY_PT.shape == (2, 3)
        assert self.M_SPXY_GPXY.shape == (2, 2)
        assert self.M_GPXY_SPXY.shape == (2, 2)
        assert self.M_PT_GPXY.shape == (3, 2)
        assert self.MIL_PT_HAE.shape == (3, 1)
        assert self.M_RRdot_SPXY.shape == (2, 2)
        assert self.M_SPXY_RRdot.shape == (2, 2)


@dataclasses.dataclass(kw_only=True)
class ImageLocationSensitivityMatrices:
    """Sensitivity matrices that relate changes in image grid location. (Table 11-3)"""

    M_IL_PT: np.ndarray
    M_GPXY_IL: np.ndarray
    M_SPXY_IL: np.ndarray
    M_IL_SPXY: np.ndarray
    M_RRdot_IL: np.ndarray
    M_IL_RRdot: np.ndarray

    def __post_init__(self):
        assert self.M_IL_PT.shape == (2, 3)
        assert self.M_GPXY_IL.shape == (2, 2)
        assert self.M_SPXY_IL.shape == (2, 2)
        assert self.M_IL_SPXY.shape == (2, 2)
        assert self.M_RRdot_IL.shape == (2, 2)
        assert self.M_IL_RRdot.shape == (2, 2)


@dataclasses.dataclass(kw_only=True)
class PosVelSensitivityMatricesMono:
    """Sensitivity matrices that relate changes in monostatic ARP position/velocity to range/range rate. (Table 11-4)"""

    M_RRdot_delta_ARP: np.ndarray
    M_RRdot_delta_VARP: np.ndarray

    def __post_init__(self):
        assert self.M_RRdot_delta_ARP.shape == (2, 3)
        assert self.M_RRdot_delta_VARP.shape == (2, 3)


@dataclasses.dataclass(kw_only=True)
class PosVelSensitivityMatricesBi:
    """Sensitivity matrices that relate changes in bistatic APC positions/velocities to range/range rate. (Table 11-5)"""

    M_RRdot_delta_Xmt: np.ndarray
    M_RRdot_delta_VXmt: np.ndarray
    M_RRdot_delta_Rcv: np.ndarray
    M_RRdot_delta_VRcv: np.ndarray

    def __post_init__(self):
        assert self.M_RRdot_delta_Xmt.shape == (2, 3)
        assert self.M_RRdot_delta_VXmt.shape == (2, 3)
        assert self.M_RRdot_delta_Rcv.shape == (2, 3)
        assert self.M_RRdot_delta_VRcv.shape == (2, 3)


@dataclasses.dataclass(kw_only=True)
class SensitivityMatricesMono(
    SlantPlaneSensitivityMatrices,
    ImageLocationSensitivityMatrices,
    PosVelSensitivityMatricesMono,
):
    """Sensitivity Matrices from IPDD for a monostatic image"""


@dataclasses.dataclass(kw_only=True)
class SensitivityMatricesBi(
    SlantPlaneSensitivityMatrices,
    ImageLocationSensitivityMatrices,
    PosVelSensitivityMatricesBi,
):
    """Sensitivity Matrices from IPDD for a bistatic image"""


@dataclasses.dataclass(kw_only=True)
class ProjGeomParamsMono:
    """Set of projection geometry parameters for Collect Type = Monostatic"""

    uPT: np.ndarray  # noqa: N815
    uPTDot: np.ndarray  # noqa: N815
    VM_0: float
    cos_DCA: float  # noqa: N815
    sin_DCA: float  # noqa: N815
    tan_DCA: float  # noqa: N815


def compute_proj_geom_params_mono(
    proj_set_0: params.ProjectionSetsMono, pt0: npt.ArrayLike
) -> ProjGeomParamsMono:
    """Compute set of projection geometry parameters for Collect Type = Monostatic

    Parameters
    ----------
    proj_set_0 : ProjectionSetsMono
        Monostatic COA projection set for a projection pair: IL0 and PT0
    pt0 : (3,) array_like
        Scene point with ECEF (WGS 84 cartesian) X, Y, Z components in meters

    Returns
    -------
    ProjGeomParamsMono
        Set of monostatic-only projection geometry parameters
    """
    pt0 = np.asarray(pt0)
    # (1)
    u_pt = (proj_set_0.ARP_COA - pt0) / proj_set_0.R_COA
    u_pt_dot = (proj_set_0.VARP_COA - proj_set_0.Rdot_COA * u_pt) / proj_set_0.R_COA

    # (2)
    vm0 = np.linalg.norm(proj_set_0.VARP_COA, axis=-1)
    cos_dca = -proj_set_0.Rdot_COA / vm0
    sin_dca = np.sqrt(1 - cos_dca**2)
    tan_dca = sin_dca / cos_dca

    return ProjGeomParamsMono(
        uPT=u_pt,
        uPTDot=u_pt_dot,
        VM_0=vm0,
        cos_DCA=cos_dca,
        sin_DCA=sin_dca,
        tan_DCA=tan_dca,
    )


@dataclasses.dataclass(kw_only=True)
class ProjGeomParamsBi:
    """Set of projection geometry parameters for Collect Type = Bistatic"""

    R_Xmt_0coa: float
    Rdot_Xmt_0coa: float
    R_Rcv_0coa: float
    Rdot_Rcv_0coa: float
    uXmt: np.ndarray  # noqa: N815
    uXmtDot: np.ndarray  # noqa: N815
    uRcv: np.ndarray  # noqa: N815
    uRcvDot: np.ndarray  # noqa: N815
    bP: np.ndarray  # noqa: N815
    bPDot: np.ndarray  # noqa: N815


def compute_proj_geom_params_bi(
    proj_set_0: params.ProjectionSetsBi, pt0: npt.ArrayLike
) -> ProjGeomParamsBi:
    """Compute set of projection geometry parameters for Collect Type = Bistatic

    Parameters
    ----------
    proj_set_0 : ProjectionSetsBi
        Bistatic COA projection set for a projection pair: IL0 and PT0
    pt0 : (3,) array_like
        Scene point with ECEF (WGS 84 cartesian) X, Y, Z components in meters

    Returns
    -------
    ProjGeomParamsBi
        Set of bistatic-only projection geometry parameters
    """
    pt0 = np.asarray(pt0)
    # (1)
    r_xmt_0coa = np.linalg.norm(proj_set_0.Xmt_COA - pt0)
    u_xmt = (proj_set_0.Xmt_COA - pt0) / r_xmt_0coa
    rdot_xmt_0coa = np.dot(proj_set_0.VXmt_COA, u_xmt)
    u_xmtdot = (proj_set_0.VXmt_COA - rdot_xmt_0coa * u_xmt) / r_xmt_0coa

    # (2)
    r_rcv_0coa = np.linalg.norm(proj_set_0.Rcv_COA - pt0)
    u_rcv = (proj_set_0.Rcv_COA - pt0) / r_rcv_0coa
    rdot_rcv_0coa = np.dot(proj_set_0.VRcv_COA, u_rcv)
    u_rcvdot = (proj_set_0.VRcv_COA - rdot_rcv_0coa * u_rcv) / r_rcv_0coa

    # (3)
    bp = (u_xmt + u_rcv) / 2.0
    bpdot = (u_xmtdot + u_rcvdot) / 2.0

    return ProjGeomParamsBi(
        R_Xmt_0coa=float(r_xmt_0coa),
        Rdot_Xmt_0coa=rdot_xmt_0coa,
        R_Rcv_0coa=float(r_rcv_0coa),
        Rdot_Rcv_0coa=rdot_rcv_0coa,
        uXmt=u_xmt,
        uXmtDot=u_xmtdot,
        uRcv=u_rcv,
        uRcvDot=u_rcvdot,
        bP=bp,
        bPDot=bpdot,
    )


def _get_proj_parameters(
    proj_metadata: params.MetadataParams,
    pt0: npt.ArrayLike | None = None,
    u_gpn0: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an ECF scene point and unit normal to scene surface"""
    if pt0 is None:
        pt0 = proj_metadata.SCP
    pt0 = np.asarray(pt0)
    pt0_lat, pt0_lon = sarkit.wgs84.cartesian_to_geodetic(pt0)[:2]
    u_up0 = np.stack(
        (
            np.cos(np.deg2rad(pt0_lat)) * np.cos(np.deg2rad(pt0_lon)),
            np.cos(np.deg2rad(pt0_lat)) * np.sin(np.deg2rad(pt0_lon)),
            np.sin(np.deg2rad(pt0_lat)),
        )
    )
    if u_gpn0 is None:
        u_gpn0 = u_up0
    u_gpn0 = np.asarray(u_gpn0)
    assert np.dot(u_gpn0, u_up0) > 0
    assert pt0.shape == (3,)
    assert u_gpn0.shape == (3,)
    return pt0, u_gpn0


def compute_slant_plane_sensitivity_matrices(
    proj_metadata: params.MetadataParams,
    pt0: npt.ArrayLike | None = None,
    u_gpn0: npt.ArrayLike | None = None,
) -> SlantPlaneSensitivityMatrices:
    """Compute the defined slant plane sensitivity matrices.

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    pt0 : array_like, optional
        ECF scene point. Defaults to SCP.
    u_gpn0 : array_like, optional
        unit normal to scene surface at pt0. Defaults to ETP normal at pt0.

    Returns
    -------
    SlantPlaneSensitivityMatrices
    """

    pt0, u_gpn0 = _get_proj_parameters(proj_metadata, pt0, u_gpn0)

    il0, _, _ = _calc.scene_to_image(proj_metadata, pt0)
    proj_set_0 = _calc.compute_projection_sets(proj_metadata, il0)

    # Projection Geometry Parameters
    geom_params: ProjGeomParamsMono | ProjGeomParamsBi
    if isinstance(proj_set_0, params.ProjectionSetsMono):
        geom_params = compute_proj_geom_params_mono(proj_set_0, pt0)
    else:
        geom_params = compute_proj_geom_params_bi(proj_set_0, pt0)

    # SPC & GPC Parameters
    # (1)
    if isinstance(geom_params, ProjGeomParamsMono):
        spx = geom_params.uPT
        spz = proj_metadata.LOOK * np.cross(geom_params.uPT, geom_params.uPTDot)
    else:
        spx = geom_params.bP
        spz = proj_metadata.LOOK * np.cross(geom_params.bP, geom_params.bPDot)

    # (2)
    u_spx = spx / np.linalg.norm(spx)
    u_spz = spz / np.linalg.norm(spz)
    u_spy = np.cross(u_spz, u_spx)

    # (3)
    u_gpz = u_gpn0
    gpy = np.cross(u_gpz, u_spx)
    u_gpy = gpy / np.linalg.norm(gpy)
    u_gpx = np.cross(u_gpy, u_gpz)

    # (4)
    cos_graz = np.dot(u_spx, u_gpx)
    sin_graz = np.dot(u_spx, u_gpz)
    cos_twst = np.dot(u_spy, u_gpy)
    sin_twst = -np.dot(u_spz, u_gpy)
    tan_graz = sin_graz / cos_graz
    tan_twst = sin_twst / cos_twst

    # Slant Plane Sensitivity Matrices
    # (1)
    m_spxy_pt = np.stack((u_spx, u_spy))

    # (2)
    m_spxy_gpxy = np.asarray([[cos_graz, 0], [-sin_graz * sin_twst, cos_twst]])

    # (3)
    m_gpxy_spxy = np.asarray([[1 / cos_graz, 0], [tan_graz * tan_twst, 1 / cos_twst]])

    # (4)
    m_pt_gpxy = np.stack((u_gpx, u_gpy)).T

    # (5)
    u_up0 = _get_proj_parameters(proj_metadata, pt0)[1]
    spz_sf = np.dot(u_up0, u_gpz) / np.dot(u_spz, u_gpz)
    mil_pt_hae = spz_sf * u_spz.reshape((3, 1))

    # (6)
    if isinstance(geom_params, ProjGeomParamsMono):
        p = geom_params.uPT
        pdot = geom_params.uPTDot
    else:
        p = geom_params.bP
        pdot = geom_params.bPDot

    # (7)
    m_rrdot_spxy = -np.array(
        [[np.dot(p, u_spx), 0.0], [np.dot(pdot, u_spx), np.dot(pdot, u_spy)]]
    )

    # (8)
    m_spxy_rrdot = np.linalg.inv(m_rrdot_spxy)

    return SlantPlaneSensitivityMatrices(
        M_SPXY_PT=m_spxy_pt,
        M_SPXY_GPXY=m_spxy_gpxy,
        M_GPXY_SPXY=m_gpxy_spxy,
        M_PT_GPXY=m_pt_gpxy,
        MIL_PT_HAE=mil_pt_hae,
        M_RRdot_SPXY=m_rrdot_spxy,
        M_SPXY_RRdot=m_spxy_rrdot,
    )


def compute_image_location_sensitivity_matrices(
    proj_metadata: params.MetadataParams,
    pt0: npt.ArrayLike | None = None,
    u_gpn0: npt.ArrayLike | None = None,
    delta_xrow: float | None = None,
    delta_ycol: float | None = None,
) -> ImageLocationSensitivityMatrices:
    """Compute the defined image location sensitivity matrices

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    pt0 : array_like, optional
        ECF scene point. Defaults to SCP.
    u_gpn0 : array_like, optional
        unit normal to scene surface at pt0. Defaults to ETP normal at pt0.
    delta_xrow : float, optional
        row coordinate increment (m).  Defaults to min(Row_SS, 1.0).
    delta_ycol : float, optional
        col coordinate increment (m).  Defaults to min(Col_SS, 1.0).

    Returns
    -------
    ImageLocationSensitivityMatrices
    """

    pt0, u_gpn0 = _get_proj_parameters(proj_metadata, pt0, u_gpn0)

    if delta_xrow is None:
        delta_xrow = min(1.0, proj_metadata.Row_SS)
    if delta_ycol is None:
        delta_ycol = min(1.0, proj_metadata.Col_SS)
    assert np.isscalar(delta_xrow)
    assert np.isscalar(delta_ycol)

    il0, _, _ = _calc.scene_to_image(proj_metadata, pt0)
    proj_set_0 = _calc.compute_projection_sets(proj_metadata, il0)

    # Projection Geometry Parameters
    proj_parameters: ProjGeomParamsMono | ProjGeomParamsBi
    if isinstance(proj_set_0, params.ProjectionSetsMono):
        proj_parameters = compute_proj_geom_params_mono(proj_set_0, pt0)
    else:
        proj_parameters = compute_proj_geom_params_bi(proj_set_0, pt0)

    sp_mats = compute_slant_plane_sensitivity_matrices(proj_metadata, pt0, u_gpn0)

    # Image Location Sensitivity Matrices
    # (1)
    il1x = il0 + [delta_xrow, 0]

    # (2)
    il1y = il0 + [0, delta_ycol]

    proj_set_1x = _calc.compute_projection_sets(proj_metadata, il1x)
    proj_set_1y = _calc.compute_projection_sets(proj_metadata, il1y)

    if isinstance(proj_set_1x, params.ProjectionSetsMono):
        assert isinstance(proj_set_0, params.ProjectionSetsMono)
        assert isinstance(proj_set_1y, params.ProjectionSetsMono)
        assert isinstance(proj_parameters, ProjGeomParamsMono)
        # Monostatic delta RRdot
        # (2)
        delta_arp1x_coa = proj_set_1x.ARP_COA - proj_set_0.ARP_COA
        delta_varp1x_coa = proj_set_1x.VARP_COA - proj_set_0.VARP_COA

        # (3)
        delta_r_1x = (
            proj_set_1x.R_COA
            - proj_set_0.R_COA
            - np.dot(delta_arp1x_coa, proj_parameters.uPT)
        )
        delta_rdot_1x = (
            proj_set_1x.Rdot_COA
            - proj_set_0.Rdot_COA
            - (
                np.dot(delta_arp1x_coa, proj_parameters.uPTDot)
                + np.dot(delta_varp1x_coa, proj_parameters.uPT)
            )
        )

        # (4) - Done above
        # (5)
        delta_arp1y_coa = proj_set_1y.ARP_COA - proj_set_0.ARP_COA
        delta_varp1y_coa = proj_set_1y.VARP_COA - proj_set_0.VARP_COA

        # (6)
        delta_r_1y = (
            proj_set_1y.R_COA
            - proj_set_0.R_COA
            - np.dot(delta_arp1y_coa, proj_parameters.uPT)
        )
        delta_rdot_1y = (
            proj_set_1y.Rdot_COA
            - proj_set_0.Rdot_COA
            - (
                np.dot(delta_arp1y_coa, proj_parameters.uPTDot)
                + np.dot(delta_varp1y_coa, proj_parameters.uPT)
            )
        )

        # (7)
        m_rrdot_il = np.array(
            [
                [delta_r_1x / delta_xrow, delta_r_1y / delta_ycol],
                [delta_rdot_1x / delta_xrow, delta_rdot_1y / delta_ycol],
            ]
        )
        m_il_rrdot = np.linalg.inv(m_rrdot_il)
    else:
        assert isinstance(proj_set_0, params.ProjectionSetsBi)
        assert isinstance(proj_set_1y, params.ProjectionSetsBi)
        assert isinstance(proj_parameters, ProjGeomParamsBi)
        # Bistatic
        # (1) - Done above
        # (2)
        delta_xmt_1xcoa = proj_set_1x.Xmt_COA - proj_set_0.Xmt_COA
        delta_vxmt_1xcoa = proj_set_1x.VXmt_COA - proj_set_0.VXmt_COA
        delta_rcv_1xcoa = proj_set_1x.Rcv_COA - proj_set_0.Rcv_COA
        delta_vrcv_1xcoa = proj_set_1x.VRcv_COA - proj_set_0.VRcv_COA

        # (3)
        delta_r_avg_1x = (
            proj_set_1x.R_Avg_COA
            - proj_set_0.R_Avg_COA
            - 0.5
            * (
                np.dot(delta_xmt_1xcoa, proj_parameters.uXmt)
                + np.dot(delta_rcv_1xcoa, proj_parameters.uRcv)
            )
        )
        delta_rdot_avg_1x = (
            proj_set_1x.Rdot_Avg_COA
            - proj_set_0.Rdot_Avg_COA
            - 0.5
            * (
                np.dot(delta_xmt_1xcoa, proj_parameters.uXmtDot)
                + np.dot(delta_vxmt_1xcoa, proj_parameters.uXmt)
            )
            - 0.5
            * (
                np.dot(delta_rcv_1xcoa, proj_parameters.uRcvDot)
                + np.dot(delta_vrcv_1xcoa, proj_parameters.uRcv)
            )
        )

        # (4) - Done above
        # (5)
        delta_xmt_1ycoa = proj_set_1y.Xmt_COA - proj_set_0.Xmt_COA
        delta_vxmt_1ycoa = proj_set_1y.VXmt_COA - proj_set_0.VXmt_COA
        delta_rcv_1ycoa = proj_set_1y.Rcv_COA - proj_set_0.Rcv_COA
        delta_vrcv_1ycoa = proj_set_1y.VRcv_COA - proj_set_0.VRcv_COA

        # (6)
        delta_r_avg_1y = (
            proj_set_1y.R_Avg_COA
            - proj_set_0.R_Avg_COA
            - 0.5
            * (
                np.dot(delta_xmt_1ycoa, proj_parameters.uXmt)
                + np.dot(delta_rcv_1ycoa, proj_parameters.uRcv)
            )
        )
        delta_rdot_avg_1y = (
            proj_set_1y.Rdot_Avg_COA
            - proj_set_0.Rdot_Avg_COA
            - 0.5
            * (
                np.dot(delta_xmt_1ycoa, proj_parameters.uXmtDot)
                + np.dot(delta_vxmt_1ycoa, proj_parameters.uXmt)
            )
            - 0.5
            * (
                np.dot(delta_rcv_1ycoa, proj_parameters.uRcvDot)
                + np.dot(delta_vrcv_1ycoa, proj_parameters.uRcv)
            )
        )

        # (7)
        m_rrdot_il = np.array(
            [
                [delta_r_avg_1x / delta_xrow, delta_r_avg_1y / delta_ycol],
                [delta_rdot_avg_1x / delta_xrow, delta_rdot_avg_1y / delta_ycol],
            ]
        )
        m_il_rrdot = np.linalg.inv(m_rrdot_il)

    # Image Location & Slant Plane Sensitivity
    # (1)
    m_spxy_il = sp_mats.M_SPXY_RRdot @ m_rrdot_il
    m_il_spxy = m_il_rrdot @ sp_mats.M_RRdot_SPXY

    # (2)
    m_gpxy_il = sp_mats.M_GPXY_SPXY @ m_spxy_il

    # (3)
    m_il_pt = m_il_spxy @ sp_mats.M_SPXY_PT

    return ImageLocationSensitivityMatrices(
        M_IL_PT=m_il_pt,
        M_SPXY_IL=m_spxy_il,
        M_IL_SPXY=m_il_spxy,
        M_GPXY_IL=m_gpxy_il,
        M_RRdot_IL=m_rrdot_il,
        M_IL_RRdot=m_il_rrdot,
    )


def compute_pos_vel_sensitity_matrices_mono(
    proj_metadata: params.MetadataParams,
    pt0: npt.ArrayLike | None = None,
) -> PosVelSensitivityMatricesMono:
    """Compute the defined monostatic position/velocity sensitivity matrices.

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    pt0 : array_like, optional
        ECF scene point. Defaults to SCP.

    Returns
    -------
    PosVelSensitivityMatricesMono
    """
    pt0, _ = _get_proj_parameters(proj_metadata, pt0)

    il0, _, _ = _calc.scene_to_image(proj_metadata, pt0)
    proj_set_0 = _calc.compute_projection_sets(proj_metadata, il0)
    assert isinstance(proj_set_0, params.ProjectionSetsMono)
    geom_params = compute_proj_geom_params_mono(proj_set_0, pt0)

    # (1)
    m_rrdot_delta_arp = -np.stack([geom_params.uPT, geom_params.uPTDot])

    # (2)
    delta_t_il0_coa = proj_set_0.t_COA - proj_metadata.t_SCP_COA
    m_rrdot_delta_varp = -np.stack(
        [
            geom_params.uPT * delta_t_il0_coa,
            geom_params.uPT + geom_params.uPTDot * delta_t_il0_coa,
        ]
    )

    return PosVelSensitivityMatricesMono(
        M_RRdot_delta_ARP=m_rrdot_delta_arp,
        M_RRdot_delta_VARP=m_rrdot_delta_varp,
    )


def compute_pos_vel_sensitity_matrices_bi(
    proj_metadata: params.MetadataParams,
    pt0: npt.ArrayLike | None = None,
) -> PosVelSensitivityMatricesBi:
    """Compute the defined bistatic position/velocity sensitivity matrices.

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    pt0 : array_like, optional
        ECF scene point. Defaults to SCP.

    Returns
    -------
    PosVelSensitivityMatricesBi
    """
    pt0, _ = _get_proj_parameters(proj_metadata, pt0)

    il0, _, _ = _calc.scene_to_image(proj_metadata, pt0)
    proj_set_0 = _calc.compute_projection_sets(proj_metadata, il0)
    assert isinstance(proj_set_0, params.ProjectionSetsBi)
    geom_params = compute_proj_geom_params_bi(proj_set_0, pt0)

    # (1)
    m_rrdot_delta_xmt = -0.5 * np.stack([geom_params.uXmt, geom_params.uXmtDot])

    # (2)
    delta_tx_il0_coa = proj_set_0.tx_COA - proj_metadata.t_SCP_COA
    m_rrdot_delta_vxmt = -0.5 * np.stack(
        [
            geom_params.uXmt * delta_tx_il0_coa,
            geom_params.uXmt + geom_params.uXmtDot * delta_tx_il0_coa,
        ]
    )

    # (3)
    m_rrdot_delta_rcv = -0.5 * np.stack([geom_params.uRcv, geom_params.uRcvDot])

    # (4)
    delta_tr_il0_coa = proj_set_0.tr_COA - proj_metadata.t_SCP_COA
    m_rrdot_delta_vrcv = -0.5 * np.stack(
        [
            geom_params.uRcv * delta_tr_il0_coa,
            geom_params.uRcv + geom_params.uRcvDot * delta_tr_il0_coa,
        ]
    )

    return PosVelSensitivityMatricesBi(
        M_RRdot_delta_Xmt=m_rrdot_delta_xmt,
        M_RRdot_delta_VXmt=m_rrdot_delta_vxmt,
        M_RRdot_delta_Rcv=m_rrdot_delta_rcv,
        M_RRdot_delta_VRcv=m_rrdot_delta_vrcv,
    )


def compute_sensitivity_matrices(
    proj_metadata: params.MetadataParams,
    pt0: npt.ArrayLike | None = None,
    u_gpn0: npt.ArrayLike | None = None,
    delta_xrow: float | None = None,
    delta_ycol: float | None = None,
) -> SensitivityMatricesLike:
    """Compute the defined sensitivity matrices

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    pt0 : array_like, optional
        ECF scene point. Defaults to SCP.
    u_gpn0 : array_like, optional
        unit normal to scene surface at pt0. Defaults to ETP normal at pt0.
    delta_xrow : float, optional
        row coordinate increment (m).  Defaults to min(Row_SS, 1.0).
    delta_ycol : float, optional
        col coordinate increment (m).  Defaults to min(Col_SS, 1.0).

    Returns
    -------
    SensitivityMatricesLike
    """

    pt0, u_gpn0 = _get_proj_parameters(proj_metadata, pt0, u_gpn0)
    sp_mats = compute_slant_plane_sensitivity_matrices(proj_metadata, pt0, u_gpn0)
    il_mats = compute_image_location_sensitivity_matrices(
        proj_metadata, pt0, u_gpn0, delta_xrow, delta_ycol
    )
    pv_mats: PosVelSensitivityMatricesMono | PosVelSensitivityMatricesBi
    if proj_metadata.is_monostatic():
        pv_mats = compute_pos_vel_sensitity_matrices_mono(proj_metadata, pt0)
        return SensitivityMatricesMono(
            **dataclasses.asdict(sp_mats),
            **dataclasses.asdict(il_mats),
            **dataclasses.asdict(pv_mats),
        )
    pv_mats = compute_pos_vel_sensitity_matrices_bi(proj_metadata, pt0)
    return SensitivityMatricesBi(
        **dataclasses.asdict(sp_mats),
        **dataclasses.asdict(il_mats),
        **dataclasses.asdict(pv_mats),
    )
