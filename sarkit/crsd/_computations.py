"""
Select calculations from the CRSD D&I
"""

import functools

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt

import sarkit.wgs84

from . import _io as skcrsd_io
from . import _xml as skcrsd_xml


def compute_ref_point_parameters(rpt: npt.ArrayLike):
    """Computes the reference point parameters as in CRSD D&I 8.2"""
    rpt_llh = sarkit.wgs84.cartesian_to_geodetic(rpt)
    rpt_lat = rpt_llh[..., 0]
    rpt_lon = rpt_llh[..., 1]
    ueast = np.stack(
        [
            -np.sin(np.deg2rad(rpt_lon)),
            np.cos(np.deg2rad(rpt_lon)),
            np.zeros_like(rpt_lat),
        ],
        axis=-1,
    )
    unor = np.stack(
        [
            -np.sin(np.deg2rad(rpt_lat)) * np.cos(np.deg2rad(rpt_lon)),
            -np.sin(np.deg2rad(rpt_lat)) * np.sin(np.deg2rad(rpt_lon)),
            np.cos(np.deg2rad(rpt_lat)),
        ]
    )
    uup = np.stack(
        [
            np.cos(np.deg2rad(rpt_lat)) * np.cos(np.deg2rad(rpt_lon)),
            np.cos(np.deg2rad(rpt_lat)) * np.sin(np.deg2rad(rpt_lon)),
            np.sin(np.deg2rad(rpt_lat)),
        ],
        axis=-1,
    )
    return rpt_llh, (ueast, unor, uup)


def compute_apc_to_pt_geometry_parameters(
    apc: npt.ArrayLike,
    vapc: npt.ArrayLike,
    pt: npt.ArrayLike,
    ueast: npt.ArrayLike,
    unor: npt.ArrayLike,
    uup: npt.ArrayLike,
):
    """Computes APC geometry parameters as in CRSD D&I 8.3"""
    apc = np.asarray(apc)
    vapc = np.asarray(vapc)
    pt = np.asarray(pt)
    ueast = np.asarray(ueast)
    unor = np.asarray(unor)
    uup = np.asarray(uup)

    # (1)
    r_apc_pt = np.linalg.norm(apc - pt, axis=-1)
    uapc = (apc - pt) / r_apc_pt[..., np.newaxis]
    rdot_apc_pt = np.inner(vapc, uapc)
    uapcdot = (vapc - rdot_apc_pt[..., np.newaxis] * uapc) / r_apc_pt[..., np.newaxis]

    # (2)
    pt_dec = np.linalg.norm(pt, axis=-1)
    uec_pt = pt / pt_dec[..., np.newaxis]

    # (3)
    apc_dec = np.linalg.norm(apc, axis=-1)
    uec_apc = apc / apc_dec[..., np.newaxis]
    ag = pt_dec[..., np.newaxis] * uec_apc

    # (4)
    ea_apc = np.arccos(np.inner(uec_apc, uec_pt))
    rg_pt = pt_dec * ea_apc

    # (5)
    vat = vapc - np.inner(vapc, uec_apc)[..., np.newaxis] * uec_apc
    vat_m = np.linalg.norm(vat, axis=-1)

    # (6)  side of track
    uat = vat / vat_m[..., np.newaxis]
    uleft = np.cross(uec_apc, uat)
    is_left = np.asarray(np.inner(uleft, uapc) < 0)
    is_left[vat_m == 0] = True
    is_left[rg_pt == 0] = True
    side_of_track = np.where(is_left, "L", "R")

    # (7)
    vapc_m = np.linalg.norm(vapc, axis=-1)
    dca = np.asarray(np.rad2deg(np.arccos(-rdot_apc_pt / vapc_m)))
    dca[vapc_m == 0] = 90.0

    # (8)
    pt_at = np.inner(uat, pt - ag)
    pt_ct = np.abs(np.inner(uleft, pt - ag))
    sqnt = np.rad2deg(np.arctan2(pt_at, pt_ct))

    # (9)
    uapc_e = np.inner(ueast, uapc)
    uapc_n = np.inner(unor, uapc)
    uapc_up = np.inner(uup, uapc)

    # (10)
    azim = np.asarray(np.rad2deg(np.arctan2(uapc_e, uapc_n)) % 360)
    azim[rg_pt == 0] = 0.0

    # (11)
    incd = np.rad2deg(np.arccos(uapc_up))
    graz = 90 - incd

    return {
        "R_APC_PT": r_apc_pt,
        "Rdot_APC_PT": rdot_apc_pt,
        "Rg_PT": rg_pt,
        "SideOfTrack": side_of_track,
        "uAPC": uapc,
        "uAPCDot": uapcdot,
        "DCA": dca,
        "SQNT": sqnt,
        "AZIM": azim,
        "GRAZ": graz,
        "INCD": incd,
    }


def compute_apc_to_pt_geometry_parameters_xmlnames(
    apc: npt.ArrayLike,
    vapc: npt.ArrayLike,
    pt: npt.ArrayLike,
    ueast: npt.ArrayLike,
    unor: npt.ArrayLike,
    uup: npt.ArrayLike,
):
    """Computes APC geometry parameters as in CRSD D&I 8.3 but with the XML names"""
    geom = compute_apc_to_pt_geometry_parameters(apc, vapc, pt, ueast, unor, uup)
    return {
        "APCPos": apc,
        "APCVel": vapc,
        "SideOfTrack": geom["SideOfTrack"],
        "SlantRange": geom["R_APC_PT"],
        "GroundRange": geom["Rg_PT"],
        "DopplerConeAngle": geom["DCA"],
        "SquintAngle": geom["SQNT"],
        "AzimuthAngle": geom["AZIM"],
        "GrazeAngle": geom["GRAZ"],
        "IncidenceAngle": geom["INCD"],
    }


def arp_to_rpt_geometry_xmlnames(xmt, vxmt, rcv, vrcv, pt, ueast, unor, uup):
    """Computes ARP geometry as in CRSD D&I 8.4.2 with the XML names"""
    xmt_geom = compute_apc_to_pt_geometry_parameters(xmt, vxmt, pt, ueast, unor, uup)
    rcv_geom = compute_apc_to_pt_geometry_parameters(rcv, vrcv, pt, ueast, unor, uup)
    bp = (xmt_geom["uAPC"] + rcv_geom["uAPC"]) / 2
    bpdot = (xmt_geom["uAPCDot"] + rcv_geom["uAPCDot"]) / 2
    bp_mag = np.linalg.norm(bp, axis=-1)
    bistat_ang = 2 * np.arccos(bp_mag)
    bistat_ang_rate = np.asarray((-4 / np.sin(bistat_ang)) * np.inner(bp, bpdot))
    bistat_ang_rate[bp_mag >= 1] = 0
    bistat_ang_rate[bp_mag == 0] = 0
    uarp = bp / bp_mag[..., np.newaxis]
    uarpdot = (bpdot - np.inner(bpdot, uarp)[..., np.newaxis] * uarp) / bp_mag[
        ..., np.newaxis
    ]
    r_arp_rpt = np.asarray((xmt_geom["R_APC_PT"] + rcv_geom["R_APC_PT"]) / 2)
    rdot_arp_rpt = np.asarray((xmt_geom["Rdot_APC_PT"] + rcv_geom["Rdot_APC_PT"]) / 2)
    arp = pt + r_arp_rpt[..., np.newaxis] * uarp
    varp = rdot_arp_rpt[..., np.newaxis] * uarp + r_arp_rpt * uarpdot
    r_arp_rpt[bp_mag == 0] = 0
    rdot_arp_rpt[bp_mag == 0] = 0
    arp[bp_mag == 0, :] = pt
    varp[bp_mag == 0, :] = 0

    # The next section of calulations (5) - (10) and (13) - (15) are the same as the ones in compute_apc_to_pt_geometry_parameters
    # we call that function instead
    arp_geom = compute_apc_to_pt_geometry_parameters_xmlnames(
        arp, varp, pt, ueast, unor, uup
    )

    # (11)
    ugpz = uup
    bp_gpz = np.inner(bp, ugpz)
    bp_etp = bp - bp_gpz[..., np.newaxis] * ugpz
    bp_gpx = np.linalg.norm(bp_etp, axis=-1)

    # (12)
    ugpx = bp_etp / bp_gpx[..., np.newaxis]
    ugpy = np.cross(ugpz, ugpx)

    # (16)
    bpdot_gpy = np.inner(bpdot, ugpy)

    # (17)
    sgn = np.where(bpdot_gpy > 0, 1, -1)
    spn = sgn[..., np.newaxis] * np.cross(bp, bpdot)
    uspn = spn / np.linalg.norm(spn, axis=-1)[..., np.newaxis]

    # (18)
    arp_twst = np.rad2deg(-np.arcsin(np.inner(uspn, ugpy)))

    # (19)
    arp_slope = np.rad2deg(np.arccos(np.inner(uspn, ugpz)))

    # (20)
    lo_e = -np.inner(ueast, uspn)
    lo_n = -np.inner(unor, uspn)
    arp_lo_ang = np.rad2deg(np.arctan2(lo_e, lo_n)) % 360.0

    return arp_geom | {
        "ARPPos": arp,
        "ARPVel": varp,
        "BistaticAngle": bistat_ang * 180 / np.pi,
        "BistaticAngleRate": bistat_ang_rate * 180 / np.pi,
        "TwistAngle": arp_twst,
        "SlopeAngle": arp_slope,
        "LayoverAngle": arp_lo_ang,
    }


def compute_h_v_los_unit_vectors(apc: npt.ArrayLike, gpt: npt.ArrayLike):
    """Compute H, V, LOS unit vectors as in CRSD D&I 9.4.3"""
    apc = np.asarray(apc)
    gpt = np.asarray(gpt)

    # (1)
    _, (_, _, uup) = compute_ref_point_parameters(gpt)

    # (2)
    r_apc_gpt = np.linalg.norm(gpt - apc, axis=-1)
    ulos = (gpt - apc) / r_apc_gpt[..., np.newaxis]

    # (3)
    hor = np.cross(uup, ulos)
    uhor = hor / np.linalg.norm(hor, axis=-1)[..., np.newaxis]

    # (4)
    uvert = np.cross(ulos, uhor)

    return uhor, uvert, ulos


def compute_h_v_pol_parameters(apc, uacx, uacy, gpt, xr, ampx, ampy, phasex, phasey):
    """Compute H, V polarization parameters as in CRSD D&I 9.4.4"""
    # (1)
    uhor, uvert, ulos = compute_h_v_los_unit_vectors(apc, gpt)

    # (2)
    acxn = uacx - np.inner(uacx, ulos)[..., np.newaxis] * ulos
    acyn = uacy - np.inner(uacy, ulos)[..., np.newaxis] * ulos

    # (3)
    axh = ampx * np.inner(acxn, uhor)
    ayh = ampy * np.inner(acyn, uhor)
    axv = ampx * np.inner(acxn, uvert)
    ayv = ampy * np.inner(acyn, uvert)

    # (4)
    ch = axh * np.exp(xr * 2j * np.pi * phasex) + ayh * np.exp(xr * 2j * np.pi * phasey)
    ah = np.abs(ch)
    phaseh = np.angle(ch) / (2 * np.pi)

    # (5)
    cv = axv * np.exp(xr * 2j * np.pi * phasex) + ayv * np.exp(xr * 2j * np.pi * phasey)
    av = np.abs(cv)
    phasev = np.angle(cv) / (2 * np.pi)

    amph = ah / (ah**2 + av**2) ** 0.5
    ampv = av / (ah**2 + av**2) ** 0.5
    return amph, ampv, phaseh, phasev


def interpolate_support_array(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    x_0: float,
    y_0: float,
    x_ss: float,
    y_ss: float,
    sa: npt.ArrayLike,
    dv_sa: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Method for computing values from a support array as in 9.3.2

    The method in 9.3.2 is specified for a support array containing two parameters.
    This function only handles a single parameter.

    Due to its reuse in 7.5.3, some variables have been renamed/generalized:

    * dcx, dcx -> x, y
    * {Gsa, PhiSA} -> sa
    * {NumRows, NumCols} -> sa.shape
    * {G, Phi} -> values

    Parameters
    ----------
    x, y : array_like
        Row and column values in support array coordinates at which to interpolate.
        Must have the same shape.
    x_0, y_0 : float
        Row and column 0 coordinate
    x_ss, y_ss : float
        Row and column coordinate sample spacing
    sa : array_like
        2-D support array to interpolate
    dv_sa : array_like or None, optional
        Data valid array of the same shape as the support array.
        If ``None``, the entire support array is treated as valid.

    Returns
    -------
    values : ndarray
        Computed array values at (``x``, ``y``)
    dv : ndarray
        Data valid array, where `True` indicates that ``values`` contains a valid value
    """
    sa = np.asarray(sa)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    num_rows, num_cols = sa.shape
    dv = np.ones_like(x, dtype=bool)

    # (1)
    m_x = (x - x_0) / x_ss
    m0 = np.floor(m_x).astype(int)
    m1 = m0 + 1

    dv[(m0 < 0) | (m1 > (num_rows - 1))] = False
    # clip for convenience
    m0 = np.clip(m0, 0, num_rows - 1)
    m1 = np.clip(m1, 0, num_rows - 1)

    # (2)
    n_y = (y - y_0) / y_ss
    n0 = np.floor(n_y).astype(int)
    n1 = n0 + 1

    dv[(n0 < 0) | (n1 > (num_cols - 1))] = False
    # clip for convenience
    n0 = np.clip(n0, 0, num_cols - 1)
    n1 = np.clip(n1, 0, num_cols - 1)

    # (3)
    # a: do bilinear interpolation
    neighbors = np.stack(
        [sa[m0, n0], sa[m0, n1], sa[m1, n0], sa[m1, n1]], axis=-1
    ).reshape(x.shape + (2, 2))

    m_frac = m_x - m0
    n_frac = n_y - n0
    values = np.zeros(shape=neighbors.shape[:-2], dtype=float)
    values = (
        np.stack([1 - m_frac, m_frac], axis=-1).reshape(x.shape + (1, 2))
        @ neighbors
        @ np.stack([1 - n_frac, n_frac], axis=-1).reshape(x.shape + (2, 1))
    )
    # b: DV is false if one or more surrounding elements are invalid
    if dv_sa is not None:
        dv_sa = np.asarray(dv_sa)
        neighbors_dv = np.stack(
            [dv_sa[m0, n0], dv_sa[m0, n1], dv_sa[m1, n0], dv_sa[m1, n1]], axis=-1
        )
        valid_neighbors = neighbors_dv.all(axis=-1)
        dv[~valid_neighbors] = False
    values[~dv] = np.nan
    return values.squeeze(axis=(-2, -1)), dv


def compute_dwelltimes_using_poly(
    ch_id: str,
    iax: npt.ArrayLike,
    iay: npt.ArrayLike,
    crsd_xmltree: lxml.etree.ElementTree,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute center of dwell times and dwell times for scene points using polynomials.

    Parameters
    ----------
    ch_id : str
        Channel unique identifier
    iax, iay : array_like
        Image area coordinates (in meters) of the scene points for which to compute the dwell times
    crsd_xmltree : lxml.etree.ElementTree
        CRSD XML

    Returns
    -------
    t_cod : ndarray
        Center of dwell times (sec) for the scene points relative to the Collection Reference Time
    t_dwell : ndarray
       Dwell times (sec) for which the channel signal array contains the echo signals from the scene points
    """
    iax, iay = np.broadcast_arrays(iax, iay)

    crsdroot = skcrsd_xml.ElementWrapper(crsd_xmltree.getroot())
    chan_params = {x["Identifier"]: x for x in crsdroot["Channel"]["Parameters"]}[ch_id]
    if "Polynomials" not in chan_params["SARImage"]["DwellTimes"]:
        raise ValueError(
            f"Channel {ch_id=} does not use Polynomials. Consider using compute_dwelltimes_using_dta"
        )
    cod_id = chan_params["SARImage"]["DwellTimes"]["Polynomials"]["CODId"]
    dwell_id = chan_params["SARImage"]["DwellTimes"]["Polynomials"]["DwellId"]
    cod_poly = {
        x["Identifier"]: x["CODTimePoly"]
        for x in crsdroot["DwellPolynomials"]["CODTime"]
    }[cod_id]
    dwell_poly = {
        x["Identifier"]: x["DwellTimePoly"]
        for x in crsdroot["DwellPolynomials"]["DwellTime"]
    }[dwell_id]
    t_cod = npp.polyval2d(iax, iay, cod_poly)
    t_dwell = npp.polyval2d(iax, iay, dwell_poly)
    return t_cod, t_dwell


def compute_dwelltimes_using_dta(
    ch_id: str,
    iax: npt.ArrayLike,
    iay: npt.ArrayLike,
    crsd_xmltree: lxml.etree.ElementTree,
    dta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute center of dwell times and dwell times for scene points using a dwell time array.

    Parameters
    ----------
    ch_id : str
        Channel unique identifier
    iax, iay : array_like
        Image area coordinates (in meters) of the scene points for which to compute the dwell times
    crsd_xmltree : lxml.etree.ElementTree
        CRSD XML
    dta : ndarray
        Dwell time array to use

    Returns
    -------
    t_cod : ndarray
        Center of dwell times (sec) for the scene points relative to the Collection Reference Time
    t_dwell : ndarray
       Dwell times (sec) for which the channel signal array contains the echo signals from the scene points
    """
    iax, iay = np.broadcast_arrays(iax, iay)

    crsdroot = skcrsd_xml.ElementWrapper(crsd_xmltree.getroot())
    chan_params = {x["Identifier"]: x for x in crsdroot["Channel"]["Parameters"]}[ch_id]
    if "Array" not in chan_params["SARImage"]["DwellTimes"]:
        raise ValueError(
            f"Channel {ch_id=} does not use a DTA. Consider using compute_dwelltimes_using_poly"
        )
    dta_id = chan_params["SARImage"]["DwellTimes"]["Array"]["DTAId"]
    expected_shape, expected_dtype, sa_elem = skcrsd_io.describe_support_array(
        crsd_xmltree, dta_id
    )
    assert lxml.etree.QName(sa_elem).localname == "DwellTimeArray"
    wrapped_dta = skcrsd_xml.ElementWrapper(sa_elem)
    if dta.shape != expected_shape:
        raise ValueError(f"{dta.shape=} does not match {expected_shape=}")
    if dta.dtype.newbyteorder("=") != expected_dtype:
        raise ValueError(f"{dta.dtype=} is not compatible with {expected_dtype=}")

    interp_dta = functools.partial(
        interpolate_support_array,
        x_0=wrapped_dta["X0"],
        y_0=wrapped_dta["Y0"],
        x_ss=wrapped_dta["XSS"],
        y_ss=wrapped_dta["YSS"],
    )
    cod_ma = skcrsd_io.mask_support_array(dta["COD"], wrapped_dta["NODATA"] or None)
    t_cod = interp_dta(
        x=iax,
        y=iay,
        sa=cod_ma,
        dv_sa=~cod_ma.mask if np.ma.is_masked(cod_ma) else None,
    )[0]

    dt_ma = skcrsd_io.mask_support_array(dta["DT"], wrapped_dta["NODATA"] or None)
    t_dwell = interp_dta(
        x=iax,
        y=iay,
        sa=dt_ma,
        dv_sa=~dt_ma.mask if np.ma.is_masked(dt_ma) else None,
    )[0]
    return t_cod, t_dwell
