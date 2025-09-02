"""
Select calculations from the CRSD D&I
"""

import copy
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
        "SideOfTrack": side_of_track.tolist(),
        "uAPC": uapc,
        "uAPCDot": uapcdot,
        "DCA": dca,
        "SQNT": sqnt,
        "AZIM": azim,
        "GRAZ": graz,
        "INCD": incd,
    }


def compute_arp_to_rpt_geometry(xmt, vxmt, rcv, vrcv, pt, ueast, unor, uup):
    """Computes ARP geometry as in CRSD D&I 8.4.2"""
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
    arp_geom = compute_apc_to_pt_geometry_parameters(arp, varp, pt, ueast, unor, uup)

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

    return {
        "ARP_COA": arp,
        "VARP_COA": varp,
        "Bistat_Ang": bistat_ang,
        "Bistat_Ang_Rate": bistat_ang_rate,
        "ARP_SideOfTrack": arp_geom["SideOfTrack"],
        "R_ARP_RPT": arp_geom["R_APC_PT"],
        "ARP_Rg_RPT": arp_geom["Rg_PT"],
        "ARP_DCA": arp_geom["DCA"],
        "ARP_SQNT": arp_geom["SQNT"],
        "ARP_AZIM": arp_geom["AZIM"],
        "ARP_GRAZ": arp_geom["GRAZ"],
        "ARP_INCD": arp_geom["INCD"],
        "ARP_TWST": arp_twst,
        "ARP_SLOPE": arp_slope,
        "ARP_LO_ANG": arp_lo_ang,
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


def compute_reference_geometry(
    crsd_xmltree: lxml.etree.ElementTree,
    *,
    pvps: np.ndarray | None = None,
    ppps: np.ndarray | None = None,
    dta: np.ndarray | None = None,
) -> lxml.etree.Element:
    """Return a CRSD/ReferenceGeometry XML element containing parameters computed from other metadata.

    Parameters
    ----------
    crsd_xmltree : lxml.etree.ElementTree
        CRSD XML
    pvps : ndarray or None, optional
        CRSD PVP array for the reference channel
    ppps : ndarray or None, optional
        CRSD PPP array for the reference sequence
    dta : ndarray or None, optional
        Dwell time array to use

    Returns
    -------
    lxml.etree.Element
        New CRSD/ReferenceGeometry XML element
    """
    crsdroot = skcrsd_xml.ElementWrapper(
        copy.deepcopy(crsd_xmltree).getroot(),
    )
    crsdroot.pop("ReferenceGeometry", None)  # remove ReferenceGeometry if present
    refgeom = crsdroot["ReferenceGeometry"]

    # 8.1
    crsd_type = lxml.etree.QName(crsd_xmltree.getroot()).localname
    if crsd_type == "CRSDtx":
        ref_tx_id = crsdroot["TxSequence"]["RefTxId"]
        ref_tx_seq = {x["Identifier"]: x for x in crsdroot["TxSequence"]["Parameters"]}[
            ref_tx_id
        ]
        rpt = ref_tx_seq["TxRefPoint"]["ECF"]
        rpt_iac = ref_tx_seq["TxRefPoint"]["IAC"]
    elif crsd_type in ("CRSDsar", "CRSDrcv"):
        ref_ch_id = crsdroot["Channel"]["RefChId"]
        ref_rcv_chan = {x["Identifier"]: x for x in crsdroot["Channel"]["Parameters"]}[
            ref_ch_id
        ]
        rpt = ref_rcv_chan["RcvRefPoint"]["ECF"]
        rpt_iac = ref_rcv_chan["RcvRefPoint"]["IAC"]
    else:
        raise ValueError(f"Unrecognized {crsd_type=}")

    if crsd_type in ("CRSDtx", "CRSDsar"):
        if crsd_type == "CRSDtx":
            p_ref = ref_tx_seq["RefPulseIndex"]
        if crsd_type == "CRSDsar":
            p_ref = ref_rcv_chan["SARImage"]["RefVectorPulseIndex"]
        if ppps is None:
            raise ValueError("ppps are required for CRSDtx, CRSDsar")
        txc_ref = ppps["TxTime"][p_ref]["Int"] + ppps["TxTime"][p_ref]["Frac"]
        xmt_ref = ppps["TxPos"][p_ref]
        vxmt_ref = ppps["TxVel"][p_ref]

    if crsd_type in ("CRSDsar", "CRSDrcv"):
        v_ref = ref_rcv_chan["RefVectorIndex"]
        if pvps is None:
            raise ValueError("pvps are required for CRSDsar, CRSDrcv")
        trs_ref = pvps["RcvStart"][v_ref]["Int"] + pvps["RcvStart"][v_ref]["Frac"]
        rcv_ref = pvps["RcvPos"][v_ref]
        vrcv_ref = pvps["RcvVel"][v_ref]

    # 8.2
    _, (u_east, u_nor, u_up) = compute_ref_point_parameters(rpt)
    refgeom["RefPoint"]["ECF"] = rpt
    refgeom["RefPoint"]["IAC"] = rpt_iac

    # 8.3.2
    if crsd_type in ("CRSDsar", "CRSDtx"):
        tx_geom_params = compute_apc_to_pt_geometry_parameters(
            xmt_ref, vxmt_ref, rpt, u_east, u_nor, u_up
        )
        refgeom["TxParameters"] = {
            "Time": txc_ref,
            "APCPos": xmt_ref,
            "APCVel": vxmt_ref,
            "SideOfTrack": tx_geom_params["SideOfTrack"],
            "SlantRange": tx_geom_params["R_APC_PT"],
            "GroundRange": tx_geom_params["Rg_PT"],
            "DopplerConeAngle": tx_geom_params["DCA"],
            "SquintAngle": tx_geom_params["SQNT"],
            "AzimuthAngle": tx_geom_params["AZIM"],
            "GrazeAngle": tx_geom_params["GRAZ"],
            "IncidenceAngle": tx_geom_params["INCD"],
        }

    # 8.3.3
    if crsd_type in ("CRSDsar", "CRSDrcv"):
        rcv_geom_params = compute_apc_to_pt_geometry_parameters(
            rcv_ref, vrcv_ref, rpt, u_east, u_nor, u_up
        )
        refgeom["RcvParameters"] = {
            "Time": trs_ref,
            "APCPos": rcv_ref,
            "APCVel": vrcv_ref,
            "SideOfTrack": rcv_geom_params["SideOfTrack"],
            "SlantRange": rcv_geom_params["R_APC_PT"],
            "GroundRange": rcv_geom_params["Rg_PT"],
            "DopplerConeAngle": rcv_geom_params["DCA"],
            "AzimuthAngle": rcv_geom_params["AZIM"],
            "GrazeAngle": rcv_geom_params["GRAZ"],
            "IncidenceAngle": rcv_geom_params["INCD"],
            "SquintAngle": rcv_geom_params["SQNT"],
        }

    # 8.4
    if crsd_type == "CRSDsar":
        # 8.4.1
        # (1)
        t_ref = txc_ref + (
            tx_geom_params["R_APC_PT"]
            / (tx_geom_params["R_APC_PT"] + rcv_geom_params["R_APC_PT"])
        ) * (trs_ref - txc_ref)

        # (2)
        if dta is not None:
            t_cod_rpt, t_dwell_rpt = compute_dwelltimes_using_dta(
                ref_ch_id, rpt_iac[0], rpt_iac[1], crsd_xmltree, dta
            )
        else:
            t_cod_rpt, t_dwell_rpt = compute_dwelltimes_using_poly(
                ref_ch_id, rpt_iac[0], rpt_iac[1], crsd_xmltree
            )

        # 8.4.2
        arp_to_rpt_geom = compute_arp_to_rpt_geometry(
            xmt_ref, vxmt_ref, rcv_ref, vrcv_ref, rpt, u_east, u_nor, u_up
        )

        # 8.4.3
        refgeom["SARImage"] = {
            "CODTime": t_cod_rpt.item(),
            "DwellTime": t_dwell_rpt.item(),
            "ReferenceTime": t_ref,
            "ARPPos": arp_to_rpt_geom["ARP_COA"],
            "ARPVel": arp_to_rpt_geom["VARP_COA"],
            "BistaticAngle": arp_to_rpt_geom["Bistat_Ang"] * (180 / np.pi),
            "BistaticAngleRate": arp_to_rpt_geom["Bistat_Ang_Rate"] * (180 / np.pi),
            "SideOfTrack": arp_to_rpt_geom["ARP_SideOfTrack"],
            "SlantRange": arp_to_rpt_geom["R_ARP_RPT"],
            "GroundRange": arp_to_rpt_geom["ARP_Rg_RPT"],
            "DopplerConeAngle": arp_to_rpt_geom["ARP_DCA"],
            "SquintAngle": arp_to_rpt_geom["ARP_SQNT"],
            "AzimuthAngle": arp_to_rpt_geom["ARP_AZIM"],
            "GrazeAngle": arp_to_rpt_geom["ARP_GRAZ"],
            "IncidenceAngle": arp_to_rpt_geom["ARP_INCD"],
            "TwistAngle": arp_to_rpt_geom["ARP_TWST"],
            "SlopeAngle": arp_to_rpt_geom["ARP_SLOPE"],
            "LayoverAngle": arp_to_rpt_geom["ARP_LO_ANG"],
        }

    return refgeom.elem
