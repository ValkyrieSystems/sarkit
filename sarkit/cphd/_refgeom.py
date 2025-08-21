import copy

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp

import sarkit.wgs84

from . import _xml as skcphd_xml


def compute_reference_geometry(
    cphd_xmltree: lxml.etree.ElementTree, pvps: np.ndarray
) -> lxml.etree.Element:
    """Return a CPHD/ReferenceGeometry XML element containing parameters computed from other metadata.

    Parameters
    ----------
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML
    pvps : ndarray
        CPHD PVP array for the reference channel

    Returns
    -------
    lxml.etree.Element
        New CPHD/ReferenceGeometry XML element
    """
    xh = skcphd_xml.XmlHelper(cphd_xmltree)
    cphdroot = skcphd_xml.ElementWrapper(
        copy.deepcopy(cphd_xmltree).getroot(),
    )
    cphdroot.pop("ReferenceGeometry", None)  # remove ReferenceGeometry if present
    refgeom = cphdroot["ReferenceGeometry"]

    # 6.5.1 - Reference Vector Parameters
    ref_id = xh.load("{*}Channel/{*}RefChId")
    ref_chan_parameters = cphd_xmltree.find(
        f"{{*}}Channel/{{*}}Parameters[{{*}}Identifier='{ref_id}']"
    )
    v_ch_ref = int(ref_chan_parameters.findtext("./{*}RefVectorIndex"))

    ref_vector = pvps[v_ch_ref]
    txc = ref_vector["TxTime"]
    xmt = ref_vector["TxPos"]
    vxmt = ref_vector["TxVel"]
    trc_srp = ref_vector["RcvTime"]
    rcv = ref_vector["RcvPos"]
    vrcv = ref_vector["RcvVel"]
    srp = ref_vector["SRPPos"]

    ref_cod_id = ref_chan_parameters.findtext("{*}DwellTimes/{*}CODId")
    ref_dwell_id = ref_chan_parameters.findtext("{*}DwellTimes/{*}DwellId")
    xy2cod = xh.load(
        f"{{*}}Dwell/{{*}}CODTime[{{*}}Identifier='{ref_cod_id}']/{{*}}CODTimePoly"
    )
    xy2dwell = xh.load(
        f"{{*}}Dwell/{{*}}DwellTime[{{*}}Identifier='{ref_dwell_id}']/{{*}}DwellTimePoly"
    )

    # (1)
    srp_llh = sarkit.wgs84.cartesian_to_geodetic(srp)
    srp_lat, srp_lon = np.deg2rad(srp_llh[:2])

    # TODO: factor this out into a separate calculation
    ref_surface = cphd_xmltree.find(
        "./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar"
    )
    if ref_surface is None:  # TODO: Add HAE
        raise NotImplementedError(
            "Non-Planar reference surfaces (e.g. HAE) are currently not supported."
        )

    iax = xh.load_elem(ref_surface.find("./{*}uIAX"))
    iay = xh.load_elem(ref_surface.find("./{*}uIAY"))
    iaz = np.cross(iax, iay)
    iarp = xh.load_elem(cphd_xmltree.find("./{*}SceneCoordinates/{*}IARP/{*}ECF"))
    srp_iac = np.dot([iax, iay, iaz], srp - iarp)

    # (2)
    srp_dec = np.linalg.norm(srp)
    uec_srp = srp / srp_dec

    # (3)
    ueast = np.array((-np.sin(srp_lon), np.cos(srp_lon), 0))
    unor = np.array(
        (
            -np.sin(srp_lat) * np.cos(srp_lon),
            -np.sin(srp_lat) * np.sin(srp_lon),
            np.cos(srp_lat),
        )
    )
    uup = np.array(
        (
            np.cos(srp_lat) * np.cos(srp_lon),
            np.cos(srp_lat) * np.sin(srp_lon),
            np.sin(srp_lat),
        )
    )

    # (4)
    r_xmt_srp = np.linalg.norm(xmt - srp)
    r_rcv_srp = np.linalg.norm(rcv - srp)

    # (5)
    t_ref = txc + r_xmt_srp / (r_xmt_srp + r_rcv_srp) * (trc_srp - txc)

    # (6)
    srp_iax, srp_iay = srp_iac[:2]
    t_cod_srp = npp.polyval2d(srp_iax, srp_iay, c=xy2cod)
    t_dwell_srp = npp.polyval2d(srp_iax, srp_iay, c=xy2dwell)

    # (7)
    refgeom["SRP"]["ECF"] = srp
    refgeom["SRP"]["IAC"] = srp_iac
    refgeom["ReferenceTime"] = t_ref
    refgeom["SRPCODTime"] = t_cod_srp
    refgeom["SRPDwellTime"] = t_dwell_srp

    def calc_arp_geom_relative_to_srp(position, velocity):
        """Calculate ARP geometry parameters relative to the SRP as in section 6.5.2"""
        # (1)
        arp = position
        varp = velocity

        # (2)
        r_arp_srp = np.linalg.norm(arp - srp)
        uarp = (arp - srp) / r_arp_srp
        rdot_arp_srp = np.dot(uarp, varp)

        # (3)
        arp_dec = np.linalg.norm(arp)
        uec_arp = arp / arp_dec

        # (4)
        ea_arp = np.arccos(np.dot(uec_arp, uec_srp))
        rg_arp_srp = srp_dec * ea_arp

        # (5)
        varp_m = np.linalg.norm(varp)
        uvarp = varp / varp_m
        left = np.cross(uec_arp, uvarp)

        # (6)
        look = +1 if np.dot(left, uarp) < 0 else -1
        side_of_track = "L" if look == +1 else "R"

        # (7)
        dca = np.arccos(-rdot_arp_srp / varp_m)

        # (8)
        ugpz = uup
        gpy = np.cross(uup, uarp)
        ugpy = gpy / np.linalg.norm(gpy)
        ugpx = np.cross(ugpy, ugpz)

        # (9)
        graz = np.arccos(np.dot(uarp, ugpx))
        # incidence angle in (15)

        # (10)
        gpx_n = np.dot(ugpx, unor)
        gpx_e = np.dot(ugpx, ueast)
        azim = np.arctan2(gpx_e, gpx_n)

        # (11)
        spn = look * np.cross(uarp, uvarp)
        uspn = spn / np.linalg.norm(spn)

        # (12)
        twst = -np.arcsin(np.dot(uspn, ugpy))

        # (13)
        slope = np.arccos(np.dot(ugpz, uspn))

        # (14)
        lodir_n = np.dot(-uspn, unor)
        lodir_e = np.dot(-uspn, ueast)
        lo_ang = np.arctan2(lodir_e, lodir_n)

        # (15)
        return {
            "ARPPos": arp,
            "ARPVel": varp,
            "SideOfTrack": side_of_track,
            "SlantRange": r_arp_srp,
            "GroundRange": rg_arp_srp,
            "DopplerConeAngle": np.rad2deg(dca),
            "GrazeAngle": np.rad2deg(graz),
            "IncidenceAngle": 90 - np.rad2deg(graz),
            "AzimuthAngle": np.rad2deg(azim) % 360,
            "TwistAngle": np.rad2deg(twst),
            "SlopeAngle": np.rad2deg(slope),
            "LayoverAngle": np.rad2deg(lo_ang) % 360,
        }

    def calc_apc_parameters_bi(time, position, velocity):
        """Compute APC parameters using computations described in 6.5.2 for monostatic."""
        apc_params = calc_arp_geom_relative_to_srp(position, velocity)
        apc_params["Time"] = time
        apc_params["Pos"] = apc_params.pop("ARPPos")
        apc_params["Vel"] = apc_params.pop("ARPVel")
        del apc_params["TwistAngle"]
        del apc_params["SlopeAngle"]
        del apc_params["LayoverAngle"]

        # Conditions unique to bistatic (6.5.3 18-19)
        if np.linalg.norm(velocity) == 0:
            apc_params["DopplerConeAngle"] = 90
            apc_params["SideOfTrack"] = "L"
        if apc_params["GroundRange"] == 0:
            apc_params["GrazeAngle"] = 90
            apc_params["IncidenceAngle"] = 0
            apc_params["AzimuthAngle"] = 0

        return apc_params

    collect_type = cphd_xmltree.findtext("{*}CollectionID/{*}CollectType")
    if collect_type == "MONOSTATIC":
        # 6.5.2
        # (1)
        arp = (xmt + rcv) / 2
        varp = (vxmt + vrcv) / 2

        # (2-15)
        refgeom["Monostatic"] = calc_arp_geom_relative_to_srp(arp, varp)
    elif collect_type == "BISTATIC":
        # 6.5.3
        # (1)
        uxmt = (xmt - srp) / r_xmt_srp
        rdot_xmt_srp = np.dot(uxmt, vxmt)
        uxmtdot = (vxmt - np.dot(rdot_xmt_srp, uxmt)) / r_xmt_srp

        # (2)
        urcv = (rcv - srp) / r_rcv_srp
        rdot_rcv_srp = np.dot(urcv, vrcv)
        urcvdot = (vrcv - np.dot(rdot_rcv_srp, urcv)) / r_rcv_srp

        # (3)
        bp = (uxmt + urcv) / 2
        bpdot = (uxmtdot + urcvdot) / 2

        # (4)
        bp_mag = np.linalg.norm(bp)
        bistat_ang = 2 * np.arccos(bp_mag)

        # (5)
        bistat_ang_rate = (
            0.0 if bp_mag in (0, 1) else -(4 * np.dot(bp, bpdot) / np.sin(bistat_ang))
        )

        # (6)
        ugpz = uup
        bp_gpz = np.dot(bp, ugpz)
        bp_gp = bp - np.dot(bp_gpz, ugpz)
        bp_gpx = np.linalg.norm(bp_gp)

        # (7)
        ubgpx = bp_gp / bp_gpx
        ubgpy = np.cross(ugpz, ubgpx)

        # (8)
        bistat_graz = np.arctan(bp_gpz / bp_gpx)

        # (9)
        bgpx_n = np.dot(ubgpx, unor)
        bgpx_e = np.dot(ubgpx, ueast)
        bistat_azim = np.arctan2(bgpx_e, bgpx_n)

        # (10)
        bpdot_bgpy = np.dot(bpdot, ubgpy)
        bistat_azim_rate = -(bpdot_bgpy / bp_gpx)

        # (11)
        bistat_sgn = +1 if bpdot_bgpy > 0 else -1

        # (12)
        ubp = bp / bp_mag
        bpdotp = np.dot(bpdot, ubp) * ubp
        bpdotn = bpdot - bpdotp

        # (13)
        bipn = bistat_sgn * np.cross(bp, bpdotn)
        ubipn = bipn / np.linalg.norm(bipn)

        # (14)
        bistat_twst = -np.arcsin(np.dot(ubipn, ubgpy))

        # (15)
        bistat_slope = np.arccos(np.dot(ugpz, ubipn))

        # (16)
        b_lodir_n = np.dot(-ubipn, unor)
        b_lodir_e = np.dot(-ubipn, ueast)
        bistat_lo_ang = np.arctan2(b_lodir_e, b_lodir_n)

        # Caveat in (6)
        if bp_gpx == 0:
            bistat_azim = 0
            bistat_azim_rate = 0
            bistat_graz = 0
            bistat_twst = 0
            bistat_slope = 0
            bistat_lo_ang = 0

        # Caveat in (10)
        if bpdot_bgpy == 0:
            bistat_twst = 0
            bistat_slope = 0
            bistat_lo_ang = 0

        # (17)
        refgeom["Bistatic"] = {
            "AzimuthAngle": np.rad2deg(bistat_azim) % 360,
            "AzimuthAngleRate": np.rad2deg(bistat_azim_rate),
            "BistaticAngle": np.rad2deg(bistat_ang),
            "BistaticAngleRate": np.rad2deg(bistat_ang_rate),
            "GrazeAngle": np.rad2deg(bistat_graz),
            "TwistAngle": np.rad2deg(bistat_twst),
            "SlopeAngle": np.rad2deg(bistat_slope),
            "LayoverAngle": np.rad2deg(bistat_lo_ang) % 360,
        }

        # (18)
        refgeom["Bistatic"]["TxPlatform"] = calc_apc_parameters_bi(txc, xmt, vxmt)

        # (19)
        refgeom["Bistatic"]["RcvPlatform"] = calc_apc_parameters_bi(trc_srp, rcv, vrcv)
    else:
        raise ValueError(f"Unrecognized CollectType: {collect_type}")

    return refgeom.elem
