"""
Functions for interacting with SICD XML
"""

import copy
import importlib.resources
import pathlib
from collections.abc import Sequence

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt

import sarkit.sicd.projection as ss_proj
import sarkit.xmlhelp as skxml
import sarkit.xmlhelp._transcoders as skxt
from sarkit import _constants

from . import _constants as sicdconst


# The following transcoders happen to share common implementation across several standards
@skxt.inheritdocstring
class TxtType(skxt.TxtType):
    pass


@skxt.inheritdocstring
class EnuType(skxt.EnuType):
    pass


@skxt.inheritdocstring
class BoolType(skxt.BoolType):
    pass


@skxt.inheritdocstring
class IntType(skxt.IntType):
    pass


@skxt.inheritdocstring
class DblType(skxt.DblType):
    pass


@skxt.inheritdocstring
class XdtType(skxt.XdtType):
    pass


@skxt.inheritdocstring
class RowColType(skxt.RowColType):
    pass


@skxt.inheritdocstring
class CmplxType(skxt.CmplxType):
    pass


@skxt.inheritdocstring
class XyzType(skxt.XyzType):
    pass


@skxt.inheritdocstring
class LatLonHaeType(skxt.LatLonHaeType):
    pass


@skxt.inheritdocstring
class LatLonType(skxt.LatLonType):
    pass


@skxt.inheritdocstring
class PolyType(skxt.PolyType):
    pass


@skxt.inheritdocstring
class Poly2dType(skxt.Poly2dType):
    pass


@skxt.inheritdocstring
class XyzPolyType(skxt.XyzPolyType):
    pass


@skxt.inheritdocstring
class MtxType(skxt.MtxType):
    pass


@skxt.inheritdocstring
class ParameterType(skxt.ParameterType):
    pass


class ImageCornersType(skxt.NdArrayType):
    """
    Transcoder for SICD-like GeoData/ImageCorners XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__("ICP", skxt.LatLonType())

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns the array of ImageCorners encoded in ``elem``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to parse

        Returns
        -------
        coefs : (4, 2) ndarray
            Array of [latitude (deg), longitude (deg)] image corners.

        """
        return np.asarray(
            [
                self.sub_type.parse_elem(x)
                for x in sorted(elem, key=lambda x: x.get("index"))
            ]
        )

    def set_elem(
        self, elem: lxml.etree.Element, val: Sequence[Sequence[float]]
    ) -> None:
        """Set the ICP children of ``elem`` using the ordered vertices from ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : (4, 2) array_like
            Array of [latitude (deg), longitude (deg)] image corners.

        """
        elem[:] = []
        labels = ("1:FRFC", "2:FRLC", "3:LRLC", "4:LRFC")
        elem_ns = lxml.etree.QName(elem).namespace
        ns = f"{{{elem_ns}}}" if elem_ns else ""
        for label, coord in zip(labels, val):
            icp = lxml.etree.SubElement(
                elem, ns + self.sub_tag, attrib={"index": label}
            )
            self.sub_type.set_elem(icp, coord)


class XmlHelper(skxml.XmlHelper):
    """
    :py:class:`~sarkit.xmlhelp.XmlHelper` for SICD

    """

    def __init__(self, element_tree):
        root_ns = lxml.etree.QName(element_tree.getroot()).namespace
        super().__init__(element_tree, XsdHelper(root_ns))


class XsdHelper(skxml.XsdHelper):
    """
    :py:class:`~sarkit.xmlhelp.XsdHelper` for SICD

    """

    def _read_xsdtypes_json(self, root_ns: str) -> str:
        """Return the text contents of the appropriate xsdtypes JSON"""
        schema_name = sicdconst.VERSION_INFO[root_ns]["schema"].name
        return importlib.resources.read_text(
            "sarkit.sicd.xsdtypes",
            pathlib.PurePath(schema_name).with_suffix(".json").name,
        )

    def get_transcoder(self, typename, tag=None):
        """Return the appropriate transcoder given the typename (and optionally tag)."""
        known_builtins = {
            "{http://www.w3.org/2001/XMLSchema}boolean": BoolType(),
            "{http://www.w3.org/2001/XMLSchema}dateTime": XdtType(),
            "{http://www.w3.org/2001/XMLSchema}double": DblType(),
            "{http://www.w3.org/2001/XMLSchema}int": IntType(),
            "{http://www.w3.org/2001/XMLSchema}integer": IntType(),
            "{http://www.w3.org/2001/XMLSchema}nonNegativeInteger": IntType(),
            "{http://www.w3.org/2001/XMLSchema}positiveInteger": IntType(),
            "{http://www.w3.org/2001/XMLSchema}string": TxtType(),
        }
        typedef = self.xsdtypes[typename]
        sicd_110 = {
            "<UNNAMED>-{urn:SICD:1.1.0}DirParamType/{urn:SICD:1.1.0}WgtFunct": skxt.NdArrayType(
                "Wgt", DblType()
            ),
            "<UNNAMED>-{urn:SICD:1.1.0}GeoDataType/{urn:SICD:1.1.0}ImageCorners": ImageCornersType(),
            "<UNNAMED>-{urn:SICD:1.1.0}ImageDataType/{urn:SICD:1.1.0}AmpTable": skxt.NdArrayType(
                "Amplitude", DblType(), index_start=0
            ),
            "<UNNAMED>-{urn:SICD:1.1.0}ImageDataType/{urn:SICD:1.1.0}ValidData": skxt.NdArrayType(
                "Vertex", RowColType()
            ),
            "<UNNAMED>-{urn:SICD:1.1.0}LineType/{urn:SICD:1.1.0}Endpoint": LatLonType(),
            "<UNNAMED>-{urn:SICD:1.1.0}PolygonType/{urn:SICD:1.1.0}Vertex": LatLonType(),
            "<UNNAMED>-{urn:SICD:1.1.0}PositionType/{urn:SICD:1.1.0}RcvAPC": skxt.ListType(
                "RcvAPCPoly", XyzPolyType()
            ),
            (
                "<UNNAMED>-{urn:SICD:1.1.0}RadarCollectionType"
                "/{urn:SICD:1.1.0}Area"
                "/{urn:SICD:1.1.0}Corner"
            ): skxt.NdArrayType("ACP", LatLonHaeType(), include_size_attr=False),
            "{urn:SICD:1.1.0}ComplexType": CmplxType(),
            "{urn:SICD:1.1.0}LatLonCornerStringType": LatLonType(),
            "{urn:SICD:1.1.0}LatLonHAECornerRestrictType": LatLonHaeType(),
            "{urn:SICD:1.1.0}LatLonHAERestrictionType": LatLonHaeType(),
            "{urn:SICD:1.1.0}LatLonRestrictionType": LatLonType(),
            "{urn:SICD:1.1.0}LineType": skxt.NdArrayType("Endpoint", LatLonType()),
            "{urn:SICD:1.1.0}ParameterType": ParameterType(),
            "{urn:SICD:1.1.0}Poly1DType": PolyType(),
            "{urn:SICD:1.1.0}Poly2DType": Poly2dType(),
            "{urn:SICD:1.1.0}PolygonType": skxt.NdArrayType("Vertex", LatLonType()),
            "{urn:SICD:1.1.0}RowColType": RowColType(),
            "{urn:SICD:1.1.0}RowColvertexType": RowColType(),
            "{urn:SICD:1.1.0}XYZPolyAttributeType": XyzPolyType(),
            "{urn:SICD:1.1.0}XYZPolyType": XyzPolyType(),
            "{urn:SICD:1.1.0}XYZType": XyzType(),
        }
        sicd_121 = {
            k.replace("urn:SICD:1.1.0", "urn:SICD:1.2.1"): v
            for k, v in sicd_110.items()
        }
        sicd_130 = {
            k.replace("urn:SICD:1.1.0", "urn:SICD:1.3.0"): v
            for k, v in sicd_110.items()
        }
        sicd_140 = {
            k.replace("urn:SICD:1.1.0", "urn:SICD:1.4.0"): v
            for k, v in sicd_110.items()
        }
        sicd_140 |= {
            "<UNNAMED>-{urn:SICD:1.4.0}ErrorStatisticsType/{urn:SICD:1.4.0}AdjustableParameterOffsets/{urn:SICD:1.4.0}APOError": MtxType(
                (8, 8)
            ),
            "<UNNAMED>-{urn:SICD:1.4.0}ErrorStatisticsType/{urn:SICD:1.4.0}BistaticAdjustableParameterOffsets/{urn:SICD:1.4.0}APOError": MtxType(
                (16, 16)
            ),
            "<UNNAMED>-{urn:SICD:1.4.0}BistaticComponentsErrorType/{urn:SICD:1.4.0}RadarSensor/{urn:SICD:1.4.0}TxRcvTimeFreq": MtxType(
                (4, 4)
            ),
            "{urn:SICD:1.4.0}Matrix6x6Type": MtxType((6, 6)),
        }
        easy = sicd_110 | sicd_121 | sicd_130 | sicd_140
        if tag is not None and lxml.etree.QName(tag).localname == "CalibrationDate":
            return skxt.XdtType(force_utc=False)
        if typename.startswith("{http://www.w3.org/2001/XMLSchema}"):
            return known_builtins[typename]
        if typename in easy:
            return easy[typename]
        if not typedef.children:
            return known_builtins.get(typedef.text_typename, TxtType())
        return None


class ElementWrapper(skxml.ElementWrapper):
    """:py:class:`~sarkit.xmlhelp.ElementWrapper` for SICD that can set ``xsdhelper`` automatically.

    Refer to :py:class:`sarkit.xmlhelp.ElementWrapper` for full documentation.
    """

    def __init__(
        self,
        elem,
        xsdhelper=None,
        wrapped_parent=None,
        typename=None,
        elementpath=None,
        roottag=None,
    ):
        if xsdhelper is None:
            root_ns = lxml.etree.QName(roottag or elem).namespace
            xsdhelper = XsdHelper(root_ns)
        super().__init__(
            elem, xsdhelper, wrapped_parent, typename, elementpath, roottag
        )


def compute_scp_coa(sicd_xmltree: lxml.etree.ElementTree) -> lxml.etree.ElementTree:
    """Return a SICD/SCPCOA XML containing parameters computed from other metadata.

    The namespace of the new SICD/SCPCOA element is retained from ``sicd_xmltree``.

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree

    Returns
    -------
    lxml.etree.Element
        New SICD/SCPCOA XML element
    """
    version_ns = lxml.etree.QName(sicd_xmltree.getroot()).namespace
    sicdroot = ElementWrapper(
        copy.deepcopy(sicd_xmltree).getroot(),
    )
    sicd_versions = list(sicdconst.VERSION_INFO)
    pre_1_4 = sicd_versions.index(version_ns) < sicd_versions.index("urn:SICD:1.4.0")

    # COA Parameters for All Images
    sicdroot.pop("SCPCOA", None)  # remove SCPCOA if present
    scpcoa = sicdroot["SCPCOA"]
    t_coa = sicdroot["Grid"]["TimeCOAPoly"][0, 0]
    scpcoa["SCPTime"] = t_coa
    scp = sicdroot["GeoData"]["SCP"]["ECF"]

    arp_poly = sicdroot["Position"]["ARPPoly"]
    arp_coa = npp.polyval(t_coa, arp_poly).squeeze()
    scpcoa["ARPPos"] = arp_coa
    varp_coa = npp.polyval(t_coa, npp.polyder(arp_poly, m=1)).squeeze()
    scpcoa["ARPVel"] = varp_coa
    aarp_coa = npp.polyval(t_coa, npp.polyder(arp_poly, m=2)).squeeze()
    scpcoa["ARPAcc"] = aarp_coa

    r_coa = np.linalg.norm(scp - arp_coa)
    scpcoa["SlantRange"] = r_coa
    arp_dec_coa = np.linalg.norm(arp_coa)
    u_arp_coa = arp_coa / arp_dec_coa
    scp_dec = np.linalg.norm(scp)
    u_scp = scp / scp_dec
    ea_coa = np.arccos(np.dot(u_arp_coa, u_scp))
    rg_coa = scp_dec * ea_coa
    scpcoa["GroundRange"] = rg_coa

    vm_coa = np.linalg.norm(varp_coa)
    u_varp_coa = varp_coa / vm_coa
    u_los_coa = (scp - arp_coa) / r_coa
    left_coa = np.cross(u_arp_coa, u_varp_coa)
    dca_coa = np.arccos(np.dot(u_varp_coa, u_los_coa))
    scpcoa["DopplerConeAng"] = np.rad2deg(dca_coa)
    side_of_track = "L" if np.dot(left_coa, u_los_coa) > 0 else "R"
    scpcoa["SideOfTrack"] = side_of_track
    look = 1 if np.dot(left_coa, u_los_coa) > 0 else -1

    scp_lat, scp_lon = sicdroot["GeoData"]["SCP"]["LLH"][:2]
    u_gpz = np.array(
        [
            np.cos(np.deg2rad(scp_lon)) * np.cos(np.deg2rad(scp_lat)),
            np.sin(np.deg2rad(scp_lon)) * np.cos(np.deg2rad(scp_lat)),
            np.sin(np.deg2rad(scp_lat)),
        ]
    )
    arp_gpz_coa = np.dot(arp_coa - scp, u_gpz)
    aetp_coa = arp_coa - u_gpz * arp_gpz_coa
    arp_gpx_coa = np.linalg.norm(aetp_coa - scp)
    u_gpx = (aetp_coa - scp) / arp_gpx_coa
    u_gpy = np.cross(u_gpz, u_gpx)

    cos_graz = arp_gpx_coa / r_coa
    sin_graz = arp_gpz_coa / r_coa
    graz = np.arccos(cos_graz) if pre_1_4 else np.arcsin(sin_graz)
    scpcoa["GrazeAng"] = np.rad2deg(graz)
    incd = 90.0 - np.rad2deg(graz)
    scpcoa["IncidenceAng"] = incd

    spz = look * np.cross(u_varp_coa, u_los_coa)
    u_spz = spz / np.linalg.norm(spz)
    # u_spx intentionally omitted
    # u_spy intentionally omitted

    # arp/varp in slant plane coordinates intentionally omitted

    slope = np.arccos(np.dot(u_gpz, u_spz))
    scpcoa["SlopeAng"] = np.rad2deg(slope)

    u_east = np.array([-np.sin(np.deg2rad(scp_lon)), np.cos(np.deg2rad(scp_lon)), 0.0])
    u_north = np.cross(u_gpz, u_east)
    az_north = np.dot(u_north, u_gpx)
    az_east = np.dot(u_east, u_gpx)
    azim = np.arctan2(az_east, az_north)
    scpcoa["AzimAng"] = np.rad2deg(azim) % 360

    cos_slope = np.cos(slope)  # this symbol seems to be undefined in SICD Vol 1
    lodir_coa = u_gpz - u_spz / cos_slope
    lo_north = np.dot(u_north, lodir_coa)
    lo_east = np.dot(u_east, lodir_coa)
    layover = np.arctan2(lo_east, lo_north)
    scpcoa["LayoverAng"] = np.rad2deg(layover) % 360

    # uZI intentionally omitted

    twst = -np.arcsin(np.dot(u_gpy, u_spz))
    scpcoa["TwistAng"] = np.rad2deg(twst)

    # Additional COA Parameters for Bistatic Images
    params = ss_proj.MetadataParams.from_xml(sicd_xmltree)
    if not pre_1_4 and not params.is_monostatic():
        assert params.Xmt_Poly is not None
        assert params.Rcv_Poly is not None
        tx_coa = t_coa - (1 / _constants.speed_of_light) * np.linalg.norm(
            npp.polyval(t_coa, params.Xmt_Poly) - scp
        )
        tr_coa = t_coa + (1 / _constants.speed_of_light) * np.linalg.norm(
            npp.polyval(t_coa, params.Rcv_Poly) - scp
        )

        xmt_coa = npp.polyval(tx_coa, params.Xmt_Poly)
        vxmt_coa = npp.polyval(tx_coa, npp.polyder(params.Xmt_Poly, m=1))
        axmt_coa = npp.polyval(tx_coa, npp.polyder(params.Xmt_Poly, m=2))
        r_xmt_scp = np.linalg.norm(xmt_coa - scp)
        u_xmt_coa = (xmt_coa - scp) / r_xmt_scp

        rdot_xmt_scp = np.dot(u_xmt_coa, vxmt_coa)
        u_xmt_dot_coa = (vxmt_coa - rdot_xmt_scp * u_xmt_coa) / r_xmt_scp

        rcv_coa = npp.polyval(tr_coa, params.Rcv_Poly)
        vrcv_coa = npp.polyval(tr_coa, npp.polyder(params.Rcv_Poly, m=1))
        arcv_coa = npp.polyval(tr_coa, npp.polyder(params.Rcv_Poly, m=2))
        r_rcv_scp = np.linalg.norm(rcv_coa - scp)
        u_rcv_coa = (rcv_coa - scp) / r_rcv_scp

        rdot_rcv_scp = np.dot(u_rcv_coa, vrcv_coa)
        u_rcv_dot_coa = (vrcv_coa - rdot_rcv_scp * u_rcv_coa) / r_rcv_scp

        bp_coa = 0.5 * (u_xmt_coa + u_rcv_coa)
        bpdot_coa = 0.5 * (u_xmt_dot_coa + u_rcv_dot_coa)

        bp_mag_coa = np.linalg.norm(bp_coa)
        bistat_ang_coa = 2.0 * np.arccos(bp_mag_coa)

        if bp_mag_coa in (0.0, 1.0):
            bistat_ang_rate_coa = 0.0
        else:
            bistat_ang_rate_coa = (
                (-180 / np.pi)
                * (4 / np.sin(bistat_ang_coa))
                * np.dot(bp_coa, bpdot_coa)
            )

        def _steps_10_to_15(xmt_coa, vxmt_coa, u_xmt_coa, r_xmt_scp):
            xmt_dec = np.linalg.norm(xmt_coa)
            u_ec_xmt_coa = xmt_coa / xmt_dec
            ea_xmt_coa = np.arccos(np.dot(u_ec_xmt_coa, u_scp))
            rg_xmt_scp = scp_dec * ea_xmt_coa

            left_xmt = np.cross(u_ec_xmt_coa, vxmt_coa)
            side_of_track_xmt = "L" if np.dot(left_xmt, u_xmt_coa) < 0 else "R"

            vxmt_m = np.linalg.norm(vxmt_coa)
            dca_xmt = np.arccos(-rdot_xmt_scp / vxmt_m)

            xmt_gpz_coa = np.dot((xmt_coa - scp), u_gpz)
            xmt_etp_coa = xmt_coa - xmt_gpz_coa * u_gpz
            u_gpx_x = (xmt_etp_coa - scp) / np.linalg.norm(xmt_etp_coa - scp)

            graz_xmt = np.arcsin(xmt_gpz_coa / r_xmt_scp)
            incd_xmt = 90 - np.rad2deg(graz_xmt)

            az_xmt_n = np.dot(u_north, u_gpx_x)
            az_xmt_e = np.dot(u_east, u_gpx_x)
            azim_xmt = np.arctan2(az_xmt_e, az_xmt_n)

            return {
                "SideOfTrack": side_of_track_xmt,
                "SlantRange": r_xmt_scp,
                "GroundRange": rg_xmt_scp,
                "DopplerConeAng": np.rad2deg(dca_xmt),
                "GrazeAng": np.rad2deg(graz_xmt),
                "IncidenceAng": incd_xmt,
                "AzimAng": np.rad2deg(azim_xmt) % 360,
            }

        scpcoa["Bistatic"]["BistaticAng"] = np.rad2deg(bistat_ang_coa)
        scpcoa["Bistatic"]["BistaticAngRate"] = bistat_ang_rate_coa
        scpcoa["Bistatic"]["TxPlatform"] = {
            "Time": tx_coa,
            "Pos": xmt_coa,
            "Vel": vxmt_coa,
            "Acc": axmt_coa,
            **_steps_10_to_15(xmt_coa, vxmt_coa, u_xmt_coa, r_xmt_scp),
        }
        scpcoa["Bistatic"]["RcvPlatform"] = {
            "Time": tr_coa,
            "Pos": rcv_coa,
            "Vel": vrcv_coa,
            "Acc": arcv_coa,
            **_steps_10_to_15(rcv_coa, vrcv_coa, u_rcv_coa, r_rcv_scp),
        }
    return scpcoa.elem
