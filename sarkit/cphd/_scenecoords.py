"""Calculations related to geographic coordinates in the imaged scene."""

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.wgs84
import sarkit.xmlhelp

from . import _xml as cphd_xml


def planar_ecf_to_iac(
    pt: npt.ArrayLike, iarp: npt.ArrayLike, uiax: npt.ArrayLike, uiay: npt.ArrayLike
) -> np.ndarray:
    """Convert from ECF coordinates to IAC coordinates for Surface Type = PLANAR

    Parameters
    ----------
    pt : (..., 3) array_like
        Array of positions in ECF coordinates with X, Y, Z components in meters in the last dimension.
    iarp : (3,) array_like
        IARP position in ECF coordinates (m).
    uiax, uiay : (3,) array_like
        Image area x-coordinate and y-coordinate unit vectors in ECF coordinates.

    Returns
    -------
    pt_iac : (..., 3) ndarray
        Array of positions in IAC coordinates with IAX, IAY, IAZ components in meters in the last dimension.
    """
    uiaz = np.cross(np.asarray(uiax), np.asarray(uiay))
    pt_iac = np.stack([uiax, uiay, uiaz]) @ (np.asarray(pt) - iarp)[..., np.newaxis]
    return pt_iac[..., 0]


def planar_iac_to_ecf(
    pt_iac: npt.ArrayLike, iarp: npt.ArrayLike, uiax: npt.ArrayLike, uiay: npt.ArrayLike
) -> np.ndarray:
    """Convert from IAC coordinates to ECF coordinates for Surface Type = PLANAR

    Parameters
    ----------
    pt_iac : (..., 2) or (..., 3) array_like
        Array of positions in IAC coordinates with IAX, IAY, IAZ components in meters in the last dimension.
        When last dimension is of length 2, the IAZ component is assumed to be 0.
    iarp : (3,) array_like
        IARP position in ECF coordinates (m).
    uiax, uiay : (3,) array_like
        Image area x-coordinate and y-coordinate unit vectors in ECF coordinates.

    Returns
    -------
    pt : (..., 3) ndarray
        Array of positions in ECF coordinates with X, Y, Z components in meters in the last dimension.
    """
    pt_iac = np.asarray(pt_iac)
    pt = np.asarray(iarp) + pt_iac[..., :1] * uiax + pt_iac[..., 1:2] * uiay
    if pt_iac.shape[-1] == 3:
        pt += pt_iac[..., 2:] * np.cross(np.asarray(uiax), np.asarray(uiay))
    return pt


def hae_llh_to_iac(
    pt_llh: npt.ArrayLike,
    iarp_llh: npt.ArrayLike,
    uiax_ll: npt.ArrayLike,
    uiay_ll: npt.ArrayLike,
) -> np.ndarray:
    """Convert from geodetic LLH coordinates to IAC coordinates for Surface Type = HAE

    Parameters
    ----------
    pt_llh : (..., 3) array_like
        Array of positions in geodetic LLH coordinates with [latitude (deg), longitude (deg), and
        ellipsoidal height (m)] in the last dimension.
    iarp_llh : (3,) array_like
        IARP position in geodetic LLH coordinates.
    uiax_ll, uiay_ll : (2,) array_like
        Vectors containing increments in latitude and longitude in radians per meter for 1.0 meter displacements at the
        IARP in the +IAX and +IAY directions, respectively.

    Returns
    -------
    pt_iac : (..., 3) ndarray
        Array of positions in IAC coordinates with IAX, IAY, IAZ components in meters in the last dimension.
    """
    pt_llh = np.asarray(pt_llh)
    iarp_llh = np.asarray(iarp_llh)
    t_ll2xy = np.linalg.inv(np.stack([uiax_ll, uiay_ll], axis=-1))
    delta_pt_latlon = np.asarray(pt_llh)[..., :2] - iarp_llh[:2]
    pt_iaxy = t_ll2xy @ np.deg2rad(delta_pt_latlon)[..., np.newaxis]
    pt_iaz = pt_llh[..., 2:] - iarp_llh[2]
    return np.concatenate([pt_iaxy[..., 0], pt_iaz], axis=-1)


def hae_iac_to_llh(
    pt_iac: npt.ArrayLike,
    iarp_llh: npt.ArrayLike,
    uiax_ll: npt.ArrayLike,
    uiay_ll: npt.ArrayLike,
) -> np.ndarray:
    """Convert from IAC coordinates to geodetic LLH coordinates for Surface Type = HAE

    Parameters
    ----------
    pt_iac : (..., 2) or (..., 3) array_like
        Array of positions in IAC coordinates with IAX, IAY, IAZ components in meters in the last dimension.
        When last dimension is of length 2, the IAZ component is assumed to be 0.
    iarp_llh : (3,) array_like
        IARP position in geodetic LLH coordinates.
    uiax_ll, uiay_ll : (2,) array_like
        Vectors containing increments in latitude and longitude in radians per meter for 1.0 meter displacements at the
        IARP in the +IAX and +IAY directions, respectively.

    Returns
    -------
    pt_llh : (..., 3) ndarray
        Array of positions in geodetic LLH coordinates with [latitude (deg), longitude (deg), and
        ellipsoidal height (m)] in the last dimension.
    """
    pt_iac = np.asarray(pt_iac)
    pt_iax = pt_iac[..., :1]
    pt_iay = pt_iac[..., 1:2]
    pt_iaz = pt_iac[..., 2:] if pt_iac.shape[-1] == 3 else np.zeros_like(pt_iax)
    delta_pt_latlon = np.rad2deg(
        np.asarray(uiax_ll) * pt_iax + np.asarray(uiay_ll) * pt_iay
    )
    return np.asarray(iarp_llh) + np.concatenate([delta_pt_latlon, pt_iaz], axis=-1)


def ecf_to_iac(cphd_xmltree: lxml.etree.ElementTree, pt: npt.ArrayLike) -> np.ndarray:
    """Convert from ECF coordinates to IAC coordinates

    Parameters
    ----------
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML
    pt : (..., 3) array_like
        Array of positions in ECF coordinates with X, Y, Z components in meters in the last dimension.

    Returns
    -------
    pt_iac : (..., 3) ndarray
        Array of positions in IAC coordinates with IAX, IAY, IAZ components in meters in the last dimension.
    """
    sc_ew = cphd_xml.ElementWrapper(cphd_xmltree.find("{*}SceneCoordinates"))
    return ecf_to_iac_from_ew(sc_ew, pt)


def ecf_to_iac_from_ew(
    sc_ew: sarkit.xmlhelp.ElementWrapper, pt: npt.ArrayLike
) -> np.ndarray:
    if "Planar" in sc_ew["ReferenceSurface"]:
        return planar_ecf_to_iac(
            pt,
            sc_ew["IARP"]["ECF"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAX"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAY"],
        )
    if "HAE" in sc_ew["ReferenceSurface"]:
        return hae_llh_to_iac(
            sarkit.wgs84.cartesian_to_geodetic(pt),
            sc_ew["IARP"]["LLH"],
            sc_ew["ReferenceSurface"]["HAE"]["uIAXLL"],
            sc_ew["ReferenceSurface"]["HAE"]["uIAYLL"],
        )
    raise ValueError("Could not determine ReferenceSurface")


def iac_to_ecf(
    cphd_xmltree: lxml.etree.ElementTree, pt_iac: npt.ArrayLike
) -> np.ndarray:
    """Convert from IAC coordinates to ECF coordinates

    Parameters
    ----------
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML
    pt_iac : (..., 2) or (..., 3) array_like
        Array of positions in IAC coordinates with IAX, IAY, IAZ components in meters in the last dimension.
        When last dimension is of length 2, the IAZ component is assumed to be 0.

    Returns
    -------
    pt : (..., 3) ndarray
        Array of positions in ECF coordinates with X, Y, Z components in meters in the last dimension.
    """
    sc_ew = cphd_xml.ElementWrapper(cphd_xmltree.find("{*}SceneCoordinates"))
    return iac_to_ecf_from_ew(sc_ew, pt_iac)


def iac_to_ecf_from_ew(
    sc_ew: sarkit.xmlhelp.ElementWrapper, pt_iac: npt.ArrayLike
) -> np.ndarray:
    if "Planar" in sc_ew["ReferenceSurface"]:
        return planar_iac_to_ecf(
            pt_iac,
            sc_ew["IARP"]["ECF"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAX"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAY"],
        )
    if "HAE" in sc_ew["ReferenceSurface"]:
        pt_llh = hae_iac_to_llh(
            pt_iac,
            sc_ew["IARP"]["LLH"],
            sc_ew["ReferenceSurface"]["HAE"]["uIAXLL"],
            sc_ew["ReferenceSurface"]["HAE"]["uIAYLL"],
        )
        return sarkit.wgs84.geodetic_to_cartesian(pt_llh)
    raise ValueError("Could not determine ReferenceSurface")


def llh_to_iac(
    cphd_xmltree: lxml.etree.ElementTree, pt_llh: npt.ArrayLike
) -> np.ndarray:
    """Convert from geodetic LLH coordinates to IAC coordinates

    Parameters
    ----------
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML
    pt_llh : (..., 3) array_like
        Array of positions in geodetic LLH coordinates with [latitude (deg), longitude (deg), and
        ellipsoidal height (m)] in the last dimension.

    Returns
    -------
    pt_iac : (..., 3) ndarray
        Array of positions in IAC coordinates with IAX, IAY, IAZ components in meters in the last dimension.
    """
    sc_ew = cphd_xml.ElementWrapper(cphd_xmltree.find("{*}SceneCoordinates"))
    return llh_to_iac_from_ew(sc_ew, pt_llh)


def llh_to_iac_from_ew(
    sc_ew: sarkit.xmlhelp.ElementWrapper, pt_llh: npt.ArrayLike
) -> np.ndarray:
    if "Planar" in sc_ew["ReferenceSurface"]:
        return planar_ecf_to_iac(
            sarkit.wgs84.geodetic_to_cartesian(pt_llh),
            sc_ew["IARP"]["ECF"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAX"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAY"],
        )
    if "HAE" in sc_ew["ReferenceSurface"]:
        return hae_llh_to_iac(
            pt_llh,
            sc_ew["IARP"]["LLH"],
            sc_ew["ReferenceSurface"]["HAE"]["uIAXLL"],
            sc_ew["ReferenceSurface"]["HAE"]["uIAYLL"],
        )
    raise ValueError("Could not determine ReferenceSurface")


def iac_to_llh(
    cphd_xmltree: lxml.etree.ElementTree, pt_iac: npt.ArrayLike
) -> np.ndarray:
    """Convert from IAC coordinates to geodetic LLH coordinates

    Parameters
    ----------
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML
    pt_iac : (..., 2) or (..., 3) array_like
        Array of positions in IAC coordinates with IAX, IAY, IAZ components in meters in the last dimension.
        When last dimension is of length 2, the IAZ component is assumed to be 0.

    Returns
    -------
    pt_llh : (..., 3) ndarray
        Array of positions in geodetic LLH coordinates with [latitude (deg), longitude (deg), and
        ellipsoidal height (m)] in the last dimension.
    """
    sc_ew = cphd_xml.ElementWrapper(cphd_xmltree.find("{*}SceneCoordinates"))
    return iac_to_llh_from_ew(sc_ew, pt_iac)


def iac_to_llh_from_ew(
    sc_ew: sarkit.xmlhelp.ElementWrapper, pt_iac: npt.ArrayLike
) -> np.ndarray:
    if "Planar" in sc_ew["ReferenceSurface"]:
        pt = planar_iac_to_ecf(
            pt_iac,
            sc_ew["IARP"]["ECF"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAX"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAY"],
        )
        return sarkit.wgs84.cartesian_to_geodetic(pt)
    if "HAE" in sc_ew["ReferenceSurface"]:
        return hae_iac_to_llh(
            pt_iac,
            sc_ew["IARP"]["LLH"],
            sc_ew["ReferenceSurface"]["HAE"]["uIAXLL"],
            sc_ew["ReferenceSurface"]["HAE"]["uIAYLL"],
        )
    raise ValueError("Could not determine ReferenceSurface")
