"""Calculations related to geographic coordinates in the imaged scene."""

import lxml.etree
import numpy as np
import numpy.typing as npt

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
    if "Planar" in sc_ew["ReferenceSurface"]:
        return planar_ecf_to_iac(
            pt,
            sc_ew["IARP"]["ECF"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAX"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAY"],
        )
    if "HAE" in sc_ew["ReferenceSurface"]:
        raise NotImplementedError()
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
    if "Planar" in sc_ew["ReferenceSurface"]:
        return planar_iac_to_ecf(
            pt_iac,
            sc_ew["IARP"]["ECF"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAX"],
            sc_ew["ReferenceSurface"]["Planar"]["uIAY"],
        )
    if "HAE" in sc_ew["ReferenceSurface"]:
        raise NotImplementedError()
    raise ValueError("Could not determine ReferenceSurface")
