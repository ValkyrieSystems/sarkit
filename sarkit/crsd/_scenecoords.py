"""Calculations related to geographic coordinates in the imaged scene.

CRSD D&I DD explicitly states equivalence to CPHD.
"""

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.cphd._scenecoords as cphd_scenecoords
import sarkit.crsd._xml as crsd_xml


def ecf_to_iac(crsd_xmltree: lxml.etree.ElementTree, pt: npt.ArrayLike) -> np.ndarray:
    sc_ew = crsd_xml.ElementWrapper(crsd_xmltree.find("{*}SceneCoordinates"))
    return cphd_scenecoords.ecf_to_iac_from_ew(sc_ew, pt)


def iac_to_ecf(
    crsd_xmltree: lxml.etree.ElementTree, pt_iac: npt.ArrayLike
) -> np.ndarray:
    sc_ew = crsd_xml.ElementWrapper(crsd_xmltree.find("{*}SceneCoordinates"))
    return cphd_scenecoords.iac_to_ecf_from_ew(sc_ew, pt_iac)


def llh_to_iac(
    crsd_xmltree: lxml.etree.ElementTree, pt_llh: npt.ArrayLike
) -> np.ndarray:
    sc_ew = crsd_xml.ElementWrapper(crsd_xmltree.find("{*}SceneCoordinates"))
    return cphd_scenecoords.llh_to_iac_from_ew(sc_ew, pt_llh)


def iac_to_llh(
    crsd_xmltree: lxml.etree.ElementTree, pt_iac: npt.ArrayLike
) -> np.ndarray:
    sc_ew = crsd_xml.ElementWrapper(crsd_xmltree.find("{*}SceneCoordinates"))
    return cphd_scenecoords.iac_to_llh_from_ew(sc_ew, pt_iac)


def get_channel_image_area(
    crsd_xmltree: lxml.etree.ElementTree, ch_id: str, *, use_polygon: bool = True
) -> np.ndarray:
    # docstring copied from CPHD version
    if lxml.etree.QName(crsd_xmltree.getroot()).localname != "CRSDsar":
        raise ValueError("Only CRSDsar products have channel image areas")
    ew = crsd_xml.ElementWrapper(crsd_xmltree.getroot())
    chan_param_ew = ew["Channel"].find("Parameters", Identifier=ch_id)
    imgarea_ew = (
        chan_param_ew["SARImage"]["ImageArea"]
        if "ImageArea" in chan_param_ew["SARImage"]
        else ew["SceneCoordinates"]["ImageArea"]
    )
    return cphd_scenecoords.get_image_area_vertices_from_ew(
        imgarea_ew, use_polygon=use_polygon
    )


def get_scene_image_area(
    crsd_xmltree: lxml.etree.ElementTree, *, use_polygon: bool = True
) -> np.ndarray:
    # docstring copied from CPHD version
    ew = crsd_xml.ElementWrapper(crsd_xmltree.getroot())
    return cphd_scenecoords.get_image_area_vertices_from_ew(
        ew["SceneCoordinates"]["ImageArea"], use_polygon=use_polygon
    )


def get_extended_image_area(
    crsd_xmltree: lxml.etree.ElementTree, *, use_polygon: bool = True
) -> np.ndarray | None:
    # docstring copied from CPHD version
    if lxml.etree.QName(crsd_xmltree.getroot()).localname != "CRSDsar":
        raise ValueError("Only CRSDsar products can have extended image areas")
    ew = crsd_xml.ElementWrapper(crsd_xmltree.getroot())
    imgarea_ew = ew["SceneCoordinates"].get("ExtendedArea", None)
    return (
        None
        if imgarea_ew is None
        else cphd_scenecoords.get_image_area_vertices_from_ew(
            imgarea_ew, use_polygon=use_polygon
        )
    )


for func in (
    ecf_to_iac,
    iac_to_ecf,
    llh_to_iac,
    iac_to_llh,
    get_channel_image_area,
    get_scene_image_area,
    get_extended_image_area,
):
    newdoc = getattr(getattr(cphd_scenecoords, func.__name__), "__doc__", "")
    func.__doc__ = newdoc.replace("cphd", "crsd").replace("CPHD", "CRSD")
