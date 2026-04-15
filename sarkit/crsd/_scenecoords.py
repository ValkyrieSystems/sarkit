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


for func in (ecf_to_iac, iac_to_ecf, llh_to_iac, iac_to_llh):
    newdoc = getattr(getattr(cphd_scenecoords, func.__name__), "__doc__", "")
    func.__doc__ = newdoc.replace("cphd", "crsd").replace("CPHD", "CRSD")
