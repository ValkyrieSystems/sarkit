"""Image Pixel Array information from SIDD Volume 1 section 2"""

import enum

import lxml.etree


class CoordinateSystem(enum.StrEnum):
    """Enumeration of Coordinate Systems supported by SIDD

    .. autoattribute:: CoordinateSystem.PGD
        :no-index:
    .. autoattribute:: CoordinateSystem.PFGD
        :no-index:
    .. autoattribute:: CoordinateSystem.GGD
        :no-index:
    .. autoattribute:: CoordinateSystem.CGD
        :no-index:
    """

    PGD = "Planar Gridded Display"
    PFGD = "Polynomial Fit Gridded Display"
    GGD = "Geodetic Gridded Display"
    CGD = "Cylindrical Gridded Display"


def get_coordinate_system_type(
    sidd_xmltree: lxml.etree.ElementTree,
) -> CoordinateSystem:
    """Determine which coordinate system is used by a SIDD instance

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        a SIDD XML instance

    Returns
    -------
    `CoordinateSystem`
        The coordinate system used by the SIDD instance
    """
    if sidd_xmltree.find("./{*}Measurement/{*}PlaneProjection") is not None:
        return CoordinateSystem.PGD
    if sidd_xmltree.find("./{*}Measurement/{*}PolynomialProjection") is not None:
        return CoordinateSystem.PFGD
    if sidd_xmltree.find("./{*}Measurement/{*}GeographicProjection") is not None:
        return CoordinateSystem.GGD
    if sidd_xmltree.find("./{*}Measurement/{*}CylindricalProjection") is not None:
        return CoordinateSystem.CGD

    raise RuntimeError("Unable to determine coordinate system type")
