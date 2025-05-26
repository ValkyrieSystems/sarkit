"""Coordinate Transformations described in SIDD Volume 1 section 3"""

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.sidd as sksidd

from . import image_pixel_array


def pgd_pixel_to_ecef(
    sidd_xmltree: lxml.etree.ElementTree, pixel: npt.ArrayLike
) -> npt.NDArray:
    """3.2 PGD Pixel to ECEF Coordinate Conversion

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    pixel : (..., 2) array_like
        N-D array of PGD pixel grid coordinates with {r, c} in the last dimension

    Returns
    -------
    (..., 3) ndarray
        N-D array of ECEF coordinates with {x, y, z} (meters) in the last dimension
    """

    if (
        cs := image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    ) != image_pixel_array.CoordinateSystem.PGD:
        raise ValueError(f"Coordinate system must be PGD, not {cs}")

    pixel = np.asarray(pixel)

    xmlhelp = sksidd.XmlHelper(sidd_xmltree)
    p_pgd = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}ReferencePoint/{*}ECEF"
    )
    r_0, c_0 = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}ReferencePoint/{*}Point"
    )
    delta_r, delta_c = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}SampleSpacing"
    )
    r_pgd = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}ProductPlane/{*}RowUnitVector"
    )
    c_pgd = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}ProductPlane/{*}ColUnitVector"
    )

    r_prime = pixel[..., 0] - r_0
    c_prime = pixel[..., 1] - c_0
    d_r = delta_r * r_prime
    d_c = delta_c * c_prime
    p_ecef = p_pgd + d_r[..., np.newaxis] * r_pgd + d_c[..., np.newaxis] * c_pgd
    return p_ecef


def ecef_to_pgd_pixel(
    sidd_xmltree: lxml.etree.ElementTree, p_ecef: npt.ArrayLike
) -> npt.NDArray:
    """3.3 ECEF Coordinate to PGD Pixel Conversion

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    p_ecef : (..., 3) array_like
        N-D array of ECEF coordinates with {x, y, z} (meters) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of PGD pixel grid coordinates with {r, c} in the last dimension

    """

    if (
        cs := image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    ) != image_pixel_array.CoordinateSystem.PGD:
        raise ValueError(f"Coordinate system must be PGD, not {cs}")

    p_ecef = np.asarray(p_ecef)

    xmlhelp = sksidd.XmlHelper(sidd_xmltree)
    p_pgd = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}ReferencePoint/{*}ECEF"
    )
    r_0, c_0 = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}ReferencePoint/{*}Point"
    )
    delta_r, delta_c = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}SampleSpacing"
    )
    r_pgd = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}ProductPlane/{*}RowUnitVector"
    )
    c_pgd = xmlhelp.load(
        "./{*}Measurement/{*}PlaneProjection/{*}ProductPlane/{*}ColUnitVector"
    )

    r = r_0 + np.inner(p_ecef - p_pgd, r_pgd) / delta_r
    c = c_0 + np.inner(p_ecef - p_pgd, c_pgd) / delta_c
    return np.stack((r, c), axis=-1)


def pixel_to_ecef(
    sidd_xmltree: lxml.etree.ElementTree, pixel: npt.ArrayLike
) -> npt.NDArray:
    """Convert pixel grid coordinates to ECEF coordinates

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    pixel : (..., 2) array_like
        N-D array of pixel grid coordinates with {r, c} in the last dimension

    Returns
    -------
    (..., 3) ndarray
        N-D array of ECEF coordinates with {x, y, z} (meters) in the last dimension

    """
    cs = image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    if cs == image_pixel_array.CoordinateSystem.PGD:
        return pgd_pixel_to_ecef(sidd_xmltree, pixel)
    raise NotImplementedError(f"Unsupported Coordinate System: {cs}")


def ecef_to_pixel(
    sidd_xmltree: lxml.etree.ElementTree, p_ecef: npt.ArrayLike
) -> npt.NDArray:
    """Convert ECEF coordinates to pixel grid coordinates

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    p_ecef : (..., 3) array_like
        N-D array of ECEF coordinates with {x, y, z} (meters) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of pixel grid coordinates with {r, c} in the last dimension

    """
    cs = image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    if cs == image_pixel_array.CoordinateSystem.PGD:
        return ecef_to_pgd_pixel(sidd_xmltree, p_ecef)
    raise NotImplementedError(f"Unsupported Coordinate System: {cs}")
