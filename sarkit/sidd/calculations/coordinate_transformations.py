"""Coordinate Transformations described in SIDD Volume 1 section 3"""

import dataclasses

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt

import sarkit.sidd as sksidd
import sarkit.wgs84

from . import image_pixel_array


def pgd_pixel_to_ecef(
    sidd_xmltree: lxml.etree.ElementTree, pixel: npt.ArrayLike
) -> npt.NDArray:
    """Section 3.2 PGD Pixel to ECEF Coordinate Conversion

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
    """Section 3.3 ECEF Coordinate to PGD Pixel Conversion

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


def ggd_pixel_to_geodetic(
    sidd_xmltree: lxml.etree.ElementTree, pixel: npt.ArrayLike
) -> npt.NDArray:
    """Section 3.4 GGD Pixel to Geodetic Coordinate Conversion

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    pixel : (..., 2) array_like
        N-D array of GGD pixel grid coordinates with {r, c} in the last dimension

    Returns
    -------
    (..., 3) ndarray
        N-D array of geodetic coordinates with {phi, lambda, h} (degrees, degrees, meters) in the last dimension
    """
    if (
        cs := image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    ) != image_pixel_array.CoordinateSystem.GGD:
        raise ValueError(f"Coordinate system must be GGD, not {cs}")

    pixel = np.asarray(pixel)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())

    r_0, c_0 = siddew["Measurement"]["GeographicProjection"]["ReferencePoint"]["Point"]
    delta_r, delta_c = siddew["Measurement"]["GeographicProjection"]["SampleSpacing"]
    r_prime = pixel[..., 0] - r_0
    c_prime = pixel[..., 1] - c_0
    d_r = delta_r * r_prime
    d_c = delta_c * c_prime

    # phi0, lam0, and h0 are used in section 3.4, but not defined.
    # They are implied to be the geodetic location of P_GGD
    phi0, lam0, h0 = sarkit.wgs84.cartesian_to_geodetic(
        siddew["Measurement"]["GeographicProjection"]["ReferencePoint"]["ECEF"]
    )

    phi = phi0 - d_r / 3600.0
    lam = lam0 + d_c / 3600.0
    h = h0
    return np.stack([phi, lam, np.broadcast_to(h, phi.shape)], axis=-1)


def geodetic_to_ggd_pixel(
    sidd_xmltree: lxml.etree.ElementTree, geo: npt.ArrayLike
) -> npt.NDArray:
    """Section 3.5 Geodetic Coordinate to GGD Pixel Conversion

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    geo : (..., 2) array_like
        N-D array of ECEF coordinates with {phi, lam} (latitude degrees, longitude degrees) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of GGD pixel grid coordinates with {r, c} in the last dimension

    """
    if (
        cs := image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    ) != image_pixel_array.CoordinateSystem.GGD:
        raise ValueError(f"Coordinate system must be GGD, not {cs}")

    geo = np.asarray(geo)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())
    phi = geo[..., 0]
    lam = geo[..., 1]
    r_0, c_0 = siddew["Measurement"]["GeographicProjection"]["ReferencePoint"]["Point"]
    delta_r, delta_c = siddew["Measurement"]["GeographicProjection"]["SampleSpacing"]

    # phi0 and lam0 are used in section 3.4, but not defined.
    # They are implied to be the geodetic location of P_GGD
    phi0, lam0, _ = sarkit.wgs84.cartesian_to_geodetic(
        siddew["Measurement"]["GeographicProjection"]["ReferencePoint"]["ECEF"]
    )

    r = r_0 + 3600 * (phi0 - phi) / delta_r
    c = c_0 + 3600 * (lam - lam0) / delta_c

    return np.stack([r, c], axis=-1)


@dataclasses.dataclass(kw_only=True)
class _CgdParams:
    s_cgd: npt.NDArray
    p_cgd: npt.NDArray
    r_cgd: npt.NDArray
    c_cgd: npt.NDArray
    r_s: npt.NDArray
    u_prime: npt.NDArray


def _get_cgd_params(sidd_xmltree):
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())

    s_cgd = siddew["Measurement"]["CylindricalProjection"]["StripmapDirection"]
    p_cgd = siddew["Measurement"]["CylindricalProjection"]["ReferencePoint"]["ECEF"]

    phi, lam, height = sarkit.wgs84.cartesian_to_geodetic(p_cgd)
    x_cgd, y_cgd, z_cgd = p_cgd

    # 2.8.1
    f = sarkit.wgs84.F
    a_prime = np.sqrt(x_cgd**2 + y_cgd**2 + z_cgd**2 / (1 - f) ** 2)
    lam_prime = lam
    phi_prime = np.arctan(z_cgd / ((1 - f) ** 2 * np.sqrt(x_cgd**2 + y_cgd**2)))

    # 2.8.2
    e_prime = sarkit.wgs84.east([phi_prime, lam_prime, height])
    n_prime = sarkit.wgs84.north([phi_prime, lam_prime, height])
    u_prime = sarkit.wgs84.up([phi_prime, lam_prime, height])

    # 2.8.3
    alpha = np.arctan(np.dot(e_prime, s_cgd) / np.dot(n_prime, s_cgd))
    c_cgd = np.cos(alpha) * n_prime + np.sin(alpha) * e_prime
    r_cgd = np.cross(c_cgd, u_prime)

    r_s = siddew["Measurement"]["CylindricalProjection"].get("CurvatureRadius", None)

    if r_s is None:
        # 2.8.4 - 2.8.6
        e1 = sarkit.wgs84.FIRST_ECCENTRICITY
        r_n = a_prime * (1 - e1**2) / (1 - e1**2 * np.sin(phi_prime) ** 2) ** 1.5
        r_e = a_prime / np.sqrt(1 - e1**2 * np.sin(phi_prime) ** 2)
        r_s = 1 / (np.cos(alpha) ** 2 / r_n + np.sin(alpha) ** 2 / r_e)

    return _CgdParams(
        s_cgd=s_cgd, p_cgd=p_cgd, r_cgd=r_cgd, c_cgd=c_cgd, r_s=r_s, u_prime=u_prime
    )


def cgd_pixel_to_ecef(
    sidd_xmltree: lxml.etree.ElementTree, pixel: npt.ArrayLike
) -> npt.NDArray:
    """Section 3.8 CGD Pixel to ECEF Coordinate Conversion

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    pixel : (..., 2) array_like
        N-D array of CGD pixel grid coordinates with {r, c} in the last dimension

    Returns
    -------
    (..., 3) ndarray
        N-D array of ECEF coordinates with {x, y, z} (meters) in the last dimension
    """
    if (
        cs := image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    ) != image_pixel_array.CoordinateSystem.CGD:
        raise ValueError(f"Coordinate system must be CGD, not {cs}")

    pixel = np.asarray(pixel)
    params = _get_cgd_params(sidd_xmltree)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())
    r_0, c_0 = siddew["Measurement"]["CylindricalProjection"]["ReferencePoint"]["Point"]
    delta_r, delta_c = siddew["Measurement"]["CylindricalProjection"]["SampleSpacing"]

    r_prime = pixel[..., 0] - r_0
    c_prime = pixel[..., 1] - c_0
    d_r = (delta_r * r_prime)[..., np.newaxis]
    d_c = (delta_c * c_prime)[..., np.newaxis]
    theta = d_c / params.r_s

    p_ecef = (
        params.p_cgd
        + d_r * params.r_cgd
        + params.r_s * np.sin(theta) * params.c_cgd
        + params.r_s * (np.cos(theta) - 1) * params.u_prime
    )
    return p_ecef


def ecef_to_cgd_pixel(
    sidd_xmltree: lxml.etree.ElementTree, p_ecef: npt.ArrayLike
) -> npt.NDArray:
    """Section 3.9 ECEF Coordinate to CGD Pixel Conversion

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    p_ecef : (..., 3) array_like
        N-D array of ECEF coordinates with {x, y, z} (meters) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of CGD pixel grid coordinates with {r, c} in the last dimension

    """
    if (
        cs := image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    ) != image_pixel_array.CoordinateSystem.CGD:
        raise ValueError(f"Coordinate system must be CGD, not {cs}")

    p_ecef = np.asarray(p_ecef)
    params = _get_cgd_params(sidd_xmltree)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())
    r_0, c_0 = siddew["Measurement"]["CylindricalProjection"]["ReferencePoint"]["Point"]
    delta_r, delta_c = siddew["Measurement"]["CylindricalProjection"]["SampleSpacing"]

    r = r_0 + np.dot(p_ecef - params.p_cgd, params.r_cgd) / delta_r
    c_c = np.dot(p_ecef - params.p_cgd, params.c_cgd)
    c_u = np.dot(p_ecef - params.p_cgd, params.u_prime)

    # This is different than the document, which attempts to divide by c_c
    # c_c is 0 at the reference point and the cross-strip line that goes through the reference point.
    theta = np.arctan2(c_c, c_u + params.r_s)
    c = c_0 + params.r_s * theta / delta_c

    return np.stack([r, c], axis=-1)


def latlon_to_pfgd_pixel(
    sidd_xmltree: lxml.etree.ElementTree, latlon: npt.ArrayLike
) -> npt.NDArray:
    """Section 3.10 PFGD Latitude and Longitude to Row and Column Conversion

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    latlon : (..., 2) array_like
        N-D array of PFGD pixel grid coordinates with {lat, lon} (degrees) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of ECEF coordinates with {r, c} in the last dimension
    """
    if (
        cs := image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    ) != image_pixel_array.CoordinateSystem.PFGD:
        raise ValueError(f"Coordinate system must be PFGD, not {cs}")

    latlon = np.asarray(latlon)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())
    r = npp.polyval2d(
        latlon[..., 0],
        latlon[..., 1],
        siddew["Measurement"]["PolynomialProjection"]["LatLonToRow"],
    )
    c = npp.polyval2d(
        latlon[..., 0],
        latlon[..., 1],
        siddew["Measurement"]["PolynomialProjection"]["LatLonToCol"],
    )

    return np.stack([r, c], axis=-1)


def pfgd_pixel_to_latlon(
    sidd_xmltree: lxml.etree.ElementTree, pixel: npt.ArrayLike
) -> npt.NDArray:
    """Section 3.11 PFGD Row and Column to Latitude, Longitude and Height Conversion

    Parameters
    ----------
    sidd_xmltree : lxml.etree.ElementTree
        SIDD XML metadata
    pixel : (..., 2) array_like
        N-D array of PFGD pixel grid coordinates with {r, c} in the last dimension

    Returns
    -------
    (..., 2 or 3) ndarray
        N-D array of ECEF coordinates with {lat, lon, alt} in the last dimension.
        "alt" only present if SIDD contains the "RowColToAlt" polynomial.
    """
    if (
        cs := image_pixel_array.get_coordinate_system_type(sidd_xmltree)
    ) != image_pixel_array.CoordinateSystem.PFGD:
        raise ValueError(f"Coordinate system must be PFGD, not {cs}")

    pixel = np.asarray(pixel)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())
    lat = npp.polyval2d(
        pixel[..., 0],
        pixel[..., 1],
        siddew["Measurement"]["PolynomialProjection"]["RowColToLat"],
    )
    lon = npp.polyval2d(
        pixel[..., 0],
        pixel[..., 1],
        siddew["Measurement"]["PolynomialProjection"]["RowColToLon"],
    )

    vals = [lat, lon]
    if "RowColToAlt" in siddew["Measurement"]["PolynomialProjection"]:
        alt = npp.polyval2d(
            pixel[..., 0],
            pixel[..., 1],
            siddew["Measurement"]["PolynomialProjection"]["RowColToAlt"],
        )
        vals.append(alt)

    return np.stack(vals, axis=-1)


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
    elif cs == image_pixel_array.CoordinateSystem.GGD:
        llh = ggd_pixel_to_geodetic(sidd_xmltree, pixel)
        return sarkit.wgs84.geodetic_to_cartesian(llh)
    elif cs == image_pixel_array.CoordinateSystem.CGD:
        return cgd_pixel_to_ecef(sidd_xmltree, pixel)
    elif cs == image_pixel_array.CoordinateSystem.PFGD:
        latlon = pfgd_pixel_to_latlon(sidd_xmltree, pixel)
        if latlon.shape[-1] == 2:
            siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())
            _, _, height = sarkit.wgs84.cartesian_to_geodetic(
                siddew["Measurement"]["PolynomialProjection"]["ReferencePoint"]["ECEF"]
            )
            latlon = np.concatenate(
                [latlon, np.full(latlon.shape[:-1] + (1,), height)], axis=-1
            )
        return sarkit.wgs84.geodetic_to_cartesian(latlon)
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
    elif cs == image_pixel_array.CoordinateSystem.GGD:
        llh = sarkit.wgs84.cartesian_to_geodetic(p_ecef)
        return geodetic_to_ggd_pixel(sidd_xmltree, llh)
    elif cs == image_pixel_array.CoordinateSystem.CGD:
        return ecef_to_cgd_pixel(sidd_xmltree, p_ecef)
    elif cs == image_pixel_array.CoordinateSystem.PFGD:
        llh = sarkit.wgs84.cartesian_to_geodetic(p_ecef)
        return latlon_to_pfgd_pixel(sidd_xmltree, llh[..., 0:2])
    raise NotImplementedError(f"Unsupported Coordinate System: {cs}")
