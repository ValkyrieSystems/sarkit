"""
================================================
World Geodetic System 1984 (:mod:`sarkit.wgs84`)
================================================

Constants
=========

.. list-table:: WGS-84 Defining Parameters
   :header-rows: 1

   * - Attribute
     - Quantity
     - Units
   * - ``SEMI_MAJOR_AXIS``
     - semi-major axis (equatorial radius of the Earth)
     - m
   * - ``A``
     - semi-major axis (equatorial radius of the Earth)
     - m
   * - ``FLATTENING_FACTOR``
     - flattening factor of the Earth (1/f)
     - (unitless)
   * - ``NOMINAL_MEAN_ANGULAR_VELOCITY``
     - nominal mean angular velocity of the earth
     - rad / s

.. list-table:: WGS-84 Ellipsoid Derived Geometric Constants
   :header-rows: 1

   * - Attribute
     - Quantity
     - Units
   * - ``FLATTENING_REDUCED``
     - flattening (reduced)
     - (unitless)
   * - ``F``
     - flattening (reduced)
     - (unitless)
   * - ``SEMI_MINOR_AXIS``
     - semi-minor axis (polar radius of the Earth)
     - m
   * - ``B``
     - semi-minor axis (polar radius of the Earth)
     - m
   * - ``FIRST_ECCENTRICITY``
     - first eccentricity
     - (unitless)
   * - ``FIRST_ECCENTRICITY_SQUARED``
     - first eccentricity squared
     - (unitless)
   * - ``SECOND_ECCENTRICITY``
     - second eccentricity
     - (unitless)
   * - ``SECOND_ECCENTRICITY_SQUARED``
     - second eccentricity squared
     - (unitless)

Global Coordinate Systems
=========================

.. autosummary::
   :toctree: generated/

   cartesian_to_geodetic
   geodetic_to_cartesian

Local Coordinate Systems
=========================

.. autosummary::
   :toctree: generated/

   east
   north
   up

"""

import math

import numpy as np
import numpy.typing as npt

# Select parameters from NGA.STND.0036_1.0.0_WGS84 (2014-07-08)
# Defining Parameters
SEMI_MAJOR_AXIS = A = 6378137.0  # meters
FLATTENING_FACTOR = 298.257223563
NOMINAL_MEAN_ANGULAR_VELOCITY = 7.292115e-5  # rad/s

# Derived Parameters
FLATTENING_REDUCED = F = 1 / FLATTENING_FACTOR
SEMI_MINOR_AXIS = B = A * (1 - F)  # meters
FIRST_ECCENTRICITY_SQUARED = 2 * F - F**2
FIRST_ECCENTRICITY = math.sqrt(FIRST_ECCENTRICITY_SQUARED)
SECOND_ECCENTRICITY_SQUARED = (A**2 - B**2) / B**2
SECOND_ECCENTRICITY = math.sqrt(SECOND_ECCENTRICITY_SQUARED)

# useful intermediate terms
_A2 = A * A
_B2 = B * B
_E4 = FIRST_ECCENTRICITY_SQUARED * FIRST_ECCENTRICITY_SQUARED
_OME2 = 1.0 - FIRST_ECCENTRICITY_SQUARED


def cartesian_to_geodetic(xyz: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts WGS 84 cartesian coordinates to geodetic coordinates.

    Parameters
    ----------
    xyz : (..., 3) array_like
        Array of cartesian coordinates with X, Y, Z components in meters in the last dimension.

    Returns
    -------
    (..., 3) ndarray
        Array of geodetic coordinates with [latitude (deg), longitude (deg), and ellipsoidal height (m)] in the last
        dimension.

    Notes
    -----
    The latitude and height are computed using Heikkinen's algorithm [1]_.

    References
    ----------

    .. [1] M. Heikkinen, "Geschlossene Formeln zur Berechnung raumlicher geodatischer Koordinaten aus rechtwinkligen
       Koordinaten", Zeitschrift Vermess (in German), 107, pp. 207-211, 1982.

    Examples
    --------
    .. doctest::

        >>> import sarkit.wgs84
        >>> sarkit.wgs84.cartesian_to_geodetic(
        ...     [
        ...         [sarkit.wgs84.SEMI_MAJOR_AXIS, 0, 0],
        ...         [0, sarkit.wgs84.SEMI_MAJOR_AXIS, 0],
        ...         [0, 0, sarkit.wgs84.SEMI_MINOR_AXIS],
        ...     ]
        ... )
        array([[ 0.,  0.,  0.],
               [ 0., 90.,  0.],
               [90.,  0.,  0.]])

    """
    xyz = np.asarray(xyz)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    llh = np.full(xyz.shape, np.nan, dtype=np.float64)

    r = np.sqrt((x * x) + (y * y))

    # Check for invalid solution
    valid = (A * r) * (A * r) + (B * z) * (B * z) > (_A2 - _B2) * (_A2 - _B2)

    # calculate intermediates
    f = 54.0 * _B2 * z * z  # not the WGS 84 flattening parameter
    g = r * r + _OME2 * z * z - FIRST_ECCENTRICITY_SQUARED * (_A2 - _B2)
    c = _E4 * f * r * r / (g * g * g)
    s = (1.0 + c + np.sqrt(c * c + 2 * c)) ** (1.0 / 3)
    p = f / (3.0 * (g * (s + 1.0 / s + 1.0)) ** 2)
    q = np.sqrt(1.0 + 2.0 * _E4 * p)
    r0 = -p * FIRST_ECCENTRICITY_SQUARED * r / (1.0 + q) + np.sqrt(
        np.abs(
            0.5 * _A2 * (1.0 + 1 / q)
            - p * _OME2 * z * z / (q * (1.0 + q))
            - 0.5 * p * r * r
        )
    )
    t = r - FIRST_ECCENTRICITY_SQUARED * r0
    u = np.sqrt(t * t + z * z)
    v = np.sqrt(t * t + _OME2 * z * z)
    z0 = _B2 * z / (A * v)

    # calculate latitude
    llh[valid, 0] = np.rad2deg(
        np.arctan2(z[valid] + SECOND_ECCENTRICITY_SQUARED * z0[valid], r[valid])
    )
    # calculate longitude
    llh[valid, 1] = np.rad2deg(np.arctan2(y[valid], x[valid]))
    # calculate ellipsoidal height
    llh[valid, 2] = u[valid] * (1.0 - _B2 / (A * v[valid]))
    return llh


def geodetic_to_cartesian(latlonhae: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts WGS 84 geodetic coordinates to cartesian coordinates.

    Parameters
    ----------
    latlonhae : (..., 3) array_like
        Array of geodetic coordinates with [latitude (deg), longitude (deg), and ellipsoidal height (m)] in the last
        dimension.

    Returns
    -------
    (..., 3) ndarray
        Array of cartesian coordinates with X, Y, Z components in meters in the last dimension.

    Examples
    --------
    .. doctest::

        >>> import sarkit.wgs84
        >>> sarkit.wgs84.geodetic_to_cartesian([[0,0,0], [0,90,0], [90,0,0]]).round(3)
        array([[6378137.   ,       0.   ,       0.   ],
               [      0.   , 6378137.   ,       0.   ],
               [      0.   ,       0.   , 6356752.314]])
    """
    latlonhae = np.asarray(latlonhae)
    lat = np.deg2rad(latlonhae[..., 0])
    lon = np.deg2rad(latlonhae[..., 1])
    hae = latlonhae[..., 2]

    out = np.full(latlonhae.shape, np.nan, dtype=np.float64)
    # calculate distance to surface of ellipsoid
    r = A / np.sqrt(1.0 - FIRST_ECCENTRICITY_SQUARED * np.sin(lat) * np.sin(lat))

    # calculate coordinates
    out[..., 0] = (r + hae) * np.cos(lat) * np.cos(lon)
    out[..., 1] = (r + hae) * np.cos(lat) * np.sin(lon)
    out[..., 2] = (r + hae - FIRST_ECCENTRICITY_SQUARED * r) * np.sin(lat)
    return out


def up(latlonhae: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute local up unit vectors from WGS 84 geodetic coordinates.

    Parameters
    ----------
    latlonhae : (..., 3) array_like
        Array of geodetic coordinates with [latitude (deg), longitude (deg), and ellipsoidal height (m)] in the last
        dimension.

    Returns
    -------
    (..., 3) ndarray
        Array of local up unit vectors perpendicular to the local WGS-84 inflated ellipsoid with X, Y, Z components
        in meters in the last dimension.

    Examples
    --------
    .. doctest::

        >>> import sarkit.wgs84
        >>> sarkit.wgs84.up([[0,0,0], [0,90,0], [90,0,0]]).round(6)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
    """
    latlonhae = np.asarray(latlonhae)
    lat = np.deg2rad(latlonhae[..., 0])
    lon = np.deg2rad(latlonhae[..., 1])
    return np.stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ],
        axis=-1,
    )


def north(latlonhae: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute local north unit vectors from WGS 84 geodetic coordinates.

    Parameters
    ----------
    latlonhae : (..., 3) array_like
        Array of geodetic coordinates with [latitude (deg), longitude (deg), and ellipsoidal height (m)] in the last
        dimension.

    Returns
    -------
    (..., 3) ndarray
        Array of local north unit vectors with X, Y, Z components in meters in the last dimension.

    Examples
    --------
    .. doctest::

        >>> import sarkit.wgs84
        >>> sarkit.wgs84.north([[0,0,0], [0,90,0], [90,0,0]]).round(6)
        array([[-0., -0.,  1.],
               [-0., -0.,  1.],
               [-1., -0.,  0.]])
    """
    latlonhae = np.asarray(latlonhae)
    lat = np.deg2rad(latlonhae[..., 0])
    lon = np.deg2rad(latlonhae[..., 1])
    return np.stack(
        [
            -np.sin(lat) * np.cos(lon),
            -np.sin(lat) * np.sin(lon),
            np.cos(lat),
        ],
        axis=-1,
    )


def east(latlonhae: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute local east unit vectors from WGS 84 geodetic coordinates.

    Parameters
    ----------
    latlonhae : (..., 3) array_like
        Array of geodetic coordinates with [latitude (deg), longitude (deg), and ellipsoidal height (m)] in the last
        dimension.

    Returns
    -------
    (..., 3) ndarray
        Array of local east unit vectors with X, Y, Z components in meters in the last dimension.

    Examples
    --------
    .. doctest::

        >>> import sarkit.wgs84
        >>> sarkit.wgs84.east([[0,0,0], [0,90,0], [90,0,0]]).round(6)
        array([[-0.,  1.,  0.],
               [-1.,  0.,  0.],
               [-0.,  1.,  0.]])
    """
    latlonhae = np.asarray(latlonhae)
    lon = np.deg2rad(latlonhae[..., 1])
    return np.stack(
        [
            -np.sin(lon),
            np.cos(lon),
            np.zeros_like(lon),
        ],
        axis=-1,
    )
