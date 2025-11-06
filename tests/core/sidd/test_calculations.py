import pathlib

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import pytest

import sarkit.sidd as sksidd
import sarkit.sidd.calculations as sidd_calc
import sarkit.wgs84

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def _two_dim_poly_fit(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_order: int = 2,
    y_order: int = 2,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
):
    # based on sarpy.io.complex.utils
    x = x.flatten() * x_scale
    y = y.flatten() * y_scale
    z = z.flatten()
    # first, we need to formulate this as a*t = z
    # where a has shape (x.size, (x_order+1)*(y_order+1))
    # and t has shape ((x_order+1)*(y_order+1), )
    a = np.empty((x.size, (x_order + 1) * (y_order + 1)), dtype=np.float64)
    # noinspection PyTypeChecker
    for i, index in enumerate(np.ndindex((x_order + 1, y_order + 1))):
        a[:, i] = np.power(x, index[0]) * np.power(y, index[1])
    # perform least squares fit
    sol, residuals, rank, sing_values = np.linalg.lstsq(a, z)
    if isinstance(residuals, (np.ndarray, np.number)):
        residuals /= float(x.size)
    sol = (
        np.power(x_scale, np.arange(x_order + 1))[:, np.newaxis]
        * np.reshape(sol, (x_order + 1, y_order + 1))
        * np.power(y_scale, np.arange(y_order + 1))
    )
    return sol, residuals, rank, sing_values


def test_coordinate_system_type():
    sidd_xmlfile = DATAPATH / "example-sidd-3.0.0.xml"
    sidd_xmltree = lxml.etree.parse(sidd_xmlfile)
    assert sidd_xmltree.find("./{*}Measurement/{*}PlaneProjection") is not None
    assert (
        sidd_calc.get_coordinate_system_type(sidd_xmltree)
        == sidd_calc.CoordinateSystem.PGD
    )


def test_pgd():
    sidd_xmlfile = DATAPATH / "example-sidd-3.0.0.xml"
    sidd_xmltree = lxml.etree.parse(sidd_xmlfile)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())

    ref_pt_rc = siddew["Measurement"]["PlaneProjection"]["ReferencePoint"]["Point"]
    p_ecef = sidd_calc.pgd_pixel_to_ecef(sidd_xmltree, ref_pt_rc)
    np.testing.assert_allclose(
        p_ecef, siddew["Measurement"]["PlaneProjection"]["ReferencePoint"]["ECEF"]
    )

    roundtrip = sidd_calc.ecef_to_pgd_pixel(sidd_xmltree, p_ecef)
    np.testing.assert_allclose(ref_pt_rc, roundtrip)

    pts = np.stack(
        np.meshgrid(
            np.linspace(0, siddew["Measurement"]["PixelFootprint"][0], 11),
            np.linspace(0, siddew["Measurement"]["PixelFootprint"][1], 11),
        ),
        axis=-1,
    )
    ecef = sidd_calc.pixel_to_ecef(sidd_xmltree, pts)
    roundtrip = sidd_calc.ecef_to_pixel(sidd_xmltree, ecef)
    np.testing.assert_allclose(pts, roundtrip, atol=1e-6)


def test_ggd():
    sidd_xmlfile = DATAPATH / "example-sidd-3.0.0.xml"
    sidd_xmltree = lxml.etree.parse(sidd_xmlfile)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())

    siddew["Measurement"]["GeographicProjection"] = {
        "ReferencePoint": siddew["Measurement"]["PlaneProjection"]["ReferencePoint"],
        "SampleSpacing": (0.001, 0.002),
        "TimeCOAPoly": siddew["Measurement"]["PlaneProjection"]["TimeCOAPoly"],
    }
    del siddew["Measurement"]["PlaneProjection"]

    ref_pt_rc = siddew["Measurement"]["GeographicProjection"]["ReferencePoint"]["Point"]
    llh = sidd_calc.ggd_pixel_to_geodetic(sidd_xmltree, ref_pt_rc)
    np.testing.assert_allclose(
        llh,
        sarkit.wgs84.cartesian_to_geodetic(
            siddew["Measurement"]["GeographicProjection"]["ReferencePoint"]["ECEF"]
        ),
    )

    roundtrip = sidd_calc.geodetic_to_ggd_pixel(sidd_xmltree, llh)
    np.testing.assert_allclose(ref_pt_rc, roundtrip)

    pts = np.stack(
        np.meshgrid(
            np.linspace(0, siddew["Measurement"]["PixelFootprint"][0], 11),
            np.linspace(0, siddew["Measurement"]["PixelFootprint"][1], 11),
        ),
        axis=-1,
    )

    ecef = sidd_calc.pixel_to_ecef(sidd_xmltree, pts)
    roundtrip = sidd_calc.ecef_to_pixel(sidd_xmltree, ecef)
    np.testing.assert_allclose(pts, roundtrip, atol=1e-6)


def test_cgd():
    sidd_xmlfile = DATAPATH / "example-sidd-3.0.0.xml"
    sidd_xmltree = lxml.etree.parse(sidd_xmlfile)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())

    siddew["Measurement"]["CylindricalProjection"] = {
        "ReferencePoint": siddew["Measurement"]["PlaneProjection"]["ReferencePoint"],
        "SampleSpacing": (0.2, 0.3),
        "TimeCOAPoly": siddew["Measurement"]["PlaneProjection"]["TimeCOAPoly"],
        "StripmapDirection": siddew["Measurement"]["PlaneProjection"]["ProductPlane"][
            "RowUnitVector"
        ],
    }
    del siddew["Measurement"]["PlaneProjection"]

    ref_pt_rc = siddew["Measurement"]["CylindricalProjection"]["ReferencePoint"][
        "Point"
    ]
    ecef = sidd_calc.cgd_pixel_to_ecef(sidd_xmltree, ref_pt_rc)
    np.testing.assert_allclose(
        ecef, siddew["Measurement"]["CylindricalProjection"]["ReferencePoint"]["ECEF"]
    )

    roundtrip = sidd_calc.ecef_to_cgd_pixel(sidd_xmltree, ecef)
    np.testing.assert_allclose(ref_pt_rc, roundtrip)

    pts = np.stack(
        np.meshgrid(
            np.linspace(0, siddew["Measurement"]["PixelFootprint"][0], 11),
            np.linspace(0, siddew["Measurement"]["PixelFootprint"][1], 11),
        ),
        axis=-1,
    )

    ecef = sidd_calc.pixel_to_ecef(sidd_xmltree, pts)
    roundtrip = sidd_calc.ecef_to_pixel(sidd_xmltree, ecef)
    np.testing.assert_allclose(pts, roundtrip, atol=1e-6)


def test_pfgd():
    sidd_xmlfile = DATAPATH / "example-sidd-3.0.0.xml"
    sidd_xmltree = lxml.etree.parse(sidd_xmlfile)
    siddew = sksidd.ElementWrapper(sidd_xmltree.getroot())

    pts = np.stack(
        np.meshgrid(
            np.linspace(0, siddew["Measurement"]["PixelFootprint"][0], 11),
            np.linspace(0, siddew["Measurement"]["PixelFootprint"][1], 11),
        ),
        axis=-1,
    )
    llh = sarkit.wgs84.cartesian_to_geodetic(sidd_calc.pixel_to_ecef(sidd_xmltree, pts))

    siddew["Measurement"]["PolynomialProjection"] = {
        "ReferencePoint": siddew["Measurement"]["PlaneProjection"]["ReferencePoint"],
        "RowColToLat": _two_dim_poly_fit(pts[..., 0], pts[..., 1], llh[..., 0])[0],
        "RowColToLon": _two_dim_poly_fit(pts[..., 0], pts[..., 1], llh[..., 1])[0],
        # "RowColToAlt" is optional
        "LatLonToRow": _two_dim_poly_fit(llh[..., 0], llh[..., 1], pts[..., 0])[0],
        "LatLonToCol": _two_dim_poly_fit(llh[..., 0], llh[..., 1], pts[..., 1])[0],
    }
    del siddew["Measurement"]["PlaneProjection"]

    ecef = sidd_calc.pixel_to_ecef(sidd_xmltree, pts)
    roundtrip = sidd_calc.ecef_to_pixel(sidd_xmltree, ecef)
    np.testing.assert_allclose(pts, roundtrip, atol=1e-5)

    siddew["Measurement"]["PolynomialProjection"]["RowColToAlt"] = [[10000]]
    ecef2 = sidd_calc.pixel_to_ecef(sidd_xmltree, pts)
    assert np.allclose(np.abs(np.linalg.norm(ecef2 - ecef, axis=-1)), 10000)


def test_coordinate_transform():
    sidd_xmlfile = DATAPATH / "example-sidd-3.0.0.xml"
    sidd_xmltree = lxml.etree.parse(sidd_xmlfile)
    sidd_helper = sksidd.XmlHelper(sidd_xmltree)

    ref_pt = sidd_helper.load(
        "./{*}Measurement/{*}PlaneProjection/{*}ReferencePoint/{*}Point"
    )
    pixel = [(0, 0), (2000, 1000), ref_pt]
    ecef = sidd_calc.pixel_to_ecef(sidd_xmltree, pixel)
    assert ecef.shape[-1] == 3
    np.testing.assert_almost_equal(
        ecef[-1],
        sidd_helper.load(
            "./{*}Measurement/{*}PlaneProjection/{*}ReferencePoint/{*}ECEF"
        ),
    )
    pixel_rt = sidd_calc.ecef_to_pixel(sidd_xmltree, ecef)
    np.testing.assert_almost_equal(pixel, pixel_rt)


def test_angles():
    sidd_xmlfile = DATAPATH / "example-sidd-3.0.0.xml"
    sidd_xmltree = lxml.etree.parse(sidd_xmlfile)
    sidd_helper = sksidd.XmlHelper(sidd_xmltree)

    tcoa_poly = sidd_helper.load("./{*}Measurement/{*}PlaneProjection/{*}TimeCOAPoly")
    arp_poly = sidd_helper.load("./{*}Measurement/{*}ARPPoly")

    angles = sidd_calc.compute_angles(
        sidd_helper.load(
            "./{*}Measurement/{*}PlaneProjection/{*}ReferencePoint/{*}ECEF"
        ),
        npp.polyval(tcoa_poly[0, 0], arp_poly),
        npp.polyval(tcoa_poly[0, 0], npp.polyder(arp_poly, 1)),
        sidd_helper.load(
            "./{*}Measurement/{*}PlaneProjection/{*}ProductPlane/{*}RowUnitVector"
        ),
        sidd_helper.load(
            "./{*}Measurement/{*}PlaneProjection/{*}ProductPlane/{*}ColUnitVector"
        ),
    )
    # Regression test against canned data
    col = "./{*}ExploitationFeatures/{*}Collection/"
    assert angles.Azimuth == pytest.approx(
        sidd_helper.load(col + "{*}Geometry/{*}Azimuth")
    )
    assert angles.Slope == pytest.approx(sidd_helper.load(col + "{*}Geometry/{*}Slope"))
    assert angles.DopplerCone == pytest.approx(
        sidd_helper.load(col + "{*}Geometry/{*}DopplerConeAngle")
    )
    assert angles.Squint == pytest.approx(
        sidd_helper.load(col + "{*}Geometry/{*}Squint")
    )
    assert angles.Graze == pytest.approx(sidd_helper.load(col + "{*}Geometry/{*}Graze"))
    assert angles.Tilt == pytest.approx(sidd_helper.load(col + "{*}Geometry/{*}Tilt"))

    assert angles.Shadow == pytest.approx(
        sidd_helper.load(col + "{*}Phenomenology/{*}Shadow/{*}Angle")
    )
    assert angles.ShadowMagnitude == pytest.approx(
        sidd_helper.load(col + "{*}Phenomenology/{*}Shadow/{*}Magnitude")
    )
    assert angles.Layover == pytest.approx(
        sidd_helper.load(col + "{*}Phenomenology/{*}Layover/{*}Angle")
    )
    assert angles.LayoverMagnitude == pytest.approx(
        sidd_helper.load(col + "{*}Phenomenology/{*}Layover/{*}Magnitude")
    )
    assert angles.MultiPath == pytest.approx(
        sidd_helper.load(col + "{*}Phenomenology/{*}MultiPath")
    )
    assert angles.GroundTrack == pytest.approx(
        sidd_helper.load(col + "{*}Phenomenology/{*}GroundTrack")
    )

    assert angles.North == pytest.approx(
        sidd_helper.load("./{*}ExploitationFeatures/{*}Product/{*}North")
    )
