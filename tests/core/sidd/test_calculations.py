import pathlib

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import pytest

import sarkit.sidd as sksidd
import sarkit.sidd.calculations as sidd_calc

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def test_coordinate_system_type():
    sidd_xmlfile = DATAPATH / "example-sidd-3.0.0.xml"
    sidd_xmltree = lxml.etree.parse(sidd_xmlfile)
    assert sidd_xmltree.find("./{*}Measurement/{*}PlaneProjection") is not None
    sidd_calc.get_coordinate_system_type(sidd_xmltree) == sidd_calc.CoordinateSystem.PGD


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
