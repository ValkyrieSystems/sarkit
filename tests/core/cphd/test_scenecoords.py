import copy
import pathlib

import lxml.etree
import numpy as np
import pytest
import shapely

import sarkit.cphd as skcphd
import sarkit.wgs84
import tests.utils

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def test_planar_ecf_tofrom_iac():
    rng = np.random.default_rng()
    iarp = [1e6, -2e6, 3e6]
    a = rng.random((3, 3))
    q, _ = np.linalg.qr(a)
    uiax = q[:, 0]
    uiay = q[:, 1]

    pt = iarp + 1000 * rng.random((6, 5, 4, 3))
    pt_iac_from_ecf = skcphd.planar_ecf_to_iac(pt, iarp, uiax, uiay)

    pt_ecf_from_iac = skcphd.planar_iac_to_ecf(pt_iac_from_ecf, iarp, uiax, uiay)
    assert np.allclose(pt, pt_ecf_from_iac)

    # check 2d iac_to_ecf case
    assert np.array_equal(
        skcphd.planar_iac_to_ecf(pt_iac_from_ecf[..., :2], iarp, uiax, uiay),
        skcphd.planar_iac_to_ecf(pt_iac_from_ecf * [1, 1, 0], iarp, uiax, uiay),
    )


def test_hae_llh_tofrom_iac():
    rng = np.random.default_rng()
    iarp_llh = [34.043, -118.267, 71]

    # from CPHD v1.1.0, section 6.1
    iarp = sarkit.wgs84.geodetic_to_cartesian(iarp_llh)
    iarp_xy = np.linalg.norm(iarp[:2])
    delta_lon_e = 1 / iarp_xy
    delta_dist_n = 1.0
    n1 = iarp + delta_dist_n * sarkit.wgs84.north(iarp_llh)
    n1_llh = sarkit.wgs84.cartesian_to_geodetic(n1)
    delta_lat_n = np.deg2rad(n1_llh[0] - iarp_llh[0]) / delta_dist_n
    ueast_ll = np.array([0.0, delta_lon_e])
    unor_ll = np.array([delta_lat_n, 0.0])

    # from CPHD v1.1.0, section 6.2
    theta_iax = np.deg2rad(24.0)
    uiax_ll = np.cos(theta_iax) * unor_ll + np.sin(theta_iax) * ueast_ll
    uiay_ll = np.sin(theta_iax) * unor_ll - np.cos(theta_iax) * ueast_ll

    pt_iac = 1000 * rng.random((6, 5, 4, 3))
    pt_llh_from_iac = skcphd.hae_iac_to_llh(pt_iac, iarp_llh, uiax_ll, uiay_ll)

    pt_iac_from_llh = skcphd.hae_llh_to_iac(pt_llh_from_iac, iarp_llh, uiax_ll, uiay_ll)
    assert np.allclose(pt_iac, pt_iac_from_llh)

    # check 2d iac_to_ecf case
    assert np.array_equal(
        skcphd.hae_iac_to_llh(pt_iac[..., :2], iarp_llh, uiax_ll, uiay_ll),
        skcphd.hae_iac_to_llh(pt_iac * [1, 1, 0], iarp_llh, uiax_ll, uiay_ll),
    )


def get_planar_xmltree():
    return lxml.etree.parse(DATAPATH / "example-cphd-1.1.0.xml")


def get_hae_xmltree():
    cphd_xmltree = get_planar_xmltree()
    tests.utils.replace_planar_with_hae(skcphd.ElementWrapper(cphd_xmltree.getroot()))
    return cphd_xmltree


@pytest.mark.parametrize(
    ("surf_type", "cphd_xmltree_func"),
    [
        ("Planar", get_planar_xmltree),
        ("HAE", get_hae_xmltree),
    ],
)
def test_derived_tofrom_iac(surf_type, cphd_xmltree_func):
    """Check the derived xmltree-based IAC to/from ecf & llh methods"""
    cphd_xmltree = cphd_xmltree_func()
    cphd_ew = skcphd.ElementWrapper(cphd_xmltree.getroot())
    sc_ew = cphd_ew["SceneCoordinates"]
    assert surf_type in sc_ew["ReferenceSurface"]

    rng = np.random.default_rng()

    pt_iacs = 24 * rng.random((6, 5, 4, 3))

    # to/from ecf
    pt_ecf_from_iac = skcphd.iac_to_ecf(cphd_xmltree, pt_iacs)
    pt_iac_from_ecf = skcphd.ecf_to_iac(cphd_xmltree, pt_ecf_from_iac)
    assert np.allclose(pt_iacs, pt_iac_from_ecf)

    # to/from llh
    pt_llh_from_iac = skcphd.iac_to_llh(cphd_xmltree, pt_iacs)
    pt_iac_from_llh = skcphd.llh_to_iac(cphd_xmltree, pt_llh_from_iac)
    assert np.allclose(pt_iacs, pt_iac_from_llh)
    assert np.allclose(
        pt_ecf_from_iac, sarkit.wgs84.geodetic_to_cartesian(pt_llh_from_iac)
    )

    # check 2d cases
    assert np.array_equal(
        skcphd.iac_to_ecf(cphd_xmltree, pt_iacs[..., :2]),
        skcphd.iac_to_ecf(cphd_xmltree, pt_iacs * [1, 1, 0]),
    )
    assert np.array_equal(
        skcphd.iac_to_llh(cphd_xmltree, pt_iacs[..., :2]),
        skcphd.iac_to_llh(cphd_xmltree, pt_iacs * [1, 1, 0]),
    )


@pytest.mark.parametrize("has_polygon", (True, False))
@pytest.mark.parametrize("use_polygon", (True, False))
def test_image_area_funcs(has_polygon, use_polygon):
    xmltree = lxml.etree.parse(DATAPATH / "example-cphd-1.1.0.xml")

    extended_poly = shapely.Polygon([[-10, -10], [0, 20], [10, -10]])
    scene_poly = shapely.buffer(extended_poly, -1)
    ch0_poly = shapely.buffer(scene_poly, -1)
    ch1_poly = shapely.buffer(ch0_poly, -1)

    def set_ia(ia_ew: skcphd.ElementWrapper, polygon: shapely.Polygon):
        ia_ew["X1Y1"] = [polygon.bounds[0], polygon.bounds[1]]
        ia_ew["X2Y2"] = [polygon.bounds[2], polygon.bounds[3]]
        del ia_ew["Polygon"]
        if has_polygon:
            ia_ew["Polygon"] = shapely.get_coordinates(polygon)[:-1, :]

    ew = skcphd.ElementWrapper(xmltree.getroot())
    set_ia(ew["SceneCoordinates"]["ExtendedArea"], extended_poly)
    set_ia(ew["SceneCoordinates"]["ImageArea"], scene_poly)
    chpar0 = ew["Channel"]["Parameters"][0]
    set_ia(chpar0["ImageArea"], ch0_poly)
    chpar1 = copy.deepcopy(chpar0)
    chpar1["Identifier"] = "chpar1_id"  # assume this is unique
    set_ia(chpar1["ImageArea"], ch1_poly)
    ew["Channel"].add("Parameters", chpar1)
    chpar2 = copy.deepcopy(chpar0)
    chpar2["Identifier"] = "chpar2_id"  # assume this is unique
    del chpar2["ImageArea"]
    ew["Channel"].add("Parameters", chpar2)

    def check_imgarea(actual, expected):
        if has_polygon and use_polygon:
            assert shapely.equals(shapely.Polygon(actual), expected)
        else:
            assert shapely.equals(shapely.Polygon(actual), expected.envelope)

    check_imgarea(
        skcphd.get_channel_image_area(
            xmltree, chpar0["Identifier"], use_polygon=use_polygon
        ),
        ch0_poly,
    )
    check_imgarea(
        skcphd.get_channel_image_area(
            xmltree, chpar1["Identifier"], use_polygon=use_polygon
        ),
        ch1_poly,
    )
    check_imgarea(
        skcphd.get_channel_image_area(
            xmltree, chpar2["Identifier"], use_polygon=use_polygon
        ),
        scene_poly,
    )

    check_imgarea(
        skcphd.get_scene_image_area(xmltree, use_polygon=use_polygon), scene_poly
    )

    check_imgarea(
        skcphd.get_extended_image_area(xmltree, use_polygon=use_polygon), extended_poly
    )

    del ew["SceneCoordinates"]["ExtendedArea"]
    assert skcphd.get_extended_image_area(xmltree, use_polygon=use_polygon) is None
