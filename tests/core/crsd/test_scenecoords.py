import copy
import pathlib

import lxml.etree
import numpy as np
import pytest
import shapely

import sarkit.crsd as skcrsd
import sarkit.wgs84
import tests.utils

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def get_planar_xmltree():
    return lxml.etree.parse(DATAPATH / "example-crsd-1.0.xml")


def get_hae_xmltree():
    xmltree = get_planar_xmltree()
    tests.utils.replace_planar_with_hae(skcrsd.ElementWrapper(xmltree.getroot()))
    return xmltree


@pytest.mark.parametrize(
    ("surf_type", "xmltree_func"),
    [
        ("Planar", get_planar_xmltree),
        ("HAE", get_hae_xmltree),
    ],
)
def test_derived_tofrom_iac(surf_type, xmltree_func):
    """Check the derived xmltree-based IAC to/from ecf & llh methods"""
    xmltree = xmltree_func()
    ew = skcrsd.ElementWrapper(xmltree.getroot())
    sc_ew = ew["SceneCoordinates"]
    assert surf_type in sc_ew["ReferenceSurface"]

    rng = np.random.default_rng()

    pt_iacs = 24 * rng.random((6, 5, 4, 3))

    # to/from ecf
    pt_ecf_from_iac = skcrsd.iac_to_ecf(xmltree, pt_iacs)
    pt_iac_from_ecf = skcrsd.ecf_to_iac(xmltree, pt_ecf_from_iac)
    assert np.allclose(pt_iacs, pt_iac_from_ecf)

    # to/from llh
    pt_llh_from_iac = skcrsd.iac_to_llh(xmltree, pt_iacs)
    pt_iac_from_llh = skcrsd.llh_to_iac(xmltree, pt_llh_from_iac)
    assert np.allclose(pt_iacs, pt_iac_from_llh)
    assert np.allclose(
        pt_ecf_from_iac, sarkit.wgs84.geodetic_to_cartesian(pt_llh_from_iac)
    )

    # check 2d cases
    assert np.array_equal(
        skcrsd.iac_to_ecf(xmltree, pt_iacs[..., :2]),
        skcrsd.iac_to_ecf(xmltree, pt_iacs * [1, 1, 0]),
    )
    assert np.array_equal(
        skcrsd.iac_to_llh(xmltree, pt_iacs[..., :2]),
        skcrsd.iac_to_llh(xmltree, pt_iacs * [1, 1, 0]),
    )


@pytest.mark.parametrize("crsd_type", ("CRSDsar", "CRSDrcv"))
@pytest.mark.parametrize("use_polygon", (True, False))
def test_image_area_funcs(crsd_type, use_polygon):
    xmltree = lxml.etree.parse(DATAPATH / "example-crsd-1.0.xml")
    if crsd_type == "CRSDrcv":
        tests.utils.crsdsar_xml_to_crsdrcv(xmltree)

    extended_poly = shapely.Polygon([[-10, -10], [0, 20], [10, -10]])
    scene_poly = shapely.buffer(extended_poly, -1)
    ch0_poly = shapely.buffer(scene_poly, -1)
    ch1_poly = shapely.buffer(ch0_poly, -1)

    def set_ia(ia_ew: skcrsd.ElementWrapper, polygon: shapely.Polygon):
        ia_ew["X1Y1"] = [polygon.bounds[0], polygon.bounds[1]]
        ia_ew["X2Y2"] = [polygon.bounds[2], polygon.bounds[3]]
        ia_ew["Polygon"] = shapely.get_coordinates(polygon)[:-1, :]

    ew = skcrsd.ElementWrapper(xmltree.getroot())
    set_ia(ew["SceneCoordinates"]["ImageArea"], scene_poly)

    def check_imgarea(actual, expected):
        if use_polygon:
            assert shapely.equals(shapely.Polygon(actual), expected)
        else:
            assert shapely.equals(shapely.Polygon(actual), expected.envelope)

    check_imgarea(
        skcrsd.get_scene_image_area(xmltree, use_polygon=use_polygon), scene_poly
    )

    if crsd_type == "CRSDsar":
        set_ia(ew["SceneCoordinates"]["ExtendedArea"], extended_poly)
        chpar0 = ew["Channel"]["Parameters"][0]
        set_ia(chpar0["SARImage"]["ImageArea"], ch0_poly)
        chpar1 = copy.deepcopy(chpar0)
        chpar1["Identifier"] = "chpar1_id"  # assume this is unique
        set_ia(chpar1["SARImage"]["ImageArea"], ch1_poly)
        ew["Channel"].add("Parameters", chpar1)
        chpar2 = copy.deepcopy(chpar0)
        chpar2["Identifier"] = "chpar2_id"  # assume this is unique
        del chpar2["SARImage"]["ImageArea"]
        ew["Channel"].add("Parameters", chpar2)

        check_imgarea(
            skcrsd.get_channel_image_area(
                xmltree, chpar0["Identifier"], use_polygon=use_polygon
            ),
            ch0_poly,
        )
        check_imgarea(
            skcrsd.get_channel_image_area(
                xmltree, chpar1["Identifier"], use_polygon=use_polygon
            ),
            ch1_poly,
        )
        check_imgarea(
            skcrsd.get_channel_image_area(
                xmltree, chpar2["Identifier"], use_polygon=use_polygon
            ),
            scene_poly,
        )

        check_imgarea(
            skcrsd.get_extended_image_area(xmltree, use_polygon=use_polygon),
            extended_poly,
        )

        del ew["SceneCoordinates"]["ExtendedArea"]
        assert skcrsd.get_extended_image_area(xmltree, use_polygon=use_polygon) is None
    else:
        with pytest.raises(ValueError, match="Only CRSDsar products"):
            skcrsd.get_channel_image_area(
                xmltree, xmltree.findtext(".//{*}RefChId"), use_polygon=use_polygon
            )
        with pytest.raises(ValueError, match="Only CRSDsar products"):
            skcrsd.get_extended_image_area(xmltree, use_polygon=use_polygon)
