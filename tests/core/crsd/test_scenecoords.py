import pathlib

import lxml.etree
import numpy as np
import pytest

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
