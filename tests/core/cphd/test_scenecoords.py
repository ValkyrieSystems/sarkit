import pathlib

import lxml.etree
import numpy as np
import pytest

import sarkit.cphd as skcphd

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


@pytest.mark.parametrize(
    ("surf_type", "cphd_xmlpath"),
    [
        ("Planar", DATAPATH / "example-cphd-1.0.1.xml"),
    ],
)
def test_ecf_to_from_iac(surf_type, cphd_xmlpath):
    cphd_xmltree = lxml.etree.parse(cphd_xmlpath)
    cphd_ew = skcphd.ElementWrapper(cphd_xmltree.getroot())
    sc_ew = cphd_ew["SceneCoordinates"]
    assert surf_type in sc_ew["ReferenceSurface"]

    rng = np.random.default_rng()

    pt_iacs = 24 * rng.random((6, 5, 4, 3))
    pt_ecf_from_iac = skcphd.iac_to_ecf(cphd_xmltree, pt_iacs)

    pt_iac_from_ecf = skcphd.ecf_to_iac(cphd_xmltree, pt_ecf_from_iac)
    assert np.allclose(pt_iacs, pt_iac_from_ecf)

    # check 2d iac_to_ecf case
    assert np.array_equal(
        skcphd.iac_to_ecf(cphd_xmltree, pt_iacs[..., :2]),
        skcphd.iac_to_ecf(cphd_xmltree, pt_iacs * [1, 1, 0]),
    )
