import copy
import pathlib

import lxml.etree
import numpy as np
import numpy.testing as npt
import pytest

import sarkit.sicd as sksicd
from tests.core import testing

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def test_image_corners_type():
    etree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    xml_helper = sksicd.XmlHelper(etree)
    schema = lxml.etree.XMLSchema(file=sksicd.VERSION_INFO["urn:SICD:1.3.0"]["schema"])
    schema.assertValid(etree)

    new_corner_coords = np.array(
        [
            [-1.23, -4.56],
            [-7.89, -10.11],
            [16.17, 18.19],
            [12.13, 14.15],
        ]
    )
    xml_helper.set("./{*}GeoData/{*}ImageCorners", new_corner_coords)
    schema.assertValid(xml_helper.element_tree)
    assert np.array_equal(
        xml_helper.load("./{*}GeoData/{*}ImageCorners"), new_corner_coords
    )

    new_elem = sksicd.ImageCornersType().make_elem("FauxIC", new_corner_coords)
    assert np.array_equal(
        sksicd.ImageCornersType().parse_elem(new_elem),
        new_corner_coords,
    )


def test_transcoders():
    no_transcode_leaf = set()
    for xml_file in (DATAPATH / "syntax_only/sicd").glob("*.xml"):
        etree = lxml.etree.parse(xml_file)
        basis_version = lxml.etree.QName(etree.getroot()).namespace
        schema = lxml.etree.XMLSchema(file=sksicd.VERSION_INFO[basis_version]["schema"])
        schema.assertValid(etree)
        xml_helper = sksicd.XmlHelper(etree)
        for elem in reversed(list(xml_helper.element_tree.iter())):
            try:
                val = xml_helper.load_elem(elem)
                xml_helper.set_elem(elem, val)
                schema.assertValid(xml_helper.element_tree)
                np.testing.assert_equal(xml_helper.load_elem(elem), val)
            except LookupError:
                if len(elem) == 0:
                    no_transcode_leaf.add(xml_helper.element_tree.getelementpath(elem))
    assert not no_transcode_leaf


@pytest.mark.parametrize(
    "xmlpath",
    list((DATAPATH / "syntax_only/sicd").glob("*.xml"))
    + list(DATAPATH.glob("example-sicd*.xml")),
)
def test_elementwrapper_tofromdict(xmlpath):
    xmlroot = lxml.etree.parse(xmlpath).getroot()
    root_ns = lxml.etree.QName(xmlroot).namespace
    xsdhelp = sksicd.XsdHelper(root_ns)
    wrapped_sicdroot = sksicd.ElementWrapper(xmlroot)

    dict1 = wrapped_sicdroot.to_dict()
    wrapped_root_fromdict = sksicd.ElementWrapper(
        lxml.etree.Element(xmlroot.tag),
    )
    wrapped_root_fromdict.from_dict(dict1)
    dict2 = wrapped_root_fromdict.to_dict()

    npt.assert_equal(dict1, dict2)
    assert testing.elem_cmp(xmlroot, wrapped_root_fromdict.elem, xsdhelp)


def _replace_scpcoa(sicd_xmltree):
    sicd_xmltree.find(".//{*}SCPCOA").clear()
    scpcoa = sksicd.compute_scp_coa(sicd_xmltree)
    sicd_xmltree.getroot().replace(sicd_xmltree.find(".//{*}SCPCOA"), scpcoa)
    basis_version = lxml.etree.QName(sicd_xmltree.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=sksicd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(sicd_xmltree)
    return scpcoa


@pytest.mark.parametrize("xml_file", DATAPATH.glob("example-sicd*.xml"))
def test_compute_scp_coa(xml_file):
    _replace_scpcoa(lxml.etree.parse(xml_file))


def test_compute_scp_coa_bistatic():
    etree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    # Monostatic
    assert etree.findtext("./{*}CollectionInfo/{*}CollectType") == "MONOSTATIC"
    scpcoa_mono = _replace_scpcoa(copy.deepcopy(etree))
    assert scpcoa_mono.find(".//{*}Bistatic") is None

    # Bistatic
    etree_bistatic = copy.deepcopy(etree)
    for elem in etree_bistatic.iter():
        elem.tag = f"{{urn:SICD:1.4.0}}{lxml.etree.QName(elem).localname}"
    xmlhelp_bistatic = sksicd.XmlHelper(etree_bistatic)
    xmlhelp_bistatic.set("./{*}CollectionInfo/{*}CollectType", "BISTATIC")
    scpcoa_bistatic_diff = _replace_scpcoa(etree_bistatic)
    assert scpcoa_bistatic_diff.find(".//{*}Bistatic") is not None
