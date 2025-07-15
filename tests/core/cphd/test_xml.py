import pathlib

import lxml.etree
import numpy as np
import numpy.testing as npt
import pytest

import sarkit.cphd as skcphd
from tests.core import testing

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def test_pvp():
    pvp_dict = {"Offset": 11, "Size": 1, "dtype": np.dtype("float64")}
    elem = skcphd.PvpType().make_elem("{faux-ns}PvpNode", pvp_dict)
    assert skcphd.PvpType().parse_elem(elem) == pvp_dict

    with pytest.raises(TypeError):
        pvp_dict = {"Offset": 11, "Size": 1, "dtype": "F8"}
        skcphd.PvpType().set_elem(elem, pvp_dict)

    with pytest.raises(ValueError):
        added_pvp_dict = {
            "Name": "ADDED_PVP",
            "Offset": 11,
            "Size": 1,
            "dtype": np.dtype("float64"),
        }
        skcphd.PvpType().set_elem(elem, added_pvp_dict)


def test_addedpvp():
    added_pvp_dict = {
        "Name": "ADDED_PVP",
        "Offset": 11,
        "Size": 1,
        "dtype": np.dtype("float64"),
    }
    elem = skcphd.AddedPvpType().make_elem("{faux-ns}AddedPvpNode", added_pvp_dict)
    assert skcphd.AddedPvpType().parse_elem(elem) == added_pvp_dict


def test_transcoders():
    no_transcode_leaf = set()
    for xml_file in (DATAPATH / "syntax_only/cphd").glob("*.xml"):
        etree = lxml.etree.parse(xml_file)
        basis_version = lxml.etree.QName(etree.getroot()).namespace
        schema = lxml.etree.XMLSchema(file=skcphd.VERSION_INFO[basis_version]["schema"])
        schema.assertValid(etree)
        xml_helper = skcphd.XmlHelper(etree)
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
    list((DATAPATH / "syntax_only/cphd").glob("*.xml"))
    + list(DATAPATH.glob("example-cphd*.xml")),
)
def test_elementwrapper_tofromdict(xmlpath):
    xmlroot = lxml.etree.parse(xmlpath).getroot()
    root_ns = lxml.etree.QName(xmlroot).namespace
    xsdhelp = skcphd.XsdHelper(root_ns)
    wrapped_cphdroot = skcphd.ElementWrapper(xmlroot)

    dict1 = wrapped_cphdroot.to_dict()
    wrapped_root_fromdict = skcphd.ElementWrapper(lxml.etree.Element(xmlroot.tag))
    wrapped_root_fromdict.from_dict(dict1)
    dict2 = wrapped_root_fromdict.to_dict()

    npt.assert_equal(dict1, dict2)
    assert testing.elem_cmp(xmlroot, wrapped_root_fromdict.elem, xsdhelp)
