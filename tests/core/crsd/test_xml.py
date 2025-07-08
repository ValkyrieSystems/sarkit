import pathlib

import lxml.etree
import numpy as np
import numpy.testing as npt
import pytest

import sarkit.crsd as skcrsd
from tests.core import testing

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def test_transcoders():
    no_transcode_leaf = set()
    for xml_file in (DATAPATH / "syntax_only/crsd").glob("*.xml"):
        etree = lxml.etree.parse(xml_file)
        basis_version = lxml.etree.QName(etree.getroot()).namespace
        schema = lxml.etree.XMLSchema(file=skcrsd.VERSION_INFO[basis_version]["schema"])
        schema.assertValid(etree)
        xml_helper = skcrsd.XmlHelper(etree)
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
    list((DATAPATH / "syntax_only/crsd").glob("*.xml"))
    + list(DATAPATH.glob("example-crsd*.xml")),
)
def test_elementwrapper_tofromdict(xmlpath):
    xmlroot = lxml.etree.parse(xmlpath).getroot()
    root_ns = lxml.etree.QName(xmlroot).namespace
    xsdhelp = skcrsd.XsdHelper(root_ns)
    wrapped_crsdroot = skcrsd.ElementWrapper(xmlroot)

    dict1 = wrapped_crsdroot.to_dict()
    wrapped_root_fromdict = skcrsd.ElementWrapper(
        lxml.etree.Element(xmlroot.tag),
    )
    wrapped_root_fromdict.from_dict(dict1)
    dict2 = wrapped_root_fromdict.to_dict()

    npt.assert_equal(dict1, dict2)
    assert testing.elem_cmp(xmlroot, wrapped_root_fromdict.elem, xsdhelp)
