import pathlib

import lxml.etree
import numpy as np
import numpy.testing as npt
import pytest

import sarkit.sidd as sksidd
from tests.core import testing

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def test_anglemagnitude():
    data = np.random.default_rng().random((2,))
    elem = lxml.etree.Element("{faux-ns}AngleMagnitude")
    type_obj = sksidd.AngleMagnitudeType()
    type_obj.set_elem(elem, data)
    assert np.array_equal(type_obj.parse_elem(elem), data)


def test_filtercoefficient():
    data = np.random.default_rng().random((4, 7))
    elem = lxml.etree.Element("{faux-ns}FilterCoefficients")
    type_obj = sksidd.FilterCoefficientType("rowcol")
    type_obj.set_elem(elem, data)
    assert np.array_equal(type_obj.parse_elem(elem), data)
    type_obj = sksidd.FilterCoefficientType("phasingpoint")
    type_obj.set_elem(elem, data)
    assert np.array_equal(type_obj.parse_elem(elem), data)


def test_intlist():
    data = np.random.default_rng().integers(256, size=11)
    elem = lxml.etree.Element("{faux-ns}IntList")
    type_obj = sksidd.IntListType()
    type_obj.set_elem(elem, data)
    assert np.array_equal(type_obj.parse_elem(elem), data)


def test_image_corners_type():
    data = np.array(
        [
            [-1.23, -4.56],
            [-7.89, -10.11],
            [16.17, 18.19],
            [12.13, 14.15],
        ]
    )
    elem = lxml.etree.Element("{faux-ns}ImageCorners")
    type_obj = sksidd.ImageCornersType()
    type_obj.set_elem(elem, data)
    assert np.array_equal(type_obj.parse_elem(elem), data)


def test_rangeazimuth():
    data = np.random.default_rng().random((2,))
    elem = lxml.etree.Element("{faux-ns}RangeAzimuth")
    type_obj = sksidd.RangeAzimuthType()
    type_obj.set_elem(elem, data)
    assert np.array_equal(type_obj.parse_elem(elem), data)


def test_rowcoldble():
    data = np.random.default_rng().random((2,))
    elem = lxml.etree.Element("{faux-ns}RowColDbl")
    type_obj = sksidd.RowColDblType()
    type_obj.set_elem(elem, data)
    assert np.array_equal(type_obj.parse_elem(elem), data)


def test_sfapointtype():
    data = [1.1, 1.2, 1.3]
    elem = sksidd.SfaPointType().make_elem("{ns}SfaPoint", data)
    assert np.array_equal(sksidd.SfaPointType().parse_elem(elem), data)
    sksidd.SfaPointType().set_elem(elem, data[:-1])
    assert np.array_equal(sksidd.SfaPointType().parse_elem(elem), data[:-1])


def test_transcoders():
    no_transcode_leaf = set()
    for xml_file in (DATAPATH / "syntax_only/sidd").glob("*.xml"):
        etree = lxml.etree.parse(xml_file)
        basis_version = lxml.etree.QName(etree.getroot()).namespace
        schema = lxml.etree.XMLSchema(file=sksidd.VERSION_INFO[basis_version]["schema"])
        schema.assertValid(etree)
        xml_helper = sksidd.XmlHelper(etree)
        for elem in reversed(list(xml_helper.element_tree.iter())):
            try:
                val = xml_helper.load_elem(elem)
                xml_helper.set_elem(elem, val)
                schema.assertValid(xml_helper.element_tree)
                np.testing.assert_equal(xml_helper.load_elem(elem), val)
            except LookupError:
                if len(elem) == 0:
                    no_transcode_leaf.add(xml_helper.element_tree.getelementpath(elem))

    todos = {xmlpath for xmlpath in no_transcode_leaf if "Classification" in xmlpath}
    assert not (no_transcode_leaf - todos)


@pytest.mark.parametrize(
    "xmlpath",
    list((DATAPATH / "syntax_only/sidd").glob("*.xml"))
    + list(DATAPATH.glob("example-sidd*.xml")),
)
def test_elementwrapper_tofromdict(xmlpath):
    siddroot = lxml.etree.parse(xmlpath).getroot()
    root_ns = lxml.etree.QName(siddroot).namespace
    xsdhelp = sksidd.XsdHelper(root_ns)
    wrapped_siddroot = sksidd.ElementWrapper(siddroot)

    dict1 = wrapped_siddroot.to_dict()
    wrapped_root_fromdict = sksidd.ElementWrapper(
        lxml.etree.Element(lxml.etree.QName(root_ns, "SIDD")),
    )
    wrapped_root_fromdict.from_dict(dict1)
    dict2 = wrapped_root_fromdict.to_dict()

    npt.assert_equal(dict1, dict2)
    assert testing.elem_cmp(siddroot, wrapped_root_fromdict.elem, xsdhelp)
