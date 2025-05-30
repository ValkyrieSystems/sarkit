import copy
import datetime
import random

import lxml.etree
import numpy as np
import numpy.testing as npt
import pytest

import sarkit._xmlhelp as skxml
import sarkit._xmlhelp2
import sarkit.sidd._xmlhelp2


def test_xdt_naive():
    dt = datetime.datetime.now()
    elem = skxml.XdtType().make_elem("Xdt", dt)
    assert skxml.XdtType().parse_elem(elem) == dt.replace(tzinfo=datetime.timezone.utc)


def test_xdt_aware():
    dt = datetime.datetime.now(
        tz=datetime.timezone(offset=datetime.timedelta(hours=5.5))
    )
    elem = skxml.XdtType().make_elem("Xdt", dt)
    assert skxml.XdtType().parse_elem(elem) == dt


@pytest.mark.parametrize("ndim", (1, 2))
def test_poly(ndim):
    shape = np.arange(3, 3 + ndim)
    coefs = np.arange(np.prod(shape)).reshape(shape)
    polytype = skxml.PolyNdType(ndim)
    elem = polytype.make_elem("Poly", coefs)
    assert np.array_equal(polytype.parse_elem(elem), coefs)


def test_xyzpoly():
    coefs = np.linspace(-10, 10, 33).reshape((11, 3))
    elem = skxml.XyzPolyType().make_elem("{faux-ns}XyzPoly", coefs)
    assert np.array_equal(skxml.XyzPolyType().parse_elem(elem), coefs)


def test_xyz():
    xyz = [-10.0, 10.0, 0.20]
    elem = skxml.XyzType().make_elem("{faux-ns}XyzNode", xyz)
    assert np.array_equal(skxml.XyzType().parse_elem(elem), xyz)


def test_txt():
    elem = lxml.etree.Element("{faux-ns}Node")
    assert skxml.TxtType().parse_elem(elem) == ""
    new_str = "replacement string"
    new_elem = skxml.TxtType().make_elem("Txt", new_str)
    assert skxml.TxtType().parse_elem(new_elem) == new_str


@pytest.mark.parametrize("val", (True, False))
def test_bool(val):
    elem = skxml.BoolType().make_elem("node", val)
    assert skxml.BoolType().parse_elem(elem) == val


@pytest.mark.parametrize("val", (1.23, -4.56j, 1.23 - 4.56j))
def test_cmplx(val):
    elem = skxml.CmplxType().make_elem("node", val)
    assert skxml.CmplxType().parse_elem(elem) == val


def test_line_samp():
    ls_data = [1000, 2000]
    type_obj = skxml.LineSampType()
    elem = type_obj.make_elem("{faux-ns}LsNode", ls_data)
    assert np.array_equal(type_obj.parse_elem(elem), ls_data)


def test_array():
    data = np.random.default_rng().random((3,))
    elem = lxml.etree.Element("{faux-ns}ArrayDblNode")
    type_obj = skxml.ArrayType({c: skxml.DblType() for c in ("a", "b", "c")})
    type_obj.set_elem(elem, data)
    assert np.array_equal(type_obj.parse_elem(elem), data)
    with pytest.raises(ValueError, match="len.*does not match expected"):
        type_obj.set_elem(elem, np.tile(data, 2))


def test_xy():
    xy = [-10.0, 10.0]
    elem = skxml.XyType().make_elem("{faux-ns}XyNode", xy)
    assert np.array_equal(skxml.XyType().parse_elem(elem), xy)


def test_hex():
    hexval = b"\xba\xdd"
    elem = skxml.HexType().make_elem("{faux-ns}HexNode", hexval)
    assert np.array_equal(skxml.HexType().parse_elem(elem), hexval)


def test_parameter():
    name = "TestName"
    val = "TestVal"
    elem = skxml.ParameterType().make_elem("{faux-ns}Parameter", (name, val))
    assert skxml.ParameterType().parse_elem(elem) == (name, val)


def test_mtx_type():
    data = np.arange(np.prod(6)).reshape((2, 3))
    type_obj = skxml.MtxType(data.shape)
    elem = type_obj.make_elem("{faux-ns}MtxNode", data)
    assert np.array_equal(type_obj.parse_elem(elem), data)

    with pytest.raises(ValueError, match="shape.*does not match expected"):
        type_obj.set_elem(elem, np.tile(data, 2))


def test_dict_type_with_children():
    # TODO: may want to do something better for namespace control/registry
    # for now, just force it globally to make the element comparison easier
    lxml.etree.register_namespace("si", "urn:SICommon:1.0")

    # need to test with an instantiable class, so arbitrarily choose SIDD
    xh = sarkit.sidd._xmlhelp2.XmlHelper2("urn:SIDD:3.0.0")
    type_obj = xh.get_transcoder(
        "<UNNAMED>-{urn:SIDD:3.0.0}ExploitationFeaturesType/{urn:SIDD:3.0.0}Collection"
    )
    assert isinstance(type_obj, sarkit._xmlhelp2.DictType)
    assert type_obj.typedef.children
    assert type_obj.typedef.attributes

    # attributes/children are optional
    val = {"Information": {"SensorName": "the_sensor"}}
    elem = type_obj.make_elem("{urn:SIDD:3.0.0}testelem", val)
    npt.assert_equal(val, type_obj.parse_elem(elem))

    # attributes are prefixed with @
    val["@identifier"] = "myid"
    type_obj.set_elem(elem, val)
    assert elem.get("identifier") == "myid"
    npt.assert_equal(val, type_obj.parse_elem(elem))

    # nested dicts are handled
    val["Information"]["RadarMode"] = {"ModeType": "SPOTLIGHT"}
    val["Information"]["CollectionDateTime"] = datetime.datetime.now(
        tz=datetime.timezone.utc
    )
    type_obj.set_elem(elem, val)
    npt.assert_equal(val, type_obj.parse_elem(elem))

    # repeatable nodes are lists
    val["Information"]["Polarization"] = [
        {"RcvPolarizationOffset": x} for x in range(24)
    ]
    type_obj.set_elem(elem, val)
    npt.assert_equal(val, type_obj.parse_elem(elem))
    assert len(elem.findall(".//{*}RcvPolarizationOffset")) == 24

    # key order doesn't matter
    info_items = list(val["Information"].items())
    random.shuffle(info_items)
    val["Information"] = dict(info_items)
    elem_from_shuffle = type_obj.make_elem(elem.tag, val)
    npt.assert_equal(val, type_obj.parse_elem(elem_from_shuffle))
    assert lxml.etree.tostring(elem, method="c14n") == lxml.etree.tostring(
        elem_from_shuffle, method="c14n"
    )

    # error if unexpected repetition in elem during parse
    badelem = copy.deepcopy(elem)
    sn = badelem.find(".//{*}SensorName")
    sn.addnext(copy.deepcopy(sn))
    with pytest.raises(ValueError, match="SensorName not expected to repeat"):
        type_obj.parse_elem(badelem)

    # error if unrecognized keys during set
    badval = copy.deepcopy(val)
    badval["badkey"] = "shouldn't be here"
    with pytest.raises(ValueError, match="unrecognized_keys.*badkey"):
        type_obj.make_elem("cannotmake", badval)


def test_dict_type_without_children():
    # need to test with an instantiable class, so arbitrarily choose SIDD
    xh = sarkit.sidd._xmlhelp2.XmlHelper2("urn:SIDD:3.0.0")
    type_obj = xh.get_transcoder("{urn:SICommon:1.0}ParameterType")
    assert isinstance(type_obj, sarkit._xmlhelp2.DictType)
    assert not type_obj.typedef.children
    assert type_obj.typedef.attributes
    assert type_obj.typedef.text_typename

    # text is stored in #text
    val = {"#text": "parameter-text"}
    elem = type_obj.make_elem("{urn:SIDD:3.0.0}testelem", val)
    npt.assert_equal(val, type_obj.parse_elem(elem))

    # attributes are prefixed with @
    val["@name"] = "parameter-name"
    type_obj.set_elem(elem, val)
    assert elem.get("name") == "parameter-name"
    npt.assert_equal(val, type_obj.parse_elem(elem))

    # error if unexpected children
    badelem = copy.deepcopy(elem)
    badelem.append(copy.deepcopy(badelem))
    with pytest.raises(ValueError, match="not expected to have children"):
        type_obj.parse_elem(badelem)

    # error if #text is missing
    badval = copy.deepcopy(val)
    del badval["#text"]
    with pytest.raises(KeyError):
        type_obj.make_elem("cannotmake", badval)
