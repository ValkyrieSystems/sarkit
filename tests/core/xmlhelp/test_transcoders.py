import datetime

import lxml.etree
import numpy as np
import pytest

import sarkit.xmlhelp._transcoders as skxt


@pytest.mark.parametrize("force_utc", (True, False))
def test_xdt_naive(force_utc):
    xdt_t = skxt.XdtType(force_utc=force_utc)
    dt = datetime.datetime.now()
    elem = xdt_t.make_elem("Xdt", dt)
    if force_utc:
        assert elem.text.endswith("Z")
        assert xdt_t.parse_elem(elem) == dt.replace(tzinfo=datetime.timezone.utc)
    else:
        assert xdt_t.parse_elem(elem) == dt


@pytest.mark.parametrize("force_utc", (True, False))
def test_xdt_aware(force_utc):
    xdt_t = skxt.XdtType(force_utc=force_utc)
    dt = datetime.datetime.now(
        tz=datetime.timezone(offset=datetime.timedelta(hours=5.5))
    )
    elem = xdt_t.make_elem("Xdt", dt)
    if force_utc:
        assert elem.text.endswith("Z")
        assert xdt_t.parse_elem(elem) == dt
    else:
        assert xdt_t.parse_elem(elem) == dt.replace(tzinfo=None)


@pytest.mark.parametrize("ndim", (1, 2))
def test_poly(ndim):
    shape = np.arange(3, 3 + ndim)
    coefs = np.arange(np.prod(shape)).reshape(shape)
    polytype = skxt.PolyNdType(ndim)
    elem = polytype.make_elem("Poly", coefs)
    assert np.array_equal(polytype.parse_elem(elem), coefs)


def test_xyzpoly():
    coefs = np.linspace(-10, 10, 33).reshape((11, 3))
    elem = skxt.XyzPolyType().make_elem("{faux-ns}XyzPoly", coefs)
    assert np.array_equal(skxt.XyzPolyType().parse_elem(elem), coefs)


def test_xyz():
    xyz = [-10.0, 10.0, 0.20]
    elem = skxt.XyzType().make_elem("{faux-ns}XyzNode", xyz)
    assert np.array_equal(skxt.XyzType().parse_elem(elem), xyz)


def test_txt():
    elem = lxml.etree.Element("{faux-ns}Node")
    assert skxt.TxtType().parse_elem(elem) == ""
    new_str = "replacement string"
    new_elem = skxt.TxtType().make_elem("Txt", new_str)
    assert skxt.TxtType().parse_elem(new_elem) == new_str


@pytest.mark.parametrize("val", (True, False))
def test_bool(val):
    elem = skxt.BoolType().make_elem("node", val)
    assert skxt.BoolType().parse_elem(elem) == val


@pytest.mark.parametrize("val", (1.23, -4.56j, 1.23 - 4.56j))
def test_cmplx(val):
    elem = skxt.CmplxType().make_elem("node", val)
    assert skxt.CmplxType().parse_elem(elem) == val


def test_line_samp():
    ls_data = [1000, 2000]
    type_obj = skxt.LineSampType()
    elem = type_obj.make_elem("{faux-ns}LsNode", ls_data)
    assert np.array_equal(type_obj.parse_elem(elem), ls_data)


def test_array():
    data = np.random.default_rng().random((3,))
    elem = lxml.etree.Element("{faux-ns}ArrayDblNode")
    type_obj = skxt.ArrayType({c: skxt.DblType() for c in ("a", "b", "c")})
    type_obj.set_elem(elem, data)
    assert np.array_equal(type_obj.parse_elem(elem), data)
    with pytest.raises(ValueError, match="len.*does not match expected"):
        type_obj.set_elem(elem, np.tile(data, 2))


def test_xy():
    xy = [-10.0, 10.0]
    elem = skxt.XyType().make_elem("{faux-ns}XyNode", xy)
    assert np.array_equal(skxt.XyType().parse_elem(elem), xy)


def test_hex():
    hexval = b"\xba\xdd"
    elem = skxt.HexType().make_elem("{faux-ns}HexNode", hexval)
    assert np.array_equal(skxt.HexType().parse_elem(elem), hexval)


def test_parameter():
    name = "TestName"
    val = "TestVal"
    elem = skxt.ParameterType().make_elem("{faux-ns}Parameter", (name, val))
    assert skxt.ParameterType().parse_elem(elem) == (name, val)


def test_mtx_type():
    data = np.arange(np.prod(6)).reshape((2, 3))
    type_obj = skxt.MtxType(data.shape)
    elem = type_obj.make_elem("{faux-ns}MtxNode", data)
    assert np.array_equal(type_obj.parse_elem(elem), data)

    with pytest.raises(ValueError, match="shape.*does not match expected"):
        type_obj.set_elem(elem, np.tile(data, 2))
