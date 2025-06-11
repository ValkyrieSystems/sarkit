import datetime

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


def test_elementwrapper():
    root_ns = "urn:SIDD:3.0.0"
    siddroot = lxml.etree.Element(f"{{{root_ns}}}SIDD")
    xmlhelp = sarkit.sidd._xmlhelp2.XmlHelper2(root_ns)
    wrapped_siddroot = sarkit._xmlhelp2.ElementWrapper(siddroot, xmlhelper=xmlhelp)

    assert len(wrapped_siddroot) == 0
    assert not wrapped_siddroot

    # Subelement KeyErrors
    with pytest.raises(KeyError, match="foo"):
        wrapped_siddroot["foo"] = "doesn't exist"

    with pytest.raises(KeyError, match="foo"):
        wrapped_siddroot["foo"]

    with pytest.raises(KeyError, match="foo"):
        del wrapped_siddroot["foo"]

    # Attribute KeyErrors
    with pytest.raises(KeyError, match="@fooattr"):
        wrapped_siddroot["@fooattr"] = "doesn't exist"

    with pytest.raises(KeyError, match="@fooattr"):
        wrapped_siddroot["@fooattr"]

    with pytest.raises(KeyError, match="@fooattr"):
        del wrapped_siddroot["@fooattr"]

    # Add descendant of repeatable
    wrapped_siddroot["ProductProcessing"].add("ProcessingModule")["ModuleName"] = (
        "mn-name",
        "mn-val",
    )
    mn_elem = siddroot.find("{*}ProductProcessing/{*}ProcessingModule[1]/{*}ModuleName")
    assert mn_elem.get("name") == "mn-name"
    assert mn_elem.text == "mn-val"

    wrapped_siddroot["ProductProcessing"]["ProcessingModule"][0].add(
        "ModuleParameter", ("mp-name", "mp-val")
    )
    mp_elem = siddroot.find(
        "{*}ProductProcessing/{*}ProcessingModule[1]/{*}ModuleParameter"
    )
    assert mp_elem.get("name") == "mp-name"
    assert mp_elem.text == "mp-val"
    assert (
        "ModuleParameter"
        in wrapped_siddroot["ProductProcessing"]["ProcessingModule"][0]
    )
    assert wrapped_siddroot["ProductProcessing"]["ProcessingModule"][0][
        "ModuleParameter"
    ][0] == ("mp-name", "mp-val")

    # Set descendant
    wrapped_siddroot["ProductCreation"]["ProductName"] = "prodname"
    assert siddroot.findtext("{*}ProductCreation/{*}ProductName") == "prodname"

    with pytest.raises(ValueError, match="ProductName already exists"):
        wrapped_siddroot["ProductCreation"].add("ProductName")

    del wrapped_siddroot["ProductCreation"]["ProductName"]
    assert siddroot.find("{*}ProductCreation/{*}ProductName") is None

    wrapped_siddroot["ProductCreation"].add("ProductName", "prodname is back")
    assert siddroot.findtext("{*}ProductCreation/{*}ProductName") == "prodname is back"

    # Set attribute of new repeatable
    wrapped_siddroot["ExploitationFeatures"].add("Collection")["@identifier"] = (
        "first-id"
    )
    wrapped_siddroot["ExploitationFeatures"].add("Collection")["@identifier"] = (
        "second-id"
    )
    assert (
        siddroot.find("{*}ExploitationFeatures/{*}Collection[1]").get("identifier")
        == "first-id"
    )
    assert (
        siddroot.find("{*}ExploitationFeatures/{*}Collection[2]").get("identifier")
        == "second-id"
    )

    wrapped_siddroot["ProductCreation"]["Classification"]["@classification"] = "U"
    attribname, attribval = dict(
        siddroot.find("{*}ProductCreation/{*}Classification").attrib
    ).popitem()
    assert attribname.endswith("classification")
    assert attribval == "U"

    del wrapped_siddroot["ProductCreation"]["Classification"]["@classification"]
    assert not siddroot.find("{*}ProductCreation/{*}Classification").attrib

    # Contains for schema-valid element in missing branch
    assert (
        "ECEF"
        not in wrapped_siddroot["Measurement"]["PlaneProjection"]["ReferencePoint"]
    )


def test_elementwrapper_tofromdict():
    root_ns = "urn:SIDD:3.0.0"
    siddroot = lxml.etree.parse(
        "data/syntax_only/sidd/0000-syntax-only-sidd-3.0.xml"
    ).getroot()
    xmlhelp = sarkit.sidd._xmlhelp2.XmlHelper2(root_ns)
    wrapped_siddroot = sarkit._xmlhelp2.ElementWrapper(siddroot, xmlhelper=xmlhelp)

    dict1 = wrapped_siddroot.to_dict()
    wrapped_root_fromdict = sarkit._xmlhelp2.ElementWrapper(
        lxml.etree.Element(lxml.etree.QName(root_ns, "SIDD")), xmlhelper=xmlhelp
    )
    wrapped_root_fromdict.from_dict(dict1)
    dict2 = wrapped_root_fromdict.to_dict()

    npt.assert_equal(dict1, dict2)

    orig_elempaths = {siddroot.getroottree().getelementpath(x) for x in siddroot.iter()}
    fromdict_elempaths = {
        wrapped_root_fromdict.elem.getroottree().getelementpath(x)
        for x in wrapped_root_fromdict.elem.iter()
    }
    # transcoders can add zeros to sparse polynomials/matrices, etc.
    assert orig_elempaths.issubset(fromdict_elempaths)
