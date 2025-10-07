import lxml.etree
import numpy.testing as npt
import pytest

import sarkit.sidd as sksidd
import sarkit.xmlhelp as skxml


def test_elementwrapper():
    root_ns = "urn:SIDD:3.0.0"
    siddroot = lxml.etree.Element(f"{{{root_ns}}}SIDD")
    xmlhelp = sksidd.XsdHelper(root_ns)
    wrapped_siddroot = skxml.ElementWrapper(siddroot, xsdhelper=xmlhelp)

    assert len(wrapped_siddroot) == 0
    assert not wrapped_siddroot

    # Subelement KeyErrors
    with pytest.raises(KeyError, match="foo"):
        wrapped_siddroot["foo"] = "doesn't exist"

    with pytest.raises(KeyError, match="foo"):
        wrapped_siddroot["foo"]

    with pytest.raises(KeyError, match="foo"):
        del wrapped_siddroot["foo"]

    with pytest.raises(KeyError, match="foo"):
        "foo" in wrapped_siddroot

    with pytest.raises(KeyError, match="foo"):
        wrapped_siddroot.get("foo")

    # Attribute KeyErrors
    with pytest.raises(KeyError, match="@fooattr"):
        wrapped_siddroot["@fooattr"] = "doesn't exist"

    with pytest.raises(KeyError, match="@fooattr"):
        wrapped_siddroot["@fooattr"]

    with pytest.raises(KeyError, match="@fooattr"):
        del wrapped_siddroot["@fooattr"]

    with pytest.raises(KeyError, match="@fooattr"):
        "@fooattr" in wrapped_siddroot

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

    # get() defaults to empty ElementWrappers
    assert (
        wrapped_siddroot["Measurement"]["PlaneProjection"]["ReferencePoint"].get("ECEF")
        == wrapped_siddroot["Measurement"]["PlaneProjection"]["ReferencePoint"]["ECEF"]
    )
    assert (
        wrapped_siddroot["Measurement"]["PlaneProjection"]["ReferencePoint"].get("ECEF")
        == {}
    )
    assert (
        wrapped_siddroot["Measurement"]["PlaneProjection"]["ReferencePoint"].get(
            "ECEF", None
        )
        is None
    )


def test_elementwrapper_tofromdict():
    root_ns = "urn:SIDD:3.0.0"
    siddroot = lxml.etree.parse(
        "data/syntax_only/sidd/0000-syntax-only-sidd-3.0.xml"
    ).getroot()
    xmlhelp = sksidd.XsdHelper(root_ns)
    wrapped_siddroot = skxml.ElementWrapper(siddroot, xsdhelper=xmlhelp)

    dict1 = wrapped_siddroot.to_dict()
    wrapped_root_fromdict = skxml.ElementWrapper(
        lxml.etree.Element(lxml.etree.QName(root_ns, "SIDD")), xsdhelper=xmlhelp
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
