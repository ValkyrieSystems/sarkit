import lxml.etree

import sarkit.cphd as skcphd


def _replace_refgeom(cphd_xmltree, pvps):
    cphd_xmltree.find("{*}ReferenceGeometry").clear()
    refgeom = skcphd.compute_reference_geometry(cphd_xmltree, pvps)
    cphd_xmltree.getroot().replace(cphd_xmltree.find("{*}ReferenceGeometry"), refgeom)
    basis_version = lxml.etree.QName(cphd_xmltree.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=skcphd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(cphd_xmltree)
    return refgeom


def test_compute_reference_geometry(example_cphd):
    with open(example_cphd, "rb") as f, skcphd.Reader(f) as r:
        cphdxml = r.metadata.xmltree
        pvps = r.read_pvps(cphdxml.findtext(".//{*}RefChId"))
    _replace_refgeom(cphdxml, pvps)


def test_compute_reference_geometry_bistatic(example_cphd):
    with open(example_cphd, "rb") as f, skcphd.Reader(f) as r:
        cphdxml = r.metadata.xmltree
        pvps = r.read_pvps(cphdxml.findtext(".//{*}RefChId"))

    assert cphdxml.findtext(".//{*}CollectType") == "MONOSTATIC"
    cphdxml.find(".//{*}CollectType").text = "BISTATIC"
    new_refgeom = _replace_refgeom(cphdxml, pvps)
    assert new_refgeom.find("{*}Bistatic") is not None
