import pathlib

import lxml.etree
import numpy as np
import pytest

import sarkit._nitf_io
import sarkit.sicd as sksicd

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def _random_image(sicd_xmltree):
    xml_helper = sksicd.XmlHelper(sicd_xmltree)
    rows = xml_helper.load("./{*}ImageData/{*}NumRows")
    cols = xml_helper.load("./{*}ImageData/{*}NumCols")
    shape = (rows, cols)

    assert sicd_xmltree.findtext("./{*}ImageData/{*}PixelType") == "RE32F_IM32F"

    components = (
        2 * np.random.default_rng().random(shape + (2,), dtype=np.float32)
    ) - 1
    return components.astype(">f4").view(">c8").squeeze()


@pytest.mark.parametrize(
    "sicd_xml,pixel_type",
    [
        (DATAPATH / "example-sicd-1.1.0.xml", "RE32F_IM32F"),
        (DATAPATH / "example-sicd-1.2.1.xml", "RE16I_IM16I"),
        (DATAPATH / "example-sicd-1.3.0.xml", "AMP8I_PHS8I"),
        (DATAPATH / "example-sicd-1.4.0.xml", "RE32F_IM32F"),
    ],
)
def test_roundtrip(tmp_path, sicd_xml, pixel_type):
    out_sicd = tmp_path / "out.sicd"
    basis_etree = lxml.etree.parse(sicd_xml)
    basis_array = _random_image(basis_etree)

    dtype = sksicd.PIXEL_TYPES[pixel_type]["dtype"]
    if pixel_type == "RE16I_IM16I":
        basis_array = (
            (np.iinfo(dtype["real"]).max * basis_array.view(basis_array.real.dtype))
            .astype(dtype["real"])
            .view(dtype)
        )
    if pixel_type == "AMP8I_PHS8I":
        basis_array = (
            (
                np.iinfo(dtype["amp"]).max
                * np.abs(basis_array.view(basis_array.real.dtype))
            )
            .astype(dtype["amp"])
            .view(dtype)
        )
    basis_etree.find("{*}ImageData/{*}PixelType").text = pixel_type
    basis_version = lxml.etree.QName(basis_etree.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=sksicd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(basis_etree)

    metadata = sksicd.NitfMetadata(
        xmltree=basis_etree,
        file_header_part={
            "ostaid": "ostaid",
            "ftitle": "ftitle",
            # Data is unclassified.  These fields are filled for testing purposes only.
            "security": {
                "clas": "T",
                "clsy": "US",
                "code": "code_h",
                "ctlh": "hh",
                "rel": "rel_h",
                "dctp": "DD",
                "dcdt": "20000101",
                "dcxm": "25X1",
                "dg": "C",
                "dgdt": "20000102",
                "cltx": "CW_h",
                "catp": "O",
                "caut": "caut_h",
                "crsn": "A",
                "srdt": "",
                "ctln": "ctln_h",
            },
            "oname": "oname",
            "ophone": "ophone",
        },
        im_subheader_part={
            "tgtid": "tgtid",
            "iid2": "iid2",
            # Data is unclassified.  These fields are filled for testing purposes only.
            "security": {
                "clas": "S",
                "clsy": "II",
                "code": "code_i",
                "ctlh": "ii",
                "rel": "rel_i",
                "dctp": "",
                "dcdt": "",
                "dcxm": "X2",
                "dg": "R",
                "dgdt": "20000202",
                "cltx": "RL_i",
                "catp": "D",
                "caut": "caut_i",
                "crsn": "B",
                "srdt": "20000203",
                "ctln": "ctln_i",
            },
            "isorce": "isorce",
            "icom": ["first comment", "second comment"],
        },
        de_subheader_part={
            # Data is unclassified.  These fields are filled for testing purposes only.
            "security": {
                "clas": "U",
                "clsy": "DD",
                "code": "code_d",
                "ctlh": "dd",
                "rel": "rel_d",
                "dctp": "X",
                "dcdt": "",
                "dcxm": "X3",
                "dg": "",
                "dgdt": "20000302",
                "cltx": "CH_d",
                "catp": "M",
                "caut": "caut_d",
                "crsn": "C",
                "srdt": "20000303",
                "ctln": "ctln_d",
            },
            "desshrp": "desshrp",
            "desshli": "desshli",
            "desshlin": "desshlin",
            "desshabs": "desshabs",
        },
    )
    with out_sicd.open("wb") as f:
        with sksicd.NitfWriter(f, metadata) as writer:
            writer.write_image(basis_array)

    with out_sicd.open("rb") as f, sksicd.NitfReader(f) as reader:
        read_array = reader.read_image()

    schema.assertValid(reader.metadata.xmltree)
    assert metadata == reader.metadata
    assert np.array_equal(basis_array, read_array)


def test_nitfheaderfields_from_header():
    header = sarkit._nitf_io.FileHeader("FHDR")
    header["OSTAID"].value = "ostaid"
    header["FTITLE"].value = "ftitle"
    # Data is unclassified.  These fields are filled for testing purposes only.
    header["FSCLAS"].value = "T"
    header["FSCLSY"].value = "US"
    header["FSCODE"].value = "code_h"
    header["FSCTLH"].value = "hh"
    header["FSREL"].value = "rel_h"
    header["FSDCTP"].value = "DD"
    header["FSDCDT"].value = "20000101"
    header["FSDCXM"].value = "25X1"
    header["FSDG"].value = "C"
    header["FSDGDT"].value = "20000102"
    header["FSCLTX"].value = "CW_h"
    header["FSCATP"].value = "O"
    header["FSCAUT"].value = "caut_h"
    header["FSCRSN"].value = "A"
    header["FSSRDT"].value = ""
    header["FSCTLN"].value = "ctln_h"
    header["ONAME"].value = "oname"
    header["OPHONE"].value = "ophone"

    fields = sksicd.NitfFileHeaderPart._from_header(header)
    assert fields.ostaid == header["OSTAID"].value
    assert fields.ftitle == header["FTITLE"].value
    assert fields.security.clas == header["FSCLAS"].value
    assert fields.security.clsy == header["FSCLSY"].value
    assert fields.security.code == header["FSCODE"].value
    assert fields.security.ctlh == header["FSCTLH"].value
    assert fields.security.rel == header["FSREL"].value
    assert fields.security.dctp == header["FSDCTP"].value
    assert fields.security.dcxm == header["FSDCXM"].value
    assert fields.security.dg == header["FSDG"].value
    assert fields.security.dgdt == header["FSDGDT"].value
    assert fields.security.cltx == header["FSCLTX"].value
    assert fields.security.catp == header["FSCATP"].value
    assert fields.security.caut == header["FSCAUT"].value
    assert fields.security.crsn == header["FSCRSN"].value
    assert fields.security.srdt == header["FSSRDT"].value
    assert fields.security.ctln == header["FSCTLN"].value
    assert fields.oname == header["ONAME"].value
    assert fields.ophone == header["OPHONE"].value


def test_nitfimagesegmentfields_from_header():
    comments = ["first", "second"]
    header = sarkit._nitf_io.ImageSubHeader("name")
    header["ISORCE"].value = "isorce"
    header["NICOM"].value = 2
    header["ICOM1"].value = comments[0]
    header["ICOM2"].value = comments[1]
    # Data is unclassified.  These fields are filled for testing purposes only.
    header["ISCLAS"].value = "T"
    header["ISCLSY"].value = "US"
    header["ISCODE"].value = "code_h"
    header["ISCTLH"].value = "hh"
    header["ISREL"].value = "rel_h"
    header["ISDCTP"].value = "DD"
    header["ISDCDT"].value = "20000101"
    header["ISDCXM"].value = "25X1"
    header["ISDG"].value = "C"
    header["ISDGDT"].value = "20000102"
    header["ISCLTX"].value = "CW_h"
    header["ISCATP"].value = "O"
    header["ISCAUT"].value = "caut_h"
    header["ISCRSN"].value = "A"
    header["ISSRDT"].value = ""
    header["ISCTLN"].value = "ctln_h"

    fields = sksicd.NitfImSubheaderPart._from_header(header)
    assert fields.isorce == header["ISORCE"].value
    assert fields.icom == comments
    assert fields.security.clas == header["ISCLAS"].value
    assert fields.security.clsy == header["ISCLSY"].value
    assert fields.security.code == header["ISCODE"].value
    assert fields.security.ctlh == header["ISCTLH"].value
    assert fields.security.rel == header["ISREL"].value
    assert fields.security.dctp == header["ISDCTP"].value
    assert fields.security.dcxm == header["ISDCXM"].value
    assert fields.security.dg == header["ISDG"].value
    assert fields.security.dgdt == header["ISDGDT"].value
    assert fields.security.cltx == header["ISCLTX"].value
    assert fields.security.catp == header["ISCATP"].value
    assert fields.security.caut == header["ISCAUT"].value
    assert fields.security.crsn == header["ISCRSN"].value
    assert fields.security.srdt == header["ISSRDT"].value
    assert fields.security.ctln == header["ISCTLN"].value


def test_nitfdesegmentfields_from_header():
    header = sarkit._nitf_io.DESubHeader("name")
    header["DESID"].value = "XML_DATA_CONTENT"
    header["DESVER"].value = 1
    header["DESSHL"].value = 773
    header["DESSHF"]["DESSHRP"].value = "desshrp"
    header["DESSHF"]["DESSHLI"].value = "desshli"
    header["DESSHF"]["DESSHLIN"].value = "desshlin"
    header["DESSHF"]["DESSHABS"].value = "desshabs"
    # Data is unclassified.  These fields are filled for testing purposes only.
    header["DESCLAS"].value = "T"
    header["DESCLSY"].value = "US"
    header["DESCODE"].value = "code_h"
    header["DESCTLH"].value = "hh"
    header["DESREL"].value = "rel_h"
    header["DESDCTP"].value = "DD"
    header["DESDCDT"].value = "20000101"
    header["DESDCXM"].value = "25X1"
    header["DESDG"].value = "C"
    header["DESDGDT"].value = "20000102"
    header["DESCLTX"].value = "CW_h"
    header["DESCATP"].value = "O"
    header["DESCAUT"].value = "caut_h"
    header["DESCRSN"].value = "A"
    header["DESSRDT"].value = ""
    header["DESCTLN"].value = "ctln_h"

    fields = sksicd.NitfDeSubheaderPart._from_header(header)
    assert fields.desshrp == header["DESSHF"]["DESSHRP"].value
    assert fields.desshli == header["DESSHF"]["DESSHLI"].value
    assert fields.desshlin == header["DESSHF"]["DESSHLIN"].value
    assert fields.desshabs == header["DESSHF"]["DESSHABS"].value
    assert fields.security.clas == header["DESCLAS"].value
    assert fields.security.clsy == header["DESCLSY"].value
    assert fields.security.code == header["DESCODE"].value
    assert fields.security.ctlh == header["DESCTLH"].value
    assert fields.security.rel == header["DESREL"].value
    assert fields.security.dctp == header["DESDCTP"].value
    assert fields.security.dcxm == header["DESDCXM"].value
    assert fields.security.dg == header["DESDG"].value
    assert fields.security.dgdt == header["DESDGDT"].value
    assert fields.security.cltx == header["DESCLTX"].value
    assert fields.security.catp == header["DESCATP"].value
    assert fields.security.caut == header["DESCAUT"].value
    assert fields.security.crsn == header["DESCRSN"].value
    assert fields.security.srdt == header["DESSRDT"].value
    assert fields.security.ctln == header["DESCTLN"].value


def test_version_info():
    actual_order = [x["version"] for x in sksicd.VERSION_INFO.values()]
    expected_order = sorted(actual_order, key=lambda x: x.split("."))
    assert actual_order == expected_order

    for urn, info in sksicd.VERSION_INFO.items():
        assert lxml.etree.parse(info["schema"]).getroot().get("targetNamespace") == urn


def test_image_sizing():
    sicd_xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.4.0.xml")
    xml_helper = sksicd.XmlHelper(sicd_xmltree)
    assert xml_helper.load("./{*}ImageData/{*}PixelType") == "RE32F_IM32F"

    # Tweak SICD size to force three image segments
    li_max = 9_999_999_998
    iloc_max = 99_999
    num_cols = li_max // 8 // iloc_max  # set num_cols so that row limit is iloc_max
    last_rows = 24
    num_rows = iloc_max * 2 + last_rows
    xml_helper.set("./{*}ImageData/{*}NumRows", num_rows)
    xml_helper.set("./{*}ImageData/{*}NumCols", num_cols)
    num_is, imhdrs = sksicd.image_segment_sizing_calculations(sicd_xmltree)

    assert num_is == 3

    def _parse_dms(dms_str):
        lat_deg = int(dms_str[0:2])
        lat_min = int(dms_str[2:4])
        lat_sec = int(dms_str[4:6])
        sign = {"S": -1, "N": 1}[dms_str[6]]
        lat = sign * (lat_deg + lat_min / 60.0 + lat_sec / 3600.0)

        lon_deg = int(dms_str[7:10])
        lon_min = int(dms_str[10:12])
        lon_sec = int(dms_str[12:14])
        sign = {"W": -1, "E": 1}[dms_str[14]]
        lon = sign * (lon_deg + lon_min / 60.0 + lon_sec / 3600.0)
        return lat, lon

    outer_corners_ll = [
        _parse_dms(imhdrs[0].igeolo[:15]),
        _parse_dms(imhdrs[0].igeolo[15:30]),
        _parse_dms(imhdrs[-1].igeolo[30:45]),
        _parse_dms(imhdrs[-1].igeolo[45:60]),
    ]
    icp_latlon = xml_helper.load("./{*}GeoData/{*}ImageCorners")
    np.testing.assert_allclose(outer_corners_ll, icp_latlon, atol=0.5 / 3600)

    for idx in range(len(imhdrs) - 1):
        assert imhdrs[idx].igeolo[45:] == imhdrs[idx + 1].igeolo[:15]
        assert imhdrs[idx].igeolo[30:45] == imhdrs[idx + 1].igeolo[15:30]

    for imhdr in imhdrs:
        imhdr.igeolo = ""

    expected_imhdrs = [
        sksicd.SizingImhdr(
            idlvl=1,
            ialvl=0,
            iloc_rows=0,
            nrows=iloc_max,
            igeolo="",
        ),
        sksicd.SizingImhdr(
            idlvl=2,
            ialvl=1,
            iloc_rows=iloc_max,
            nrows=iloc_max,
            igeolo="",
        ),
        sksicd.SizingImhdr(
            idlvl=3,
            ialvl=2,
            iloc_rows=iloc_max,
            nrows=24,
            igeolo="",
        ),
    ]

    assert expected_imhdrs == imhdrs
