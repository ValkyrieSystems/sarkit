import copy
import itertools
import pathlib
import re

import jbpy
import lxml.etree
import numpy as np
import pytest
import smart_open

import sarkit.sidd as sksidd
import sarkit.sidd._constants
import tests.utils

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def _random_image(sidd_xmltree):
    xml_helper = sksidd.XmlHelper(sidd_xmltree)
    rows = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Row")
    cols = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Col")
    shape = (rows, cols)

    assert xml_helper.load("./{*}Display/{*}PixelType") == "MONO8I"

    return np.random.default_rng().integers(
        0, 255, size=shape, dtype=np.uint8, endpoint=True
    )


@pytest.mark.parametrize("force_segmentation", [False, True])
@pytest.mark.parametrize("with_ded", [False, True])
@pytest.mark.parametrize(
    "sidd_xml",
    [
        DATAPATH / "example-sidd-1.0.0.xml",
        DATAPATH / "example-sidd-2.0.0.xml",
        DATAPATH / "example-sidd-3.0.0.xml",
    ],
)
def test_roundtrip(force_segmentation, with_ded, sidd_xml, tmp_path, monkeypatch):
    out_sidd = tmp_path / "out.sidd"
    sicd_xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.4.0.xml")
    basis_etree0 = lxml.etree.parse(sidd_xml)
    basis_array0 = _random_image(basis_etree0)

    basis_etree1 = lxml.etree.parse(sidd_xml)
    basis_etree1.find("./{*}Display/{*}PixelType").text = "MONO16I"
    basis_array1 = 2**16 - 1 - basis_array0.astype(np.uint16)

    def _set_3_bands(tree):
        ew = sksidd.ElementWrapper(tree.getroot())

        try:
            ew["Display"]["NumBands"] = 3
        except KeyError:
            # SIDD 1.0
            return

        ew["Display"].add(
            "NonInteractiveProcessing",
            copy.deepcopy(ew["Display"]["NonInteractiveProcessing"][0]),
        )
        ew["Display"].add(
            "NonInteractiveProcessing",
            copy.deepcopy(ew["Display"]["NonInteractiveProcessing"][0]),
        )
        ew["Display"]["NonInteractiveProcessing"][1]["@band"] = "2"
        ew["Display"]["NonInteractiveProcessing"][2]["@band"] = "3"
        ew["Display"].add(
            "InteractiveProcessing",
            copy.deepcopy(ew["Display"]["InteractiveProcessing"][0]),
        )
        ew["Display"].add(
            "InteractiveProcessing",
            copy.deepcopy(ew["Display"]["InteractiveProcessing"][0]),
        )
        ew["Display"]["InteractiveProcessing"][1]["@band"] = "2"
        ew["Display"]["InteractiveProcessing"][2]["@band"] = "3"

    basis_etree2 = lxml.etree.parse(sidd_xml)
    basis_etree2.find("./{*}Display/{*}PixelType").text = "RGB24I"
    _set_3_bands(basis_etree2)
    basis_array2 = np.empty(basis_array0.shape, sksidd.PIXEL_TYPES["RGB24I"]["dtype"])
    basis_array2["R"] = basis_array0
    basis_array2["G"] = basis_array0 + 1
    basis_array2["B"] = basis_array0 - 1

    basis_etree3 = lxml.etree.parse(sidd_xml)
    basis_array3 = _random_image(basis_etree3)
    basis_etree3.find("./{*}Display/{*}PixelType").text = "RGB8LU"
    _set_3_bands(basis_etree3)
    lookup_table3 = np.asarray(
        [
            np.arange(256, dtype=np.uint8),
            np.arange(256, dtype=np.uint8)[::-1],
            np.random.default_rng(12345).integers(0, 2**8, (256,), dtype=np.uint8),
        ]
    )
    lookup_table3 = (
        lookup_table3.T.reshape(-1, 3)
        .copy()
        .view(sksidd.PIXEL_TYPES["RGB24I"]["dtype"])
        .squeeze()
    )

    basis_etree4 = lxml.etree.parse(sidd_xml)
    basis_array4 = _random_image(basis_etree4)
    basis_etree4.find("./{*}Display/{*}PixelType").text = "MONO8LU"
    lookup_table4 = np.arange(256, dtype=np.uint8)[::-1]

    basis_etree5 = lxml.etree.parse(sidd_xml)
    basis_array5 = _random_image(basis_etree5)
    basis_etree5.find("./{*}Display/{*}PixelType").text = "MONO8LU"
    lookup_table5 = (np.arange(256, dtype=np.uint16) << 8) + np.arange(
        256, dtype=np.uint16
    )[::-1]
    if force_segmentation:
        monkeypatch.setattr(
            sarkit.sidd._constants, "LI_MAX", basis_array0.nbytes // 5
        )  # reduce the segment size limit to force segmentation

    basis_ded_array = np.random.default_rng().integers(
        -32768, 32767, size=(1000, 2000), dtype=np.int16
    )

    basis_legend_array = np.random.default_rng().integers(
        0, 255, size=(33, 44), dtype=np.uint8, endpoint=True
    )
    basis_legend_array_rgb = np.empty(
        basis_legend_array.shape, sksidd.PIXEL_TYPES["RGB24I"]["dtype"]
    )
    basis_legend_array_rgb["R"] = basis_legend_array
    basis_legend_array_rgb["G"] = basis_legend_array + 1
    basis_legend_array_rgb["B"] = basis_legend_array - 1

    write_metadata = sksidd.NitfMetadata(
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
        }
    )
    write_metadata.images.extend(
        [
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree0,
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
                legends=[
                    sksidd.NitfLegendMetadata(
                        attach_row=11,
                        attach_col=22,
                        nrows=basis_legend_array.shape[0],
                        ncols=basis_legend_array.shape[1],
                        im_subheader_part={
                            "tgtid": "legtgt",
                            "iid2": "legiid2",
                            "security": {
                                "clas": "U",
                            },
                        },
                    )
                ],
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree1,
                im_subheader_part={
                    "tgtid": "tgtid",
                    "iid2": "iid2",
                    "security": {
                        "clas": "U",
                    },
                },
                de_subheader_part={
                    "security": {
                        "clas": "U",
                    },
                },
                legends=[
                    sksidd.NitfLegendMetadata(
                        attach_row=11,
                        attach_col=22,
                        nrows=basis_legend_array.shape[0],
                        ncols=basis_legend_array.shape[1],
                        im_subheader_part={
                            "tgtid": "legtgt",
                            "iid2": "first",
                            "security": {
                                "clas": "U",
                            },
                        },
                    ),
                    sksidd.NitfLegendMetadata(
                        attach_row=basis_array1.shape[0] - 11,
                        attach_col=basis_array1.shape[1] - 22,
                        nrows=basis_legend_array.shape[0],
                        ncols=basis_legend_array.shape[1],
                        im_subheader_part={
                            "tgtid": "legtgt",
                            "iid2": "second",
                            "security": {
                                "clas": "U",
                            },
                        },
                    ),
                ],
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree2,
                im_subheader_part={
                    "tgtid": "tgtid",
                    "iid2": "iid2",
                    "security": {
                        "clas": "U",
                    },
                },
                de_subheader_part={
                    "security": {
                        "clas": "U",
                    },
                },
                legends=[
                    sksidd.NitfLegendMetadata(
                        attach_row=11,
                        attach_col=22,
                        nrows=basis_legend_array_rgb.shape[0],
                        ncols=basis_legend_array_rgb.shape[1],
                        im_subheader_part={
                            "tgtid": "legtgt",
                            "iid2": "rgb",
                            "security": {
                                "clas": "U",
                            },
                        },
                    ),
                ],
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree3,
                im_subheader_part={
                    "tgtid": "tgtid",
                    "iid2": "iid2",
                    "security": {
                        "clas": "U",
                    },
                },
                de_subheader_part={
                    "security": {
                        "clas": "U",
                    },
                },
                lookup_table=lookup_table3,
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree4,
                im_subheader_part={
                    "tgtid": "tgtid",
                    "iid2": "iid2",
                    "security": {
                        "clas": "U",
                    },
                },
                de_subheader_part={
                    "security": {
                        "clas": "U",
                    },
                },
                lookup_table=lookup_table4,
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree5,
                im_subheader_part={
                    "tgtid": "tgtid",
                    "iid2": "iid2",
                    "security": {
                        "clas": "U",
                    },
                },
                de_subheader_part={
                    "security": {
                        "clas": "U",
                    },
                },
                lookup_table=lookup_table5,
            ),
        ]
    )

    write_metadata.sicd_xmls.extend(
        [
            sksidd.NitfSicdXmlMetadata(
                sicd_xmltree, de_subheader_part={"security": {"clas": "U"}}
            )
        ]
        * 2
    )

    ps_xmltree0 = lxml.etree.ElementTree(
        lxml.etree.fromstring("<product><support/></product>")
    )
    ps_xmltree1 = lxml.etree.ElementTree(
        lxml.etree.fromstring(
            '<product xmlns="https://example.com"><support/></product>'
        )
    )
    write_metadata.product_support_xmls.extend(
        [
            sksidd.NitfProductSupportXmlMetadata(
                ps_xmltree0, {"security": {"clas": "U"}}
            ),
            sksidd.NitfProductSupportXmlMetadata(
                ps_xmltree1, {"security": {"clas": "U"}}
            ),
        ]
    )
    if with_ded:
        write_metadata.ded = sksidd.NitfDedMetadata(
            nrows=basis_ded_array.shape[0],
            ncols=basis_ded_array.shape[1],
            im_subheader_part={
                "tgtid": "dedtgt",
                "iid2": "dediid2",
                "security": {
                    "clas": "U",
                },
            },
        )

    with out_sidd.open("wb") as file:
        jbp = sksidd.jbp_from_nitf_metadata(write_metadata)
        jbp["FileHeader"]["UDHDL"].value = 10
        jbp["FileHeader"]["UDHD"].append(jbpy.tre_factory("SECTGA"))
        with sksidd.NitfWriter(file, write_metadata, jbp_override=jbp) as writer:
            writer.write_image(0, basis_array0)
            writer.write_legend(0, 0, basis_legend_array)
            writer.write_image(1, basis_array1)
            writer.write_legend(1, 0, basis_legend_array)
            writer.write_legend(1, 1, basis_legend_array)
            writer.write_image(2, basis_array2)
            writer.write_legend(2, 0, basis_legend_array_rgb)
            writer.write_image(3, basis_array3)
            writer.write_image(4, basis_array4)
            writer.write_image(5, basis_array5)

            with pytest.raises(IndexError):
                writer.write_image(99, basis_array0)
            with pytest.raises(IndexError):
                writer.write_legend(0, 99, basis_legend_array)

            if with_ded:
                writer.write_ded(basis_ded_array)
            else:
                with pytest.raises(RuntimeError, match="Metadata must describe DED"):
                    writer.write_ded(basis_ded_array)

    def _num_imseg(array):
        rows_per_seg = int(np.floor(sarkit.sidd._constants.LI_MAX / array[0].nbytes))
        return int(np.ceil(array.shape[0] / rows_per_seg))

    num_expected_product_imseg = (
        _num_imseg(basis_array0)
        + _num_imseg(basis_array1)
        + _num_imseg(basis_array2)
        + _num_imseg(basis_array3)
        + _num_imseg(basis_array4)
        + _num_imseg(basis_array5)
    )
    num_images = len(write_metadata.images)
    num_legends = sum(len(image.legends) for image in write_metadata.images)
    num_expected_imseg = num_expected_product_imseg + num_legends
    if with_ded:
        num_expected_imseg += 1

    if force_segmentation:
        assert (
            num_expected_product_imseg > num_images
        )  # make sure the monkeypatch caused segmentation
    with out_sidd.open("rb") as file:
        ntf = jbpy.Jbp()
        ntf.load(file)
        assert num_expected_imseg == len(ntf["ImageSegments"])

        mapping = sksidd.product_image_segment_mapping(ntf)
        assert len(mapping) == num_images
        assert (
            sum(len(indices) for indices in mapping.values())
            == num_expected_product_imseg
        )

    with out_sidd.open("rb") as file:
        with sksidd.NitfReader(file) as reader:
            read_metadata = reader.metadata
            read_jbp = reader.jbp
            assert reader.jbp["FileHeader"]["UDHD"][0]["CETAG"].value == "SECTGA"
            assert len(reader.jbp["ImageSegments"]) == num_expected_imseg
            assert len(read_metadata.images) == 6
            assert len(read_metadata.sicd_xmls) == 2
            assert len(read_metadata.product_support_xmls) == 2
            read_array0 = reader.read_image(0)
            read_legend0 = reader.read_legend(0, 0)
            read_array1 = reader.read_image(1)
            read_legend1_0 = reader.read_legend(1, 0)
            read_legend1_1 = reader.read_legend(1, 1)
            read_array2 = reader.read_image(2)
            read_legend2 = reader.read_legend(2, 0)
            read_array3 = reader.read_image(3)
            read_array4 = reader.read_image(4)
            read_array5 = reader.read_image(5)

            with pytest.raises(IndexError):
                reader.read_image(99)
            with pytest.raises(IndexError):
                reader.read_legend(99, 0)
            with pytest.raises(IndexError):
                reader.read_legend(0, 99)

            if with_ded:
                read_ded_array = reader.read_ded()
            else:
                with pytest.raises(RuntimeError, match="no DED to read"):
                    read_ded_array = reader.read_ded()

    def _normalized(xmltree):
        return lxml.etree.tostring(xmltree, method="c14n")

    # metadata structure should roundtrip (may have different, equivalent XML encodings)
    assert write_metadata.file_header_part == read_metadata.file_header_part
    for w_image, r_image in zip(
        write_metadata.images, read_metadata.images, strict=True
    ):
        assert _normalized(w_image.xmltree) == _normalized(r_image.xmltree)
        assert w_image.im_subheader_part == r_image.im_subheader_part
        assert w_image.de_subheader_part == r_image.de_subheader_part
        for w_legend, r_legend in zip(w_image.legends, r_image.legends, strict=True):
            assert w_legend == r_legend
        assert np.all(w_image.lookup_table == r_image.lookup_table)
    assert write_metadata.ded == read_metadata.ded
    for w_psxml, r_psxml in zip(
        write_metadata.product_support_xmls,
        read_metadata.product_support_xmls,
        strict=True,
    ):
        assert _normalized(w_psxml.xmltree) == _normalized(r_psxml.xmltree)
        assert w_psxml.de_subheader_part == r_psxml.de_subheader_part
    for w_sicdxml, r_sicdxml in zip(
        write_metadata.product_support_xmls,
        read_metadata.product_support_xmls,
        strict=True,
    ):
        assert _normalized(w_sicdxml.xmltree) == _normalized(r_sicdxml.xmltree)
        assert w_sicdxml.de_subheader_part == r_sicdxml.de_subheader_part

    assert np.array_equal(basis_array0, read_array0)
    assert np.array_equal(basis_legend_array, read_legend0)
    assert np.array_equal(basis_array1, read_array1)
    assert np.array_equal(basis_legend_array, read_legend1_0)
    assert np.array_equal(basis_legend_array, read_legend1_1)
    assert np.array_equal(basis_array2, read_array2)
    assert np.array_equal(basis_legend_array_rgb, read_legend2)
    assert np.array_equal(basis_array3, read_array3)
    assert np.array_equal(basis_array4, read_array4)
    assert np.array_equal(basis_array5, read_array5)
    if with_ded:
        assert np.array_equal(basis_ded_array, read_ded_array)
        assert write_metadata.ded == read_metadata.ded

    idlvls = [seg["subheader"]["IDLVL"].value for seg in read_jbp["ImageSegments"]]
    ialvls = [seg["subheader"]["IALVL"].value for seg in read_jbp["ImageSegments"]]
    iid1s = [seg["subheader"]["IID1"].value for seg in read_jbp["ImageSegments"]]
    assert sorted(idlvls) == sorted(range(1, len(read_jbp["ImageSegments"]) + 1))
    assert set(ialvls) <= set(idlvls) | set([0])
    if with_ded:
        assert iid1s[-1] == "DED001"
        iid1s.pop()
    assert iid1s == sorted(iid1s)


def test_segmentation():
    """From Figure 2.5-6 SIDD 1.0 Multiple Input Image - Multiple Product Images Requiring Segmentation"""
    sidd_xmltree = lxml.etree.parse(DATAPATH / "example-sidd-3.0.0.xml")
    xml_helper = sksidd.XmlHelper(sidd_xmltree)
    assert xml_helper.load("./{*}Display/{*}PixelType") == "MONO8I"

    # Tweak SIDD size to force three image segments
    li_max = 9_999_999_998
    iloc_max = 99_999
    num_cols = li_max // (2 * iloc_max)  # set num_cols so that row limit is iloc_max
    last_rows = 24
    num_rows = iloc_max * 2 + last_rows
    xml_helper.set("./{*}Measurement/{*}PixelFootprint/{*}Row", num_rows)
    xml_helper.set("./{*}Measurement/{*}PixelFootprint/{*}Col", num_cols)
    fhdr_numi, fhdr_li, imhdrs = sksidd.segmentation_algorithm(
        [sidd_xmltree, sidd_xmltree]
    )

    assert fhdr_numi == 6

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

    groups = itertools.groupby(imhdrs, key=lambda x: x.iid1[:7])
    for _, group_headers in groups:
        headers = list(group_headers)
        for idx in range(len(headers) - 1):
            assert headers[idx].igeolo[45:] == headers[idx + 1].igeolo[:15]
            assert headers[idx].igeolo[30:45] == headers[idx + 1].igeolo[15:30]

    for imhdr in imhdrs:
        imhdr.igeolo = ""

    # SIDD segmentation algorithm (2.4.2.1 in 1.0/2.0/3.0) would lead to overlaps of the last partial
    # image segment due to ILOC. This implements a scheme similar to SICD wherein "RRRRR" of ILOC matches
    # the NROWs in the previous segment.
    expected_imhdrs = [
        sksidd.SegmentationImhdr(
            iid1="SIDD001001",
            idlvl=1,
            ialvl=0,
            iloc="0" * 10,
            nrows=iloc_max,
            ncols=num_cols,
            igeolo="",
            icat="SAR",
        ),
        sksidd.SegmentationImhdr(
            iid1="SIDD001002",
            idlvl=2,
            ialvl=1,
            iloc=f"{iloc_max:05d}{0:05d}",
            nrows=iloc_max,
            ncols=num_cols,
            igeolo="",
            icat="SAR",
        ),
        sksidd.SegmentationImhdr(
            iid1="SIDD001003",
            idlvl=3,
            ialvl=2,
            iloc=f"{iloc_max:05d}{0:05d}",
            nrows=last_rows,
            ncols=num_cols,
            igeolo="",
            icat="SAR",
        ),
        sksidd.SegmentationImhdr(
            iid1="SIDD002001",
            idlvl=4,
            ialvl=0,
            iloc="0" * 10,
            nrows=iloc_max,
            ncols=num_cols,
            igeolo="",
            icat="SAR",
        ),
        sksidd.SegmentationImhdr(
            iid1="SIDD002002",
            idlvl=5,
            ialvl=4,
            iloc=f"{iloc_max:05d}{0:05d}",
            nrows=iloc_max,
            ncols=num_cols,
            igeolo="",
            icat="SAR",
        ),
        sksidd.SegmentationImhdr(
            iid1="SIDD002003",
            idlvl=6,
            ialvl=5,
            iloc=f"{iloc_max:05d}{0:05d}",
            nrows=last_rows,
            ncols=num_cols,
            igeolo="",
            icat="SAR",
        ),
    ]
    expected_fhdr_li = [imhdr.nrows * imhdr.ncols for imhdr in expected_imhdrs]

    assert expected_fhdr_li == fhdr_li

    assert expected_imhdrs == imhdrs


def test_NitfProductImageMetadata():  # noqa N802
    xmltree = lxml.etree.parse(DATAPATH / "example-sidd-3.0.0.xml")
    im_subheader_part = {
        "tgtid": "tgtid",
        "iid2": "iid2",
        "security": {
            "clas": "U",
        },
    }
    de_subheader_part = {
        "security": {
            "clas": "U",
        },
    }

    assert xmltree.find("./{*}Display/{*}PixelType").text == "MONO8I"
    image_info = sksidd.NitfProductImageMetadata(
        xmltree=xmltree,
        im_subheader_part=im_subheader_part,
        de_subheader_part=de_subheader_part,
        lookup_table=None,
    )
    assert image_info.im_subheader_part.tgtid == im_subheader_part["tgtid"]
    assert image_info.im_subheader_part.iid2 == im_subheader_part["iid2"]
    assert (
        image_info.im_subheader_part.security.clas
        == im_subheader_part["security"]["clas"]
    )
    assert (
        image_info.de_subheader_part.security.clas
        == de_subheader_part["security"]["clas"]
    )

    # Can't have lookup table for MONO8I
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "lookup_table type mismatch.  pixel_type='MONO8I'  lut_dtype=dtype('uint8')"
        ),
    ):
        image_info = sksidd.NitfProductImageMetadata(
            xmltree=xmltree,
            im_subheader_part=im_subheader_part,
            de_subheader_part=de_subheader_part,
            lookup_table=np.arange(256, dtype=np.uint8),
        )

    xmltree.find("./{*}Display/{*}PixelType").text = "MONO8LU"
    image_info = sksidd.NitfProductImageMetadata(
        xmltree=xmltree,
        im_subheader_part=im_subheader_part,
        de_subheader_part=de_subheader_part,
        lookup_table=np.arange(256, dtype=np.uint8),
    )
    assert image_info.lookup_table.shape == (256,)
    assert image_info.lookup_table.dtype == np.uint8

    image_info = sksidd.NitfProductImageMetadata(
        xmltree=xmltree,
        im_subheader_part=im_subheader_part,
        de_subheader_part=de_subheader_part,
        lookup_table=np.arange(256, dtype=np.uint16),
    )
    assert image_info.lookup_table.shape == (256,)
    assert image_info.lookup_table.dtype == np.uint16

    # Must have lookup table for MONO8LU
    with pytest.raises(
        RuntimeError,
        match="lookup_table type mismatch.  pixel_type='MONO8LU'  lut_dtype=None",
    ):
        image_info = sksidd.NitfProductImageMetadata(
            xmltree=xmltree,
            im_subheader_part=im_subheader_part,
            de_subheader_part=de_subheader_part,
            lookup_table=None,
        )

    # MONO8LU lookup table must have 256 elements
    with pytest.raises(
        ValueError, match="lookup_table must contain exactly 256 elements"
    ):
        image_info = sksidd.NitfProductImageMetadata(
            xmltree=xmltree,
            im_subheader_part=im_subheader_part,
            de_subheader_part=de_subheader_part,
            lookup_table=np.arange(255, dtype=np.uint8),
        )

    # MONO8LU lookup table must be uint8 or uint16
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "lookup_table type mismatch.  pixel_type='MONO8LU'  lut_dtype=dtype('uint32')"
        ),
    ):
        image_info = sksidd.NitfProductImageMetadata(
            xmltree=xmltree,
            im_subheader_part=im_subheader_part,
            de_subheader_part=de_subheader_part,
            lookup_table=np.arange(256, dtype=np.uint32),
        )

    xmltree.find("./{*}Display/{*}PixelType").text = "RGB8LU"
    rgb_dtype = sksidd.PIXEL_TYPES["RGB24I"]["dtype"]
    good_rgb_lut = np.empty(256, dtype=rgb_dtype)
    image_info = sksidd.NitfProductImageMetadata(
        xmltree=xmltree,
        im_subheader_part=im_subheader_part,
        de_subheader_part=de_subheader_part,
        lookup_table=good_rgb_lut,
    )
    assert image_info.lookup_table.shape == (256,)
    assert image_info.lookup_table.dtype == rgb_dtype

    # Must have lookup table for RGB8LU
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "lookup_table type mismatch.  pixel_type='RGB8LU'  lut_dtype=None"
        ),
    ):
        image_info = sksidd.NitfProductImageMetadata(
            xmltree=xmltree,
            im_subheader_part=im_subheader_part,
            de_subheader_part=de_subheader_part,
            lookup_table=None,
        )

    # RGB8LU lookup table must have 256 elements
    with pytest.raises(
        ValueError, match="lookup_table must contain exactly 256 elements"
    ):
        image_info = sksidd.NitfProductImageMetadata(
            xmltree=xmltree,
            im_subheader_part=im_subheader_part,
            de_subheader_part=de_subheader_part,
            lookup_table=good_rgb_lut[:255],
        )

    # RGB8LU lookup table must be RGB structured dtype
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "lookup_table type mismatch.  pixel_type='RGB8LU'  lut_dtype=dtype('uint8')"
        ),
    ):
        image_info = sksidd.NitfProductImageMetadata(
            xmltree=xmltree,
            im_subheader_part=im_subheader_part,
            de_subheader_part=de_subheader_part,
            lookup_table=np.arange(256, dtype=np.uint8),
        )


def test_version_info():
    actual_order = [x["version"] for x in sksidd.VERSION_INFO.values()]
    expected_order = sorted(actual_order, key=lambda x: x.split("."))
    assert actual_order == expected_order

    for urn, info in sksidd.VERSION_INFO.items():
        assert lxml.etree.parse(info["schema"]).getroot().get("targetNamespace") == urn


def test_remote_read(example_sidd):
    with tests.utils.static_http_server(example_sidd.parent) as server_url:
        with smart_open.open(
            f"{server_url}/{example_sidd.name}", mode="rb"
        ) as file_object:
            with sksidd.NitfReader(file_object) as r:
                _ = r.read_image(0)
