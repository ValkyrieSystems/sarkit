import copy
import datetime
import pathlib
import unittest.mock

import jbpy
import lxml.builder
import pytest
from lxml import etree

import sarkit.verification._siddcheck
import tests.utils
from sarkit.verification._sidd_consistency import SiddConsistency
from sarkit.verification._siddcheck import main

from . import testing

DATAPATH = pathlib.Path(__file__).parents[2] / "data"

GOOD_SIDD_XML_PATH = DATAPATH / "example-sidd-3.0.0.xml"


@pytest.fixture(scope="session")
def example_sidd_file(example_sidd):
    assert not main([str(example_sidd)])
    with example_sidd.open("rb") as f:
        yield f


@pytest.fixture
def sidd_con(example_sidd_file):
    return SiddConsistency.from_file(example_sidd_file)


def _sidd_elementmaker(elem):
    return lxml.builder.ElementMaker(
        namespace=etree.QName(elem).namespace, nsmap=elem.nsmap
    )


@pytest.fixture
def sidd_con_with_dra(sidd_con):
    sidd = _sidd_elementmaker(sidd_con.xml_trees[0].getroot())
    draparameters_elem = sidd.DRAParameters(
        sidd.Pmin("0.123"),
        sidd.Pmax("0.456"),
        sidd.EminModifier("0.789"),
        sidd.EmaxModifier("0.987"),
    )
    draoverrides_elem = sidd.DRAOverrides(
        sidd.Subtractor("1234.56789"),
        sidd.Multiplier("24.8"),
    )

    dra_elem = sidd_con.xml_trees[0].find(
        "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment"
    )
    dra_elem.find("{*}AlgorithmType").text = "AUTO"
    dra_elem.find("{*}BandStatsSource").addnext(draparameters_elem)
    draparameters_elem.addnext(draoverrides_elem)
    return sidd_con


@pytest.fixture
def sidd_con_with_plane_projection(example_sidd_file):
    con = SiddConsistency.from_file(example_sidd_file)
    assert con.xml_trees[0].find(".//{*}PlaneProjection") is not None
    yield con


@pytest.mark.parametrize(
    "sidd_xml_path",
    (
        DATAPATH / "example-sidd-1.0.0.xml",
        DATAPATH / "example-sidd-2.0.0.xml",
        DATAPATH / "example-sidd-3.0.0.xml",
    ),
)
def test_main_xml(sidd_xml_path):
    assert not main([str(sidd_xml_path)])


def test_main_v1(example_sidd_v1):
    assert not main([str(example_sidd_v1)])


def test_check_against_schema():
    bad_xml = etree.parse(GOOD_SIDD_XML_PATH)
    remove_nodes(bad_xml.find("{*}ProductCreation"))
    sidd_con = SiddConsistency.from_parts(xml_trees=[bad_xml])
    sidd_con.check("check_against_schema", allow_prefix=True)
    assert sidd_con.failures()


def test_check_datetime_fields_are_utc(sidd_con):
    bad_datetime = datetime.datetime.now().isoformat()
    sidd_con.xml_trees[0].find(
        "./{*}ProductCreation/{*}ProcessorInformation/{*}ProcessingDateTime"
    ).text = bad_datetime
    sidd_con.check("check_datetime_fields_are_utc", allow_prefix=True)
    assert sidd_con.failures()


def test_check_display_numbands(sidd_con):
    sidd_con.xml_trees[0].find("./{*}Display/{*}NumBands").text += "2"
    sidd_con.check("check_display_numbands", allow_prefix=True)
    assert sidd_con.failures()


def _invalidate_display_proc(xml):
    remove_nodes(xml.find("./{*}Display/{*}InteractiveProcessing"))


def _invalidate_display_proc_attrib(xml):
    xml.find("./{*}Display/{*}NonInteractiveProcessing").set("band", "24")


@pytest.mark.parametrize(
    "invalidate_func", [_invalidate_display_proc, _invalidate_display_proc_attrib]
)
def test_check_display_processing_bands(sidd_con, invalidate_func):
    invalidate_func(sidd_con.xml_trees[0])
    sidd_con.check("check_display_processing_bands", allow_prefix=True)
    assert sidd_con.failures()


def test_check_display_antialias_filter_operation(sidd_con):
    sidd = _sidd_elementmaker(sidd_con.xml_trees[0].getroot())

    rrds = sidd_con.xml_trees[0].find(
        "./{*}Display/{*}NonInteractiveProcessing/{*}RRDS"
    )
    aa = sidd.AntiAlias(
        sidd.FilterName("testing"),
        sidd.FilterKernel(sidd.Predefined("predefined")),
        sidd.Operation("CONVOLUTION"),
    )
    rrds.append(aa)
    sidd_con.check("check_display_antialias_filter_operation", allow_prefix=True)
    assert not sidd_con.failures()

    rrds.find("./{*}AntiAlias/{*}Operation").text = "CORRELATION"
    sidd_con.check("check_display_antialias_filter_operation", allow_prefix=True)
    assert sidd_con.failures()


def test_check_display_interpolation_filter_operation(sidd_con):
    elem = sidd_con.xml_trees[0].find(
        "./{*}Display/{*}InteractiveProcessing/{*}GeometricTransform"
        "/{*}Scaling/{*}Interpolation/{*}Operation"
    )
    elem.text = "CONVOLUTION"
    sidd_con.check("check_display_interpolation_filter_operation", allow_prefix=True)
    assert sidd_con.failures()


def test_check_display_dra_bandstatssource(sidd_con):
    tree = sidd_con.xml_trees[0]
    tree.find(
        "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment/{*}BandStatsSource"
    ).text += "24"
    sidd_con.check("check_display_dra_bandstatssource", allow_prefix=True)
    assert sidd_con.failures()


def test_sidd_with_draparameters_con(sidd_con_with_dra):
    sidd_con_with_dra.check()
    assert not sidd_con_with_dra.failures()


def test_check_display_auto_dra_parameters(sidd_con_with_dra):
    tree = sidd_con_with_dra.xml_trees[0]
    remove_nodes(
        *tree.findall(
            "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment/{*}DRAParameters"
        )
    )
    sidd_con_with_dra.check("check_display_auto_dra_parameters", allow_prefix=True)
    assert sidd_con_with_dra.failures()


def test_check_display_valid_dra_parameters(sidd_con_with_dra):
    elem_path = "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment/{*}DRAParameters/{*}Pmin"
    sidd_con_with_dra.xml_trees[0].find(elem_path).text = "1.234"
    sidd_con_with_dra.check("check_display_valid_dra_parameters", allow_prefix=True)
    assert sidd_con_with_dra.failures()


def test_check_display_valid_dra_parameter_order(sidd_con_with_dra):
    tree = sidd_con_with_dra.xml_trees[0]
    drap = tree.find(
        "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment/{*}DRAParameters"
    )
    drap.find("{*}EminModifier").text = "0.5"
    drap.find("{*}EmaxModifier").text = "0.0"
    sidd_con_with_dra.check("check_display_valid_dra_parameters", allow_prefix=True)
    assert sidd_con_with_dra.failures()


def test_check_display_none_dra_overrides(sidd_con_with_dra):
    tree = sidd_con_with_dra.xml_trees[0]
    alg_type = tree.find(
        "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment/{*}AlgorithmType"
    )
    alg_type.text = "NONE"
    sidd_con_with_dra.check("check_display_none_dra_overrides", allow_prefix=True)
    assert sidd_con_with_dra.failures()


def test_check_display_valid_dra_overrides(sidd_con_with_dra):
    elem_path = "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment/{*}DRAOverrides/{*}Multiplier"
    sidd_con_with_dra.xml_trees[0].find(elem_path).text = "3000"
    sidd_con_with_dra.check("check_display_valid_dra_overrides", allow_prefix=True)
    assert sidd_con_with_dra.failures()


def _invalidate_row_uvec(xml):
    xml.find(
        "./{*}Measurement/{*}PlaneProjection/{*}ProductPlane/{*}RowUnitVector/{*}X"
    ).text = "1.1"


def _invalidate_col_uvec(xml):
    xml.find(
        "./{*}Measurement/{*}PlaneProjection/{*}ProductPlane/{*}ColUnitVector/{*}Y"
    ).text = "1.1"


def _invalidate_uvecs(xml):
    product_plane = xml.find("./{*}Measurement/{*}PlaneProjection/{*}ProductPlane")
    for component in ("X", "Y", "Z"):
        row_component = product_plane.findtext(f"./{{*}}RowUnitVector/{{*}}{component}")
        product_plane.find(
            f"./{{*}}ColUnitVector/{{*}}{component}"
        ).text = row_component


@pytest.mark.parametrize(
    "invalidate_func", [_invalidate_row_uvec, _invalidate_col_uvec, _invalidate_uvecs]
)
def test_check_measurement_productplane_unit_vectors(
    sidd_con_with_plane_projection, invalidate_func
):
    invalidate_func(sidd_con_with_plane_projection.xml_trees[0])
    sidd_con_with_plane_projection.check(
        "check_measurement_productplane_unit_vectors", allow_prefix=True
    )
    assert sidd_con_with_plane_projection.failures()


def _invalidate_validdata_index(xml):
    remove_nodes(xml.find("./{*}Measurement/{*}ValidData/{*}Vertex"))


def _invalidate_validdata_size(xml):
    xml.find("./{*}Measurement/{*}ValidData").attrib["size"] += "1"


def _invalidate_validdata_simplicity(xml):
    validdata_elem = xml.find("./{*}Measurement/{*}ValidData")
    orig_size = int(validdata_elem.attrib["size"])
    new_elem = copy.deepcopy(validdata_elem.find("./{*}Vertex[last()-1]"))
    new_elem.attrib["index"] = str(orig_size + 1)
    validdata_elem.attrib["size"] = str(orig_size + 1)
    validdata_elem.append(new_elem)


def _invalidate_validdata_winding(xml):
    validdata_elem = xml.find("./{*}Measurement/{*}ValidData")
    size = int(validdata_elem.attrib["size"])
    for index, vertex in enumerate(validdata_elem):
        if index > 0:
            orig_index = int(vertex.attrib["index"])
            vertex.attrib["index"] = str(size - orig_index + 2)


def _invalidate_validdata_first_vertex(xml):
    validdata_elem = xml.find("./{*}Measurement/{*}ValidData")
    size = int(validdata_elem.attrib["size"])
    for vertex in validdata_elem:
        orig_index = int(vertex.attrib["index"])
        vertex.attrib["index"] = str(orig_index % size + 1)


def _invalidate_validdata_bounds_negative(xml):
    xml.find("./{*}Measurement/{*}ValidData/{*}Vertex/{*}Row").text = "-10"


def _invalidate_validdata_bounds_footprint(xml):
    xml.find("./{*}Measurement/{*}ValidData/{*}Vertex/{*}Col").text = str(
        int(xml.find("./{*}Measurement/{*}PixelFootprint/{*}Col").text) + 10
    )


@pytest.mark.parametrize(
    "invalidate_func",
    [
        _invalidate_validdata_index,
        _invalidate_validdata_size,
        _invalidate_validdata_simplicity,
        _invalidate_validdata_winding,
        _invalidate_validdata_first_vertex,
        _invalidate_validdata_bounds_negative,
        _invalidate_validdata_bounds_footprint,
    ],
)
def test_check_measurement_validdata(sidd_con, invalidate_func):
    invalidate_func(sidd_con.xml_trees[0])
    sidd_con.check("check_measurement_validdata", allow_prefix=True)
    assert sidd_con.failures()


def _invalidate_expfeature_measurement(xml):
    ref_pt_x_elem = xml.find("./{*}Measurement//{*}ReferencePoint/{*}ECEF/{*}X")
    ref_pt_x_elem.text = str(float(ref_pt_x_elem.text) * 1e4)


def _invalidate_expfeature_azim(xml):
    for azim_elem in xml.findall(
        "./{*}ExploitationFeatures/{*}Collection/{*}Geometry/{*}Azimuth"
    ):
        azim_elem.text = str(float(azim_elem.text) + 5.0)


def _invalidate_expfeature_north(xml):
    for north_elem in xml.findall("./{*}ExploitationFeatures/{*}Product/{*}North"):
        north_elem.text = str(float(north_elem.text) - 5.0)


@pytest.mark.parametrize(
    "invalidate_func",
    [
        _invalidate_expfeature_measurement,
        _invalidate_expfeature_azim,
        _invalidate_expfeature_north,
    ],
)
def test_check_expfeatures_geometry(example_sidd_file, invalidate_func):
    sidd_con = SiddConsistency.from_file(example_sidd_file)
    invalidate_func(sidd_con.xml_trees[0])
    sidd_con.check("check_expfeatures_geometry", allow_prefix=True)
    assert sidd_con.failures()


def test_check_geodata_image_corners(example_sidd_file):
    sidd_con = SiddConsistency.from_file(example_sidd_file)
    icp_a = sidd_con.xml_trees[0].find("./{*}GeoData/{*}ImageCorners/{*}ICP[1]")
    icp_b = sidd_con.xml_trees[0].find("./{*}GeoData/{*}ImageCorners/{*}ICP[2]")
    old_a_index = icp_a.attrib["index"]
    icp_a.attrib["index"] = icp_b.attrib["index"]
    icp_b.attrib["index"] = old_a_index
    sidd_con.check("check_geodata_image_corners", allow_prefix=True)
    assert sidd_con.failures()


def remove_nodes(*nodes):
    for node in nodes:
        node.getparent().remove(node)


def test_smart_open_http(example_sidd):
    with tests.utils.static_http_server(example_sidd.parent) as server_url:
        assert not main([f"{server_url}/{example_sidd.name}"])


def test_smart_open_contract(example_sidd, monkeypatch):
    mock_open = unittest.mock.MagicMock(side_effect=tests.utils.simple_open_read)
    monkeypatch.setattr(sarkit.verification._siddcheck, "open", mock_open)
    assert not main([str(example_sidd)])
    mock_open.assert_called_once_with(str(example_sidd), "rb")


def test_check_nitf_des_headers_outmoded_desid(sidd_con):
    sidd_con.ntf["FileHeader"]["NUMDES"].value += 1
    new_des = sidd_con.ntf["DataExtensionSegments"][-1]
    new_des["subheader"]["DESID"].value = "SICD_XML"
    sidd_con.check("check_nitf_des_headers")
    testing.assert_failures(sidd_con, "Outmoded DESID=SICD_XML not present")


def test_check_nitf_image_segment_order(sidd_con):
    def _try_order(valid, order):
        sidd_con.ntf["FileHeader"]["NUMI"].value = len(order)
        for (iid1, icat), imseg in zip(order, sidd_con.ntf["ImageSegments"]):
            imseg["subheader"]["IID1"].value = iid1
            imseg["subheader"]["ICAT"].value = icat

        sidd_con.check("check_nitf_image_segment_order")
        if valid:
            assert not sidd_con.failures()
        else:
            assert sidd_con.failures()

    # First must be SIDD001001 SAR
    _try_order(False, [("SIDD002001", "SAR")])
    _try_order(False, [("SIDD001001", "LEG")])
    _try_order(False, [("DED001", "DED")])
    _try_order(False, [("OTHER", "")])

    # First must be SIDD001001 SAR and any valid segment can follow
    _try_order(True, [("SIDD001001", "SAR")])
    _try_order(True, [("SIDD001001", "SAR"), ("SIDD001002", "SAR")])
    _try_order(True, [("SIDD001001", "SAR"), ("SIDD001002", "LEG")])
    _try_order(True, [("SIDD001001", "SAR"), ("SIDD002001", "SAR")])
    _try_order(True, [("SIDD001001", "SAR"), ("DED001", "DED")])
    _try_order(True, [("SIDD001001", "SAR"), ("OTHER", "")])

    # IID1s out of order
    _try_order(False, [("SIDD001001", "SAR"), ("SIDD001003", "SAR")])
    _try_order(False, [("SIDD001001", "SAR"), ("SIDD001003", "LEG")])
    _try_order(False, [("SIDD001001", "SAR"), ("SIDD003001", "SAR")])

    # any valid segment can follow a SAR
    _try_order(
        True, [("SIDD001001", "SAR"), ("SIDD001002", "SAR"), ("SIDD001003", "SAR")]
    )
    _try_order(
        True, [("SIDD001001", "SAR"), ("SIDD001002", "SAR"), ("SIDD001003", "LEG")]
    )
    _try_order(
        True, [("SIDD001001", "SAR"), ("SIDD001002", "SAR"), ("SIDD002001", "SAR")]
    )
    _try_order(True, [("SIDD001001", "SAR"), ("SIDD001002", "SAR"), ("DED001", "DED")])
    _try_order(True, [("SIDD001001", "SAR"), ("SIDD001002", "SAR"), ("OTHER", "")])

    # LEGs must be at the end of a product image
    _try_order(
        True, [("SIDD001001", "SAR"), ("SIDD001002", "LEG"), ("SIDD001003", "LEG")]
    )
    _try_order(
        True, [("SIDD001001", "SAR"), ("SIDD001002", "LEG"), ("SIDD002001", "SAR")]
    )
    _try_order(True, [("SIDD001001", "SAR"), ("SIDD001002", "LEG"), ("DED001", "DED")])
    _try_order(True, [("SIDD001001", "SAR"), ("SIDD001002", "LEG"), ("OTHER", "")])
    _try_order(
        False, [("SIDD001001", "SAR"), ("SIDD001002", "LEG"), ("SIDD001002", "SAR")]
    )

    # DED must be the last SIDD segment
    _try_order(True, [("SIDD001001", "SAR"), ("DED001", "DED"), ("OTHER", "")])
    _try_order(False, [("SIDD001001", "SAR"), ("DED001", "DED"), ("SIDD001002", "LEG")])
    _try_order(False, [("SIDD001001", "SAR"), ("DED001", "DED"), ("SIDD001002", "SAR")])
    _try_order(False, [("SIDD001001", "SAR"), ("DED001", "DED"), ("SIDD002001", "SAR")])

    # no SIDD segments after an "OTHER"
    _try_order(True, [("SIDD001001", "SAR"), ("OTHER", ""), ("OTHER", "")])
    _try_order(False, [("SIDD001001", "SAR"), ("OTHER", ""), ("SIDD001002", "SAR")])
    _try_order(False, [("SIDD001001", "SAR"), ("OTHER", ""), ("SIDD001002", "LEG")])
    _try_order(False, [("SIDD001001", "SAR"), ("OTHER", ""), ("SIDD002001", "SAR")])
    _try_order(False, [("SIDD001001", "SAR"), ("OTHER", ""), ("DED001", "DED")])


def test_check_nitf_display_levels(sidd_con):
    def _try_levels(valid, levels):
        sidd_con.ntf["FileHeader"]["NUMI"].value = len(levels)
        for (idlvl, ialvl), imseg in zip(levels, sidd_con.ntf["ImageSegments"]):
            imseg["subheader"]["IDLVL"].value = idlvl
            imseg["subheader"]["IALVL"].value = ialvl

        sidd_con.check("check_nitf_display_levels")
        if valid:
            assert not sidd_con.failures()
        else:
            assert sidd_con.failures()

    _try_levels(True, [(1, 0)])
    _try_levels(True, [(1, 0), (2, 1)])
    _try_levels(True, [(1, 0), (2, 0)])
    _try_levels(True, [(2, 0), (1, 0)])  # segment order does not matter

    _try_levels(False, [(0, 0)])  # display levels start at 1
    _try_levels(False, [(2, 0)])  # display levels start at 1
    _try_levels(False, [(1, 0), (3, 0)])  # missing idlvl == 2
    _try_levels(False, [(1, 0), (2, 2)])  # attached to itself
    _try_levels(False, [(1, 0), (2, 3)])  # attached to non-existant idlvl
    _try_levels(False, [(1, 0), (2, 3), (3, 2)])  # attached to higher idlvl


@pytest.fixture
def example_segmented_sidd(tmp_path_factory, monkeypatch):
    """Force a SIDD to be segmented.  NOTE: segments are smaller than prescribed by the SIDD document"""
    sidd_etree = etree.parse(GOOD_SIDD_XML_PATH)
    tmp_sidd = (
        tmp_path_factory.mktemp("data") / GOOD_SIDD_XML_PATH.with_suffix(".sidd").name
    )
    nbytes = int(
        sidd_etree.find("./{*}Measurement/{*}PixelFootprint/{*}Col").text
    ) * int(sidd_etree.find("./{*}Measurement/{*}PixelFootprint/{*}Col").text)

    monkeypatch.setattr(
        sarkit.sidd._constants, "LI_MAX", nbytes // 5
    )  # reduce the segment size limit to force segmentation
    sec = {"security": {"clas": "U"}}
    sidd_meta = sarkit.sidd.NitfMetadata(
        file_header_part={"ostaid": "nowhere"} | sec,
        images=[
            sarkit.sidd.NitfProductImageMetadata(
                xmltree=sidd_etree,
                im_subheader_part=sec,
                de_subheader_part=sec,
            )
        ],
    )
    with tmp_sidd.open("wb") as f, sarkit.sidd.NitfWriter(f, sidd_meta):
        pass  # don't currently care about the pixels

    with tmp_sidd.open("rb") as f:
        jbp = jbpy.Jbp().load(f)
        assert len(jbp["ImageSegments"]) > 1

    yield tmp_sidd


def test_check_nitf_image_segmentation(example_segmented_sidd):
    with example_segmented_sidd.open("rb") as file:
        sidd_con = SiddConsistency.from_file(file)
    sidd_con.check("check_nitf_image_segmentation", allow_prefix=True)
    assert not sidd_con.failures()
