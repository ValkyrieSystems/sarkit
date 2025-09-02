import copy
import datetime
import pathlib
import unittest.mock

import lxml.builder
import pytest
from lxml import etree

import sarkit.verification._siddcheck
import tests.utils
from sarkit.verification._sidd_consistency import SiddConsistency
from sarkit.verification._siddcheck import main

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


def test_main_xml():
    assert not main([str(GOOD_SIDD_XML_PATH)])


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


@pytest.mark.parametrize(
    "invalidate_func",
    [
        _invalidate_validdata_index,
        _invalidate_validdata_size,
        _invalidate_validdata_simplicity,
        _invalidate_validdata_winding,
        _invalidate_validdata_first_vertex,
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
