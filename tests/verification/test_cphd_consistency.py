import copy
import pathlib
import unittest.mock

import lxml.builder
import numpy as np
import pytest
import shapely.affinity
import shapely.geometry as shg
from lxml import etree

import sarkit.cphd as skcphd
import sarkit.verification._cphdcheck
import tests.utils
from sarkit.verification._cphd_consistency import CphdConsistency
from sarkit.verification._cphdcheck import main

from . import testing

DATAPATH = pathlib.Path(__file__).parents[2] / "data"

good_cphd_xml_path = DATAPATH / "example-cphd-1.0.1.xml"


@pytest.fixture(scope="session")
def example_cphd_file(example_cphd):
    assert not main([str(example_cphd), "--thorough"])
    with example_cphd.open("rb") as f:
        yield f


@pytest.fixture(scope="session")
def example_compressed_cphd(example_cphd_file, tmp_path_factory):
    example_cphd_file.seek(0)
    with skcphd.Reader(example_cphd_file) as r:
        pvps = r.read_pvps("1")

    new_meta = r.metadata
    assert new_meta.xmltree.find("{*}Data/{*}SignalCompressionID") is None
    ns = etree.QName(new_meta.xmltree.getroot()).namespace
    em = lxml.builder.ElementMaker(namespace=ns, nsmap={None: ns})
    data_chan_elem = new_meta.xmltree.find("{*}Data/{*}Channel")
    data_chan_elem.addprevious(em.SignalCompressionID("is constant!"))
    compressed_data = b"ultra-compressed"
    data_chan_elem.append(em.CompressedSignalSize(str(len(compressed_data))))

    tmp_cphd = tmp_path_factory.mktemp("data") / "faux-compressed.cphd"
    with tmp_cphd.open("wb") as f, skcphd.Writer(f, new_meta) as w:
        w.write_pvp("1", pvps)
        w.write_signal("1", np.frombuffer(compressed_data, np.uint8))
    assert not main([str(tmp_cphd), "--thorough"])
    with tmp_cphd.open("rb") as f:
        yield f


def remove_nodes(*nodes):
    for node in nodes:
        node.getparent().remove(node)


@pytest.fixture(scope="module")
def good_xml():
    return etree.parse(good_cphd_xml_path)


@pytest.fixture
def cphd_con(good_xml):
    return CphdConsistency.from_parts(good_xml)


@pytest.fixture
def good_xml_root(good_xml):
    return copy_xml(good_xml)


@pytest.fixture
def cphd_con_from_file(example_cphd_file):
    return CphdConsistency.from_file(example_cphd_file, thorough=True)


@pytest.fixture
def em(cphd_con_from_file):
    return lxml.builder.ElementMaker(
        namespace=etree.QName(cphd_con_from_file.cphdroot).namespace,
        nsmap=cphd_con_from_file.cphdroot.nsmap,
    )


def copy_xml(elem):
    return etree.fromstring(etree.tostring(elem))


@pytest.mark.parametrize(
    "fixture_name", ("example_cphd_file", "example_compressed_cphd")
)
def test_from_file_cphd(fixture_name, request):
    file = request.getfixturevalue(fixture_name)
    cphdcon = CphdConsistency.from_file(file, thorough=True)
    cphdcon.check()
    assert not cphdcon.failures()


def test_from_file_xml():
    cphdcon = CphdConsistency.from_file(str(good_cphd_xml_path))
    assert isinstance(cphdcon, CphdConsistency)
    cphdcon.check()
    assert not cphdcon.failures()


@pytest.mark.parametrize(
    "cphd_file",
    [good_cphd_xml_path] + list(DATAPATH.glob("example-cphd*.xml")),
)
def test_main(cphd_file):
    assert not main([str(cphd_file)])


@pytest.mark.parametrize("xml_file", (DATAPATH / "syntax_only/cphd").glob("*.xml"))
def test_smoketest(xml_file):
    main([str(xml_file)])


def test_main_with_ignore(good_xml_root, tmp_path):
    good_xml_root.find("./{*}Global/{*}SGN").text += "1"
    slightly_bad_xml = tmp_path / "slightly_bad.xml"
    etree.ElementTree(good_xml_root).write(str(slightly_bad_xml))
    assert main([str(slightly_bad_xml)])
    assert not main([str(slightly_bad_xml), "--ignore", "check_against_schema"])


def test_main_schema_args(cphd_con):
    good_schema = cphd_con.schema

    assert not main(
        [
            str(good_cphd_xml_path),
            "--schema",
            str(good_schema),
        ]
    )  # pass with actual schema

    assert main(
        [
            str(good_cphd_xml_path),
            "--schema",
            str(good_cphd_xml_path),
        ]
    )  # fails with bogus schema


def test_thorough(example_cphd_file):
    cphd_con = CphdConsistency.from_file(example_cphd_file, thorough=True)
    cphd_con.check()
    num_skips_thorough = len(cphd_con.skips(include_partial=True))

    cphd_con = CphdConsistency.from_file(example_cphd_file)
    cphd_con.check()
    num_skips_default = len(cphd_con.skips(include_partial=True))
    assert num_skips_thorough < num_skips_default


def test_xml_schema_error(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    remove_nodes(bad_xml.find("./{*}Global/{*}DomainType"))
    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_against_schema")
    assert cphd_con.failures()


def test_txrcv_bad_txwfid(good_xml_root):
    root = copy_xml(good_xml_root)

    chan_param_node = root.find("./{*}Channel/{*}Parameters/[{*}Identifier='1']")
    chan_param_node.find("./{*}TxRcv/{*}TxWFId").text = "missing"

    cphd_con = CphdConsistency(root)
    cphd_con.check("check_channel_txrcv_exist", allow_prefix=True)
    assert cphd_con.failures()


def test_txrcv_bad_rcvid(good_xml_root):
    root = copy_xml(good_xml_root)

    chan_param_node = root.find("./{*}Channel/{*}Parameters/[{*}Identifier='1']")
    chan_param_node.find("./{*}TxRcv/{*}RcvId").text = "missing"

    cphd_con = CphdConsistency(root)
    cphd_con.check("check_channel_txrcv_exist", allow_prefix=True)
    assert cphd_con.failures()


def test_antenna_bad_acf_count(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    antenna_node = bad_xml.find("./{*}Antenna")
    antenna_node.find("./{*}NumACFs").text += "2"
    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_antenna")
    assert cphd_con.failures()


def test_antenna_bad_apc_count(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    antenna_node = bad_xml.find("./{*}Antenna")
    antenna_node.find("./{*}NumAPCs").text += "2"
    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_antenna")
    assert cphd_con.failures()


def test_antenna_bad_antpats_count(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    antenna_node = bad_xml.find("./{*}Antenna")
    antenna_node.find("./{*}NumAntPats").text += "2"
    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_antenna")
    assert cphd_con.failures()


def test_antenna_non_matching_acfids(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    antenna_node = bad_xml.find("./{*}Antenna")
    antenna_node.findall("./{*}AntPhaseCenter/{*}ACFId")[-1].text = "wrong"
    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_antenna")
    assert cphd_con.failures()


def test_chan_antenna_no_apcs(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    apc_nodes = bad_xml.findall("./{*}Antenna/{*}AntPhaseCenter")
    for node in apc_nodes:
        remove_nodes(node)

    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_channel_antenna_exist", allow_prefix=True)
    assert cphd_con.failures()


def test_chan_antenna_no_apat(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    apat_nodes = bad_xml.findall("./{*}Antenna/{*}AntPattern")
    for node in apat_nodes:
        remove_nodes(node)

    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_channel_antenna_exist", allow_prefix=True)
    assert cphd_con.failures()


def test_txrcv_missing_channel_node(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    chan_ids = bad_xml.findall("./{*}Channel/{*}Parameters/{*}Identifier")
    chan_param_node = bad_xml.find(
        f"./{{*}}Channel/{{*}}Parameters/[{{*}}Identifier='{chan_ids[0].text}']"
    )

    remove_nodes(chan_param_node.findall("./{*}TxRcv")[0])
    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_txrcv_ids_in_channel")
    assert cphd_con.failures()


def test_check_extended_imagearea_x1y1_x2y2(cphd_con_from_file):
    scene_coords_node = cphd_con_from_file.cphdroot.find("./{*}SceneCoordinates")
    ext_area_elem = scene_coords_node.find("./{*}ExtendedArea")
    assert ext_area_elem is not None
    ia_x1y1 = cphd_con_from_file.xmlhelp.load_elem(
        scene_coords_node.find("./{*}ImageArea/{*}X1Y1")
    )
    ia_x2y2 = cphd_con_from_file.xmlhelp.load_elem(
        scene_coords_node.find("./{*}ImageArea/{*}X2Y2")
    )
    ia_poly = shg.box(*ia_x1y1, *ia_x2y2)
    ia_poly_shrink = shapely.affinity.scale(ia_poly, 0.5, 0.5)
    ext_area_elem[:] = [
        skcphd.XyType().make_elem("X1Y1", ia_poly_shrink.bounds[:2]),
        skcphd.XyType().make_elem("X2Y2", ia_poly_shrink.bounds[2:]),
    ]
    new_con = CphdConsistency(cphd_con_from_file.cphdroot, pvps=cphd_con_from_file.pvps)

    new_con.check("check_extended_imagearea_x1y1_x2y2")
    assert new_con.failures()


def test_channel_signal_data(cphd_con_from_file):
    cphd_con_from_file.kvp_list["SIGNAL_BLOCK_SIZE"] = "0"
    cphd_con_from_file.check("check_channel_signal_data", allow_prefix=True)
    assert cphd_con_from_file.failures()


def test_image_area_polygon_size_error(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    ia_polygon_node = bad_xml.find("./{*}SceneCoordinates/{*}ImageArea/{*}Polygon")
    ia_polygon_node.attrib["size"] = "12345678890"

    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_imagearea_polygon")
    assert cphd_con.failures()


def test_image_area_polygon_winding_error(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    ia_polygon_node = bad_xml.find("./{*}SceneCoordinates/{*}ImageArea/{*}Polygon")
    size = int(ia_polygon_node.attrib["size"])
    # Reverse the order of the vertices
    for vertex in ia_polygon_node:
        vertex.attrib["index"] = str(size - int(vertex.attrib["index"]) + 1)

    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_imagearea_polygon")
    assert cphd_con.failures()


def test_image_area_missing_corner_point_error(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    iacp_nodes = bad_xml.findall(
        "./{*}SceneCoordinates/{*}ImageAreaCornerPoints/{*}IACP"
    )
    remove_nodes(iacp_nodes[3])

    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_image_area_corner_points")
    assert cphd_con.failures()


def test_image_area_corner_pts_winding_error(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    iacp_nodes = bad_xml.findall(
        "./{*}SceneCoordinates/{*}ImageAreaCornerPoints/{*}IACP"
    )
    size = len(iacp_nodes)
    # Reverse the order of the vertices
    for vertex in iacp_nodes:
        vertex.attrib["index"] = str(size - int(vertex.attrib["index"]) + 1)

    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_image_area_corner_points")
    assert cphd_con.failures()


def test_check_channel_identifier_uniqueness(cphd_con):
    tx_wf_id = cphd_con.cphdroot.find("./{*}Channel/{*}Parameters/{*}TxRcv/{*}TxWFId")
    tx_wf_id.addnext(copy.deepcopy(tx_wf_id))
    cphd_con = CphdConsistency(cphd_con.cphdroot)

    cphd_con.check("check_channel_identifier_uniqueness", allow_prefix=True)
    assert cphd_con.failures()


def test_check_channel_rcv_sample_rate(cphd_con):
    for rcv_rate in cphd_con.cphdroot.findall(
        "./{*}TxRcv/{*}RcvParameters/{*}SampleRate"
    ):
        rcv_rate.text = "0"

    cphd_con.check("check_channel_rcv_sample_rate", allow_prefix=True)
    assert cphd_con.failures()


def test_extended_imagearea_polygon_bad_extent(good_xml_root):
    root = copy_xml(good_xml_root)
    root.find("./{*}SceneCoordinates/{*}ExtendedArea/{*}X2Y2/{*}X").text = "2000"
    cphd_con = CphdConsistency(root)

    cphd_con.check("check_extended_imagearea_polygon")
    assert cphd_con.failures()


def test_check_imagearea_x1y1_x2y2(good_xml_root):
    root = copy_xml(good_xml_root)

    cphd_con = CphdConsistency(root)

    x2y2_x = cphd_con.xmlhelp.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2/{*}X")
    cphd_con.xmlhelp.set("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1/{*}X", x2y2_x)

    cphd_con.check("check_imagearea_x1y1_x2y2")
    assert cphd_con.failures()


def test_check_channel_imagearea_x1y1(good_xml_root):
    root = copy_xml(good_xml_root)

    cphd_con = CphdConsistency(root)
    x2y2_y = cphd_con.xmlhelp.load(
        "./{*}Channel/{*}Parameters/{*}ImageArea/{*}X2Y2/{*}Y"
    )
    cphd_con.xmlhelp.set("./{*}Channel/{*}Parameters/{*}ImageArea/{*}X1Y1/{*}Y", x2y2_y)

    cphd_con.check("check_channel_imagearea_x1y1", allow_prefix=True)
    assert cphd_con.failures()


def test_antenna_missing_channel_node(good_xml_root):
    bad_xml = copy_xml(good_xml_root)

    remove_nodes(bad_xml.find("./{*}Channel/{*}Parameters/{*}Antenna"))

    cphd_con = CphdConsistency(bad_xml)
    cphd_con.check("check_antenna_ids_in_channel")
    assert cphd_con.failures()


def _invalidate_order(xml):
    poly_2d = xml.find("./{*}Dwell/{*}CODTime/{*}CODTimePoly")
    poly_2d.find("./{*}Coef").set("exponent1", "1" + poly_2d.get("order1"))


def _invalidate_coef_uniqueness(xml):
    poly_2d = xml.find("./{*}Dwell/{*}DwellTime/{*}DwellTimePoly")
    poly_2d.append(copy.deepcopy(poly_2d.find("./{*}Coef")))


@pytest.mark.parametrize(
    "invalidate_func", [_invalidate_order, _invalidate_coef_uniqueness]
)
def test_check_polynomials(invalidate_func, good_xml_root):
    cphd_con = CphdConsistency(good_xml_root)
    invalidate_func(cphd_con.cphdroot)

    cphd_con.check("check_polynomials")
    assert cphd_con.failures()


def _fxn_with_toa_domain(xml):
    xml.find("./{*}Global/{*}DomainType").text = "TOA"
    fx2 = xml.find("./{*}PVP/{*}FX2")
    # Add in reverse order so they are correctly ordered after FX2
    for name in ("FXN2", "FXN1"):
        if xml.find(f"./{{*}}PVP/{{*}}{name}") is None:
            new_elem = copy.deepcopy(fx2)
            new_elem.tag = name
            fx2.addnext(new_elem)


def _fxn1_only(xml):
    xml.find("./{*}Global/{*}DomainType").text = "FX"
    fx1 = xml.find("./{*}PVP/{*}FX1")
    remove_nodes(*xml.findall("./{*}PVP/{*}FXN1"), *xml.findall("./{*}PVP/{*}FXN2"))
    new_elem = copy.deepcopy(fx1)
    new_elem.tag = "{*}FXN1"
    # Insert FXN1 node directly after FX2 node
    fx2 = xml.find("./{*}PVP/{*}FX2")
    fx2.addnext(new_elem)

    fx1.getparent().append(new_elem)


@pytest.mark.parametrize("invalidate_func", [_fxn_with_toa_domain, _fxn1_only])
def test_check_optional_pvps_fx(invalidate_func, good_xml_root):
    cphd_con = CphdConsistency(good_xml_root)
    invalidate_func(cphd_con.cphdroot)
    cphd_con.check("check_optional_pvps_fx")
    assert cphd_con.failures()


def test_check_optional_pvps_toa(good_xml_root):
    cphd_con = CphdConsistency(good_xml_root)
    toa1 = cphd_con.cphdroot.find("./{*}PVP/{*}TOA1")
    remove_nodes(
        *cphd_con.cphdroot.findall("./{*}PVP/{*}TOAE1"),
        *cphd_con.cphdroot.findall("./{*}PVP/{*}TOAE2"),
    )

    new_elem = copy.deepcopy(toa1)
    new_elem.tag = "{*}TOAE1"
    # Insert TOAE1 node directly after TOA2 node
    toa2 = cphd_con.cphdroot.find("./{*}PVP/{*}TOA2")
    toa2.addnext(new_elem)
    cphd_con.check("check_optional_pvps_toa")
    assert cphd_con.failures()


def test_fxfixed_false(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = cphd_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        cphd_con.pvps[chan_id] = np.zeros(
            num_vects, dtype=[("FX1", "f8"), ("FX2", "f8")]
        )
        cphd_con.pvps[chan_id]["FX1"] = np.linspace(1.0, 1.0, num_vects)
        cphd_con.pvps[chan_id]["FX2"] = np.linspace(1.0, 1.0, num_vects)

        chan_param_node = cphd_con.cphdroot.find(
            f"./{{*}}Channel/{{*}}Parameters/[{{*}}Identifier='{chan_id}']"
        )
        chan_param_node.find("./{*}FXFixed").text = "false"

    cphd_con.check("check_channel_fxfixed", allow_prefix=True)
    assert len(cphd_con.failures()) == 1
    cphd_con.check("check_file_fxfixed")
    assert len(cphd_con.failures()) == 2


def test_fxfixed_false_with_nan(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = cphd_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        cphd_con.pvps[chan_id] = np.zeros(
            num_vects, dtype=[("FX1", "f8"), ("FX2", "f8")]
        )
        cphd_con.pvps[chan_id]["FX1"] = np.linspace(1.0, 1.1, num_vects)
        cphd_con.pvps[chan_id]["FX2"] = np.linspace(2.0, 2.2, num_vects)
        cphd_con.pvps[chan_id]["FX1"][0] = np.nan
        cphd_con.pvps[chan_id]["FX2"][0] = np.nan

        chan_param_node = cphd_con.cphdroot.find(
            f"./{{*}}Channel/{{*}}Parameters/[{{*}}Identifier='{chan_id}']"
        )
        chan_param_node.find("./{*}FXFixed").text = "false"

    cphd_con.check("check_channel_fxfixed", allow_prefix=True)
    assert not cphd_con.failures()


def test_fxfixed_true(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = cphd_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        cphd_con.pvps[chan_id] = np.zeros(
            num_vects, dtype=[("FX1", "f8"), ("FX2", "f8")]
        )
        cphd_con.pvps[chan_id]["FX1"] = np.linspace(1.0, 1.1, num_vects)
        cphd_con.pvps[chan_id]["FX2"] = np.linspace(2.0, 2.2, num_vects)

    cphd_con.check("check_channel_fxfixed", allow_prefix=True)
    assert len(cphd_con.failures()) == 1
    cphd_con.check("check_file_fxfixed")
    assert len(cphd_con.failures()) == 2


def test_fxfixed_true_with_nan(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["FX1"][0] = np.nan
        cphd_con.pvps[chan_id]["FX2"][0] = np.nan
    cphd_con.check("check_channel_fxfixed", allow_prefix=True)
    assert not cphd_con.failures()


def test_toafixed_false(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = cphd_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        cphd_con.pvps[chan_id] = np.zeros(
            num_vects, dtype=[("TOA1", "f8"), ("TOA2", "f8")]
        )
        cphd_con.pvps[chan_id]["TOA1"] = np.linspace(1.0, 1.0, num_vects)
        cphd_con.pvps[chan_id]["TOA2"] = np.linspace(1.0, 1.0, num_vects)

        chan_param_node = cphd_con.cphdroot.find(
            f"./{{*}}Channel/{{*}}Parameters/[{{*}}Identifier='{chan_id}']"
        )
        chan_param_node.find("./{*}TOAFixed").text = "false"

    cphd_con.check("check_channel_toafixed", allow_prefix=True)
    assert len(cphd_con.failures()) == 1
    cphd_con.check("check_file_toafixed")
    assert len(cphd_con.failures()) == 2


def test_toafixed_false_with_nan(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = cphd_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        cphd_con.pvps[chan_id] = np.zeros(
            num_vects, dtype=[("TOA1", "f8"), ("TOA2", "f8")]
        )
        cphd_con.pvps[chan_id]["TOA1"] = np.linspace(1.0, 1.1, num_vects)
        cphd_con.pvps[chan_id]["TOA2"] = np.linspace(2.0, 2.2, num_vects)
        cphd_con.pvps[chan_id]["TOA1"][0] = np.nan
        cphd_con.pvps[chan_id]["TOA2"][0] = np.nan

        chan_param_node = cphd_con.cphdroot.find(
            f"./{{*}}Channel/{{*}}Parameters/[{{*}}Identifier='{chan_id}']"
        )
        chan_param_node.find("./{*}TOAFixed").text = "false"

    cphd_con.check("check_channel_toafixed", allow_prefix=True)
    assert not cphd_con.failures()


def test_toafixed_true(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = cphd_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        cphd_con.pvps[chan_id] = np.zeros(
            num_vects, dtype=[("TOA1", "f8"), ("TOA2", "f8")]
        )
        cphd_con.pvps[chan_id]["TOA1"] = np.linspace(1.0, 1.1, num_vects)
        cphd_con.pvps[chan_id]["TOA2"] = np.linspace(2.0, 2.2, num_vects)

    cphd_con.check("check_channel_toafixed", allow_prefix=True)
    assert len(cphd_con.failures()) == 1
    cphd_con.check("check_file_toafixed")
    assert len(cphd_con.failures()) == 2


def test_toafixed_true_with_nan(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["TOA1"][0] = np.nan
        cphd_con.pvps[chan_id]["TOA2"][0] = np.nan
    cphd_con.check("check_channel_toafixed", allow_prefix=True)
    assert not cphd_con.failures()


def test_srpfixed_false(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        chan_param_node = cphd_con.cphdroot.find(
            f"./{{*}}Channel/{{*}}Parameters/[{{*}}Identifier='{chan_id}']"
        )
        chan_param_node.find("./{*}SRPFixed").text = "false"

    cphd_con.check("check_channel_srpfixed", allow_prefix=True)
    assert len(cphd_con.failures()) == 1
    cphd_con.check("check_file_srpfixed")
    assert len(cphd_con.failures()) == 2


def test_srpfixed_true(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = cphd_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        cphd_con.pvps[chan_id] = np.zeros(num_vects, dtype=[("SRPPos", (">f8", (3,)))])
        cphd_con.pvps[chan_id]["SRPPos"][0] = np.array([10.0, 0.0, 0.0])

    cphd_con.check("check_channel_srpfixed", allow_prefix=True)
    assert len(cphd_con.failures()) == 1
    cphd_con.check("check_file_srpfixed")
    assert len(cphd_con.failures()) == 2


def test_global_txtime_limits(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.cphdroot.find("./{*}Global/{*}Timeline/{*}TxTime2").text = "10000"
    cphd_con.cphdroot.find("./{*}Global/{*}Timeline/{*}TxTime1").text = "10000"

    cphd_con.check("check_global_txtime_limits")
    assert cphd_con.failures()


def test_global_fx_band(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.cphdroot.find("./{*}Global/{*}FxBand/{*}FxMin").text = "10000"
    cphd_con.cphdroot.find("./{*}Global/{*}FxBand/{*}FxMax").text = "10000"

    cphd_con.check("check_global_fx_band")
    assert cphd_con.failures()


def test_global_toaswath(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.cphdroot.find("./{*}Global/{*}TOASwath/{*}TOAMin").text = "10000"
    cphd_con.cphdroot.find("./{*}Global/{*}TOASwath/{*}TOAMax").text = "10000"

    cphd_con.check("check_global_toaswath")
    assert cphd_con.failures()


def test_time_increasing(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["TxTime"][0] = 1000.0

    cphd_con.check("check_time_increasing", allow_prefix=True)
    assert cphd_con.failures()


def test_time_rcv_after_tx(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["TxTime"] = cphd_con.pvps[chan_id]["RcvTime"]

    cphd_con.check("check_rcv_after_tx", allow_prefix=True)
    assert cphd_con.failures()


def test_time_rcv_time_not_finite(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["RcvTime"][0] = np.inf

    cphd_con.check("check_rcv_finite", allow_prefix=True)
    assert cphd_con.failures()


def test_time_rcv_pos_not_finite(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["RcvPos"][0] = [np.inf, np.inf, np.inf]

    cphd_con.check("check_rcv_finite", allow_prefix=True)
    assert cphd_con.failures()


def test_afdop(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["aFDOP"][0] = 1000.0

    cphd_con.check("check_channel_afdop", allow_prefix=True)
    assert cphd_con.failures()


def test_afrr1(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["aFRR1"] = 1000.0

    cphd_con.check(
        "check_channel_afrr1",
        allow_prefix=True,
        ignore_patterns=["check_channel_afrr1_afrr2_relative"],
    )
    assert cphd_con.failures()


def test_afrr2(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["aFRR2"][0] = 1000.0

    cphd_con.check("check_channel_afrr2", allow_prefix=True)
    assert cphd_con.failures()


def test_channel_ia_poly(good_xml_root):
    badroot = good_xml_root
    badroot.find("./{*}Channel/{*}Parameters/{*}ImageArea/{*}X1Y1/{*}X").text = "0.0"
    badroot.find("./{*}Channel/{*}Parameters/{*}ImageArea/{*}X1Y1/{*}Y").text = "0.0"

    cphd_con = CphdConsistency(badroot)

    cphd_con.check("check_channel_imagearea_polygon", allow_prefix=True)
    assert cphd_con.failures()


def test_afrr_relative(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["aFRR1"] = 1000.0
        cphd_con.pvps[chan_id]["aFRR2"] = 1000.0

    cphd_con.check("check_channel_afrr1_afrr2_relative", allow_prefix=True)
    assert cphd_con.failures()


def test_afrr_relative_skip(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["aFRR1"] = 0
        cphd_con.pvps[chan_id]["aFRR2"] = 0

    cphd_con.check("check_channel_afrr1_afrr2_relative", allow_prefix=True)
    assert not cphd_con.failures()
    assert cphd_con.skips()


def test_signal_pvp(cphd_con_from_file):
    cphd_con = cphd_con_from_file

    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = cphd_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        cphd_con.pvps[chan_id] = np.zeros(num_vects, dtype=[("SIGNAL", "i8")])

    cphd_con.check("check_channel_normal_signal_pvp", allow_prefix=True)
    assert cphd_con.failures()


def test_header_filetype(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.file_type_header = "FAKE"

    cphd_con.check("check_file_type_header")
    assert cphd_con.failures()


def test_header_kvp_missing_sb_size(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    del cphd_con.kvp_list["SUPPORT_BLOCK_SIZE"]

    cphd_con.check("check_header_kvp_list")
    assert cphd_con.failures()


def test_header_kvp_0_sb_size(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.kvp_list["SUPPORT_BLOCK_SIZE"] = 0

    cphd_con.check("check_header_kvp_list")
    assert cphd_con.failures()


def test_header_bad_class(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.kvp_list["CLASSIFICATION"] = "FAKE_CLASSIFICATION"

    cphd_con.check("check_classification_and_release_info")
    assert cphd_con.failures()


def test_header_bad_release(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.kvp_list["RELEASE_INFO"] = "FAKE_RELEASE"

    cphd_con.check("check_classification_and_release_info")
    assert cphd_con.failures()


def test_no_dwelltime_node(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    remove_nodes(cphd_con.cphdroot.find("./{*}Dwell/{*}DwellTime"))

    cphd_con.check("check_channel_dwell_polys", allow_prefix=True)
    assert cphd_con.failures()


def test_bad_pvp_txtimes(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    for chan in cphd_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = cphd_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        cphd_con.pvps[chan_id]["TxTime"] = cphd_con.pvps[chan_id]["TxTime"][::-1]

    cphd_con.check("check_channel_dwell_polys", allow_prefix=True)
    assert cphd_con.failures()


def test_channel_signalnormal(cphd_con_from_file, em):
    remove_nodes(cphd_con_from_file.cphdroot.find("./{*}Channel"))

    cphd_con_from_file.cphdroot.append(
        em.Channel(
            em.Parameters(
                skcphd.TxtType().make_elem("Identifier", "1"),
                skcphd.BoolType().make_elem("SignalNormal", True),
            ),
        )
    )
    new_con = CphdConsistency(cphd_con_from_file.cphdroot, pvps=cphd_con_from_file.pvps)

    # Check SignalNormal set with no SIGNAL PVPs
    new_con.check("check_channel_signalnormal", allow_prefix=True)
    assert new_con.failures()

    # Add SIGNAL PVPs, but make them all 0
    for chan in new_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = new_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = new_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        new_con.pvps[chan_id] = np.zeros(num_vects, dtype=[("SIGNAL", "i8")])

    new_con.check("check_channel_signalnormal", allow_prefix=True)
    assert new_con.failures()


def test_bad_fxc(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.xmlhelp.set("./{*}Channel/{*}Parameters/{*}FxC", 0.0)

    cphd_con.check("check_channel_fxc", allow_prefix=True)
    assert cphd_con.failures()


def test_bad_fxbw(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.xmlhelp.set("./{*}Channel/{*}Parameters/{*}FxBW", 0.0)

    cphd_con.check("check_channel_fxbw", allow_prefix=True)
    assert cphd_con.failures()


def test_fxbwnoise(cphd_con_from_file, em):
    remove_nodes(cphd_con_from_file.cphdroot.find("./{*}Channel"))

    cphd_con_from_file.cphdroot.append(
        em.Channel(
            em.Parameters(
                skcphd.TxtType().make_elem("Identifier", "1"),
                skcphd.DblType().make_elem("FxBWNoise", 0.0),
            ),
        )
    )
    new_con = CphdConsistency(cphd_con_from_file.cphdroot, pvps=cphd_con_from_file.pvps)
    new_con.xmlhelp.set("./{*}Global/{*}DomainType", "NOTFX")

    new_con.check("check_channel_fxbwnoise", allow_prefix=True)
    assert new_con.failures()

    # Add bad FXN1 and FXN2 PVPs
    for chan in new_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = new_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = new_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        new_con.pvps[chan_id] = np.zeros(
            num_vects, dtype=[("FXN1", "f8"), ("FXN2", "f8")]
        )
        new_con.pvps[chan_id]["FXN2"] = np.ones(num_vects)

    new_con.check("check_channel_fxbwnoise", allow_prefix=True)
    assert new_con.failures()


def test_no_bad_toasaved(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.xmlhelp.set("./{*}Channel/{*}Parameters/{*}TOASaved", 10000.0)

    cphd_con.check("check_channel_toasaved", allow_prefix=True)
    assert cphd_con.failures()


def test_toaextsaved(cphd_con_from_file, em):
    remove_nodes(cphd_con_from_file.cphdroot.find("./{*}Channel"))

    cphd_con_from_file.cphdroot.append(
        em.Channel(
            em.Parameters(
                skcphd.TxtType().make_elem("Identifier", "1"),
                em.TOAExtended(skcphd.DblType().make_elem("TOAExtSaved", 0.0)),
            ),
        )
    )
    new_con = CphdConsistency(cphd_con_from_file.cphdroot, pvps=cphd_con_from_file.pvps)

    # Add bad TOAE1 and TOAE2 PVPs
    for chan in new_con.cphdroot.findall("./{*}Data/{*}Channel"):
        chan_id = new_con.xmlhelp.load_elem(chan.find("./{*}Identifier"))
        num_vects = new_con.xmlhelp.load_elem(chan.find("./{*}NumVectors"))
        new_con.pvps[chan_id] = np.zeros(
            num_vects, dtype=[("TOAE1", "f8"), ("TOAE2", "f8")]
        )
        new_con.pvps[chan_id]["TOAE2"] = np.ones(num_vects)

    new_con.check("check_channel_toaextsaved", allow_prefix=True)
    assert new_con.failures()


def test_channel_fx_osr(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    channel_pvps = next(iter(cphd_con.pvps.values()))
    channel_pvps["TOA2"][0] = channel_pvps["TOA1"][0] + 1.0 / channel_pvps["SCSS"][0]

    cphd_con.check("check_channel_fx_osr", allow_prefix=True)
    assert cphd_con.failures()
    assert not cphd_con.skips()

    cphd_con.check("check_channel_toa_osr", allow_prefix=True)
    assert cphd_con.skips()


def test_channel_toa_osr(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.cphdroot.find("./{*}Global/{*}DomainType").text = "TOA"
    for channel_pvps in cphd_con.pvps.values():
        needed_ss = 1.0 / (1.25 * (channel_pvps["FX2"] - channel_pvps["FX1"]))
        channel_pvps["SCSS"][:] = needed_ss
    cphd_con.check("check_channel_toa_osr", allow_prefix=True)
    assert not cphd_con.failures()
    assert not cphd_con.skips()

    channel_pvps["FX2"][0] = channel_pvps["FX1"][0] + 1.0 / channel_pvps["SCSS"][0]
    cphd_con.check("check_channel_toa_osr", allow_prefix=True)
    assert cphd_con.failures()
    assert not cphd_con.skips()

    cphd_con.check("check_channel_fx_osr", allow_prefix=True)
    assert cphd_con.skips()


def test_channel_global_txtime(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.cphdroot.find("./{*}Global/{*}Timeline/{*}TxTime1").text = "10000"

    cphd_con.check("check_channel_global_txtime", allow_prefix=True)
    assert cphd_con.failures()


def test_channel_global_fxminmax(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.cphdroot.find("./{*}Global/{*}FxBand/{*}FxMax").text = "0"

    cphd_con.check("check_channel_global_fxminmax", allow_prefix=True)
    assert cphd_con.failures()


def test_channel_global_toaswath(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.cphdroot.find("./{*}Global/{*}TOASwath/{*}TOAMin").text = "10000"

    cphd_con.check("check_channel_global_toaswath", allow_prefix=True)
    assert cphd_con.failures()


def test_no_channel_ant(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    remove_nodes(cphd_con.cphdroot.find("./{*}Channel/{*}Parameters/{*}Antenna"))

    cphd_con.check("check_antenna_ids_in_channel", allow_prefix=True)
    assert cphd_con.failures()


def test_no_channel_txrcv(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    remove_nodes(cphd_con.cphdroot.find("./{*}Channel/{*}Parameters/{*}TxRcv"))

    cphd_con.check("check_txrcv_ids_in_channel", allow_prefix=True)
    assert cphd_con.failures()


def test_unique_identifiers(cphd_con_from_file, em):
    remove_nodes(cphd_con_from_file.cphdroot.find("./{*}Channel"))

    cphd_con_from_file.cphdroot.append(
        em.Channel(
            em.Parameters(
                skcphd.TxtType().make_elem("Identifier", "1"),
                skcphd.TxtType().make_elem("Identifier", "1"),
            ),
        )
    )
    new_con = CphdConsistency(cphd_con_from_file.cphdroot, pvps=cphd_con_from_file.pvps)

    new_con.check("check_identifier_uniqueness", allow_prefix=True)
    assert new_con.failures()


def test_refgeom_bad_root(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    bad_node = cphd_con.cphdroot.find("./{*}ReferenceGeometry/{*}SRPCODTime")
    bad_node.text = "24" + bad_node.text

    cphd_con.check("check_refgeom")
    testing.assert_failures(cphd_con, "SRPCODTime matches*")


def test_refgeom_bad_monostatic(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    bad_node = cphd_con.cphdroot.find(
        "./{*}ReferenceGeometry/{*}Monostatic/{*}AzimuthAngle"
    )
    bad_node.text = str((float(bad_node.text) + 3) % 360)

    cphd_con.check("check_refgeom")
    testing.assert_failures(cphd_con, "AzimuthAngle matches*")


def test_image_grid_exists(cphd_con):
    remove_nodes(cphd_con.cphdroot.find("./{*}SceneCoordinates/{*}ImageGrid"))

    cphd_con.check("check_image_grid_exists")
    assert cphd_con.failures()


def test_scene_plane_axis_vectors_bad_uiax(cphd_con):
    cphd_con.xmlhelp.set(
        "./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX/{*}X", 10000
    )

    cphd_con.check("check_scene_plane_axis_vectors")
    assert cphd_con.failures()


def test_scene_plane_axis_vectors_bad_uiay(cphd_con):
    cphd_con.xmlhelp.set(
        "./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY/{*}X", 10000
    )

    cphd_con.check("check_scene_plane_axis_vectors")
    assert cphd_con.failures()


def test_signal_at_end_of_file(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    cphd_con.kvp_list["SIGNAL_BLOCK_BYTE_OFFSET"] = "0"

    cphd_con.check("check_signal_at_end_of_file")
    assert cphd_con.failures()


def test_pad_after_pvp(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    pvp_offset = cphd_con.kvp_list["PVP_BLOCK_BYTE_OFFSET"]
    cphd_con.kvp_list["PVP_BLOCK_BYTE_OFFSET"] = cphd_con.kvp_list[
        "SIGNAL_BLOCK_BYTE_OFFSET"
    ]
    cphd_con.kvp_list["SIGNAL_BLOCK_BYTE_OFFSET"] = pvp_offset

    cphd_con.check("check_pad_after_pvp")
    assert cphd_con.failures()


def test_pad_after_support(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    sb_offset = cphd_con.kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"]
    cphd_con.kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"] = cphd_con.kvp_list[
        "PVP_BLOCK_BYTE_OFFSET"
    ]
    cphd_con.kvp_list["PVP_BLOCK_BYTE_OFFSET"] = sb_offset

    cphd_con.check("check_pad_after_support")
    assert cphd_con.failures()


def test_pad_after_xml(cphd_con_from_file):
    cphd_con = cphd_con_from_file
    xml_offset = cphd_con.kvp_list["XML_BLOCK_BYTE_OFFSET"]
    cphd_con.kvp_list["XML_BLOCK_BYTE_OFFSET"] = cphd_con.kvp_list[
        "SUPPORT_BLOCK_BYTE_OFFSET"
    ]
    cphd_con.kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"] = xml_offset

    cphd_con.check("check_pad_after_xml")
    assert len(cphd_con.failures()) == 1

    cphd_con.check("check_pad_header_xml")
    assert len(cphd_con.failures()) == 2


def test_channel_decorator_logic(good_xml_root):
    root = copy_xml(good_xml_root)
    # Add extra Data/Channel entries
    data_node = root.find("./{*}Data")
    data_node.find("./{*}NumCPHDChannels").text = "3"
    data_chan_node_1 = data_node.find("./{*}Channel[{*}Identifier='1']")
    data_chan_node_2 = copy.deepcopy(data_chan_node_1)
    data_chan_node_2.find("./{*}Identifier").text = "2"
    data_node.append(data_chan_node_2)
    data_chan_node_3 = copy.deepcopy(data_chan_node_1)
    data_chan_node_3.find("./{*}Identifier").text = "3"
    data_node.append(data_chan_node_3)

    # Add extra Channel/Parameters and invalidate them
    chan_node = root.find("./{*}Channel")
    params_node_1 = chan_node.find("./{*}Parameters[{*}Identifier='1']")
    params_node_2 = copy.deepcopy(params_node_1)
    params_node_2.find("./{*}Identifier").text = "2"
    params_node_2.find("./{*}ImageArea/{*}X1Y1/{*}X").text = "0.0"
    chan_node.append(params_node_2)
    params_node_3 = copy.deepcopy(params_node_1)
    params_node_3.find("./{*}Identifier").text = "3"
    params_node_3.find("./{*}ImageArea/{*}X1Y1/{*}Y").text = "0.0"
    chan_node.append(params_node_3)

    cphd_con = CphdConsistency(root)
    cphd_con.check("check_channel_imagearea_polygon", allow_prefix=True)
    # Test for channels 2 and 3 will fail
    assert len(cphd_con.failures()) == 2
    assert len(cphd_con.passes()) == 1


@pytest.mark.parametrize(
    "fixture_name", ("example_cphd_file", "example_compressed_cphd")
)
def test_check_signal_block_size_header(fixture_name, request):
    file = request.getfixturevalue(fixture_name)
    cphdcon = CphdConsistency.from_file(file)
    cphdcon.kvp_list["SIGNAL_BLOCK_SIZE"] += "1"
    cphdcon.check("check_signal_block_size_and_packing")
    testing.assert_failures(cphdcon, "SIGNAL_BLOCK_SIZE matches the end")


@pytest.mark.parametrize(
    "fixture_name", ("example_cphd_file", "example_compressed_cphd")
)
def test_check_signal_block_size_compressedsigsize(fixture_name, request):
    file = request.getfixturevalue(fixture_name)
    cphdcon = CphdConsistency.from_file(file)
    if cphdcon.cphdroot.find("{*}Data/{*}SignalCompressionID") is not None:
        remove_nodes(
            cphdcon.cphdroot.find("{*}Data/{*}Channel/{*}CompressedSignalSize")
        )
    else:
        em = lxml.builder.ElementMaker(
            namespace=etree.QName(cphdcon.cphdroot).namespace,
            nsmap=cphdcon.cphdroot.nsmap,
        )
        cphdcon.cphdroot.find("{*}Data/{*}Channel").append(
            em.CompressedSignalSize("24"),
        )
    cphdcon.check("check_signal_block_size_and_packing")
    testing.assert_failures(cphdcon, "CompressedSignalSize( not)? in Data/Channel")


def test_check_signal_block_packing(cphd_con):
    cphd_con.cphdroot.find("{*}Data/{*}Channel/{*}SignalArrayByteOffset").text += "1"
    cphd_con.check("check_signal_block_size_and_packing")
    testing.assert_failures(cphd_con, "SIGNAL array .+ starts at offset")


def test_smart_open_http(example_cphd):
    with tests.utils.static_http_server(example_cphd.parent) as server_url:
        assert not main([f"{server_url}/{example_cphd.name}", "--thorough"])


def test_smart_open_contract(example_cphd, monkeypatch):
    mock_open = unittest.mock.MagicMock(side_effect=tests.utils.simple_open_read)
    monkeypatch.setattr(sarkit.verification._cphdcheck, "open", mock_open)
    assert not main([str(example_cphd), "--thorough"])
    mock_open.assert_called_once_with(str(example_cphd), "rb")
