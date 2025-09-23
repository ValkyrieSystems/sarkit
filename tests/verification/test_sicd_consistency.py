import copy
import pathlib
import unittest.mock

import jbpy
import lxml.builder
import numpy as np
import pytest
from lxml import etree

import sarkit.sicd as sksicd
import sarkit.verification._sicdcheck
import tests.utils
from sarkit.verification._sicd_consistency import SicdConsistency
from sarkit.verification._sicdcheck import main

from . import testing

DATAPATH = pathlib.Path(__file__).parents[2] / "data"

good_sicd_xml_path = DATAPATH / "example-sicd-1.2.1.xml"


@pytest.fixture(scope="session")
def example_sicd_file(example_sicd):
    assert not main([str(example_sicd)])
    with example_sicd.open("rb") as f:
        yield f


@pytest.fixture(scope="module")
def good_xml():
    return etree.parse(good_sicd_xml_path)


@pytest.fixture
def sicd_con(good_xml):
    return SicdConsistency.from_parts(copy.deepcopy(good_xml))


@pytest.fixture
def em(sicd_con):
    return lxml.builder.ElementMaker(
        namespace=etree.QName(sicd_con.sicdroot).namespace,
        nsmap=sicd_con.sicdroot.nsmap,
    )


def test_from_file_sicd(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    assert isinstance(sicdcon, SicdConsistency)
    sicdcon.check()
    assert len(sicdcon.failures()) == 0


def test_from_file_xml():
    sicdcon = SicdConsistency.from_file(str(good_sicd_xml_path))
    assert isinstance(sicdcon, SicdConsistency)
    sicdcon.check()
    assert len(sicdcon.failures()) == 0


@pytest.mark.parametrize(
    "file",
    [
        good_sicd_xml_path,
    ]
    + list(DATAPATH.glob("example-sicd*.xml")),
)
def test_main(file):
    assert not main([str(file), "-vv"])


@pytest.mark.parametrize("xml_file", (DATAPATH / "syntax_only/sicd").glob("*.xml"))
def test_smoketest(xml_file):
    main([str(xml_file)])


def test_main_schema_override():
    good_schema = sksicd.VERSION_INFO["urn:SICD:1.2.1"]["schema"]
    assert not main(
        [str(good_sicd_xml_path), "--schema", str(good_schema)]
    )  # pass with actual schema
    assert main(
        [
            str(good_sicd_xml_path),
            "--schema",
            str(good_sicd_xml_path),
        ]
    )  # fails with not schema


def _change_node(node, path, updated_val):
    node.find(path).text = str(updated_val)
    return node


def test_check_nitf_imseg_size_xtra_imseg(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    imsegs = sicdcon.ntf["ImageSegments"]
    imsegs._append(imsegs[0])
    sicdcon.check("check_nitf_imseg_size")
    testing.assert_failures(sicdcon, "Consistent number of image segments")


def test_check_nitf_imseg_size_mismatch_iloc(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    imsegs = sicdcon.ntf["ImageSegments"]
    imsegs[0]["subheader"]["ILOC"].value = (10000, 10000)
    sicdcon.check("check_nitf_imseg_size")
    testing.assert_failures(sicdcon, "ILOC matches expected")


@pytest.mark.parametrize("dim", ["NROWS", "NCOLS"])
def test_check_nitf_imseg_size_mismatch_dir(example_sicd_file, dim):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    imsegs = sicdcon.ntf["ImageSegments"]
    imsegs[0]["subheader"][f"{dim}"].value = 10000
    sicdcon.check("check_nitf_imseg_size")
    testing.assert_failures(sicdcon, f"Matching {dim}")


def test_check_nitf_imseg_lvls_idlvl(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    imsegs = sicdcon.ntf["ImageSegments"]
    imsegs[0]["subheader"]["IDLVL"].value = 99
    sicdcon.check("check_nitf_imseg_lvls")
    testing.assert_failures(
        sicdcon, "Consistent NITF inter-Image Subheaders Display levels"
    )


def test_check_nitf_imseg_lvls_ialvl(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    imsegs = sicdcon.ntf["ImageSegments"]
    imsegs[0]["subheader"]["IALVL"].value = 99
    sicdcon.check("check_nitf_imseg_lvls")
    testing.assert_failures(
        sicdcon, "Consistent NITF inter-Image Subheaders Attachment levels"
    )


def test_check_des_subheader_desid(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    des_header = sicdcon.ntf["DataExtensionSegments"][0]["subheader"]
    des_header["DESID"].value = "CONTENT_DATA_XML"

    sicdcon.check("check_des_subheader")
    testing.assert_failures(sicdcon, "DESID == XML_DATA_CONTENT")


def test_check_des_subheader_desshft(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    des_header = sicdcon.ntf["DataExtensionSegments"][0]["subheader"]
    des_header["DESSHF"]["DESSHFT"].value = "LXM"

    sicdcon.check("check_des_subheader")
    testing.assert_failures(sicdcon, "DESSHFT == XML")


def test_check_des_subheader_desshsi(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    des_header = sicdcon.ntf["DataExtensionSegments"][0]["subheader"]
    des_header["DESSHF"]["DESSHSI"].value = "SICD Volume 20"

    sicdcon.check("check_des_subheader")
    testing.assert_failures(sicdcon, "DESSHSI == SICD Volume 1...")


def test_check_des_subheader_desshtn(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    des_header = sicdcon.ntf["DataExtensionSegments"][0]["subheader"]
    des_header["DESSHF"]["DESSHTN"].value = "urn:SICD:10.20.10"

    sicdcon.check("check_des_subheader")
    testing.assert_failures(sicdcon, "Consistent namespace")


def test_check_des_subheader_desshlpg(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    des_header = sicdcon.ntf["DataExtensionSegments"][0]["subheader"]
    des_header["DESSHF"]["DESSHLPG"].value = "badcorners"

    sicdcon.check("check_des_subheader")
    testing.assert_failures(sicdcon, "DESSHLPG consistent with image corners")


def test_check_nitf_igeolo(example_sicd_file):
    sicdcon = SicdConsistency.from_file(example_sicd_file)
    corners = sicdcon.xmlhelp.load("./{*}GeoData/{*}ImageCorners")
    corners[0] = [0, 0]
    sicdcon.xmlhelp.set("./{*}GeoData/{*}ImageCorners", corners)
    sicdcon.check("check_nitf_igeolo")
    testing.assert_failures(sicdcon, "IGEOLO close to ICP Lon/Lat")


def _invalidate_1d_poly_coefs(node):
    # append a duplicate entry. will fail checks:
    #    Exponents are unique
    duplicate_coef = copy.deepcopy(node.find("./{*}Coef[last()]"))
    node.append(duplicate_coef)
    # add additional coefficient with exponent > order
    new_coef = copy.deepcopy(duplicate_coef)
    order = int(node.attrib["order1"])
    new_coef.attrib["exponent1"] = str(order + 1)
    node.append(new_coef)
    return node


def _invalidate_2d_poly_coefs(node):
    # append a duplicate entry. will fail checks:
    #    Exponents are unique
    duplicate_coef = copy.deepcopy(node.find("./{*}Coef[last()]"))
    node.append(duplicate_coef)
    # add additional coefficient with exponent > order
    new_coef = copy.deepcopy(duplicate_coef)
    order1 = int(node.attrib["order1"])
    order2 = int(node.attrib["order2"])
    new_coef.attrib["exponent1"] = str(order1 + 1)
    new_coef.attrib["exponent2"] = str(order2 + 1)
    node.append(new_coef)
    return node


def _invalidate_xyz_poly_coefs(node):
    for dim in ["X", "Y", "Z"]:
        _invalidate_1d_poly_coefs(node.find(f"./{{*}}{dim}"))


def _tstart_greater_than_zero(xml):
    _change_node(xml, "./{*}Timeline/{*}IPP/{*}Set/{*}TStart", 1)


def _tend_less_than_duration(xml):
    duration = float(xml.find("./{*}Timeline/{*}CollectDuration").text)
    xml.findall("./{*}Timeline/{*}IPP/{*}Set/{*}TEnd")[1].text = str(duration * 0.9)


def _tstart_greater_than_tend(xml):
    _change_node(xml, "./{*}Timeline/{*}IPP/{*}Set/{*}TEnd", 0)
    _change_node(xml, "./{*}Timeline/{*}IPP/{*}Set/{*}TStart", 1)


def _ippstart_greater_than_ippend(xml):
    _change_node(xml, "./{*}Timeline/{*}IPP/{*}Set/{*}IPPEnd", 0)
    _change_node(xml, "./{*}Timeline/{*}IPP/{*}Set/{*}IPPStart", 1)


def _increasing_tstart(xml):
    ipp2_tstart = xml.findall("./{*}Timeline/{*}IPP/{*}Set/{*}TStart")[1].text
    xml.findall("./{*}Timeline/{*}IPP/{*}Set/{*}TStart")[1].text = xml.findall(
        "./{*}Timeline/{*}IPP/{*}Set/{*}TStart"
    )[0].text
    xml.findall("./{*}Timeline/{*}IPP/{*}Set/{*}TStart")[0].text = ipp2_tstart


def _increasing_tend(xml):
    ipp2_tend = xml.findall("./{*}Timeline/{*}IPP/{*}Set/{*}TEnd")[1].text
    xml.findall("./{*}Timeline/{*}IPP/{*}Set/{*}TEnd")[1].text = xml.findall(
        "./{*}Timeline/{*}IPP/{*}Set/{*}TEnd"
    )[0].text
    xml.findall("./{*}Timeline/{*}IPP/{*}Set/{*}TEnd")[0].text = ipp2_tend


def _inconsistent_time_range(xml):
    ipp2_tend = float(xml.findall("./{*}Timeline/{*}IPP/{*}Set/{*}TEnd")[1].text)
    _change_node(xml, "./{*}Timeline/{*}IPP/{*}Set/{*}TEnd", ipp2_tend * 2)


@pytest.mark.parametrize(
    "invalidate_func",
    [
        _tstart_greater_than_zero,
        _tend_less_than_duration,
        _tstart_greater_than_tend,
        _ippstart_greater_than_ippend,
        _increasing_tstart,
        _increasing_tend,
        _inconsistent_time_range,
    ],
)
def test_ipp_poly(sicd_con, invalidate_func):
    invalidate_func(sicd_con.sicdroot)
    sicd_con.check("check_ipp_poly")
    assert sicd_con.failures()


def _invalid_position(xml):
    _change_node(xml, "./{*}SCPCOA/{*}ARPPos/{*}X", 0)


def _invalid_velocity(xml):
    _change_node(xml, "./{*}SCPCOA/{*}ARPVel/{*}X", 0)


def _invalid_acceleration(xml):
    _change_node(xml, "./{*}SCPCOA/{*}ARPAcc/{*}X", 0)


@pytest.mark.parametrize(
    "invalidate_func", [_invalid_position, _invalid_velocity, _invalid_acceleration]
)
def test_eval_scpcoa_bad_pos(sicd_con, invalidate_func):
    invalidate_func(sicd_con.sicdroot)
    sicd_con.check("check_scpcoa")
    assert sicd_con.failures()


def _remove_bistatic_params(con):
    bistatic_scpcoa = con.sicdroot.find("./{*}SCPCOA/{*}Bistatic")
    bistatic_scpcoa.getparent().remove(bistatic_scpcoa)


def _change_tx_apc_poly(con):
    con.xmlhelp.set(
        "./{*}Position/{*}TxAPCPoly",
        -1 * con.xmlhelp.load("./{*}Position/{*}TxAPCPoly"),
    )


def _change_slant_range(con):
    con.xmlhelp.set(
        ".//{*}TxPlatform/{*}SlantRange",
        10 * con.xmlhelp.load(".//{*}TxPlatform/{*}SlantRange"),
    )


@pytest.mark.parametrize(
    "invalidate_func",
    [_remove_bistatic_params, _change_tx_apc_poly, _change_slant_range],
)
def test_check_scpcoa_bistatic(invalidate_func):
    sicdcon = SicdConsistency.from_file(DATAPATH / "example-sicd-1.4.0.xml")
    assert sicdcon.xmlhelp.load(".//{*}CollectType") == "BISTATIC"
    invalidate_func(sicdcon)
    sicdcon.check("check_scpcoa")
    assert sicdcon.failures()


def test_pfa_fpn_away_from_earth(sicd_con):
    sicd_con.xmlhelp.set(
        "./{*}PFA/{*}FPN", -1 * sicd_con.xmlhelp.load("./{*}PFA/{*}FPN")
    )
    sicd_con.check("check_pfa_fpn_away_from_earth")
    assert sicd_con.failures()


def test_pfa_ipn_away_from_earth(sicd_con):
    sicd_con.xmlhelp.set(
        "./{*}PFA/{*}IPN", -1 * sicd_con.xmlhelp.load("./{*}PFA/{*}IPN")
    )
    sicd_con.check("check_pfa_ipn_away_from_earth")
    assert sicd_con.failures()


def test_pfa_ipn_with_grid(sicd_con):
    sicd_con.xmlhelp.set(
        "./{*}PFA/{*}IPN/{*}X", -1000.0 * sicd_con.xmlhelp.load("./{*}PFA/{*}IPN/{*}X")
    )
    sicd_con.check("check_pfa_ipn_with_grid")
    assert sicd_con.failures()


def test_pfa_proc_freq_min(sicd_con):
    min_proc = sicd_con.xmlhelp.load(
        "./{*}ImageFormation/{*}TxFrequencyProc/{*}MinProc"
    )
    sicd_con.xmlhelp.set(
        "./{*}ImageFormation/{*}TxFrequencyProc/{*}MinProc", 1000.0 * min_proc
    )
    sicd_con.check("check_pfa_proc_freq")
    assert sicd_con.failures()


def test_pfa_proc_freq_max(sicd_con):
    sicd_con.xmlhelp.set("./{*}ImageFormation/{*}TxFrequencyProc/{*}MaxProc", 0.0)
    sicd_con.check("check_pfa_proc_freq")
    assert sicd_con.failures()


def test_pfa_polar_ang_poly(sicd_con):
    sicd_con.xmlhelp.set("./{*}PFA/{*}PolarAngRefTime", 10.0)
    sicd_con.check("check_pfa_polar_ang_poly")
    assert sicd_con.failures()


def _invalid_num_apcs(xml):
    last_corner = xml.find("./{*}RadarCollection/{*}Area/{*}Corner/{*}ACP[last()]")
    last_corner.getparent().remove(last_corner)


def _ewrings_not_clockwise(xml):
    corners = xml.findall("./{*}RadarCollection/{*}Area/{*}Corner/{*}ACP")
    tmp = corners[1]
    corners[1] = corners[3]
    corners[3] = tmp
    corners[1].attrib["index"] = "2"
    corners[3].attrib["index"] = "4"


def _area_not_within_plane(xml):
    _change_node(xml, "./{*}RadarCollection/{*}Area/{*}Plane/{*}RefPt/{*}ECF/{*}X", 0)


@pytest.mark.parametrize(
    "invalidate_func",
    [_invalid_num_apcs, _ewrings_not_clockwise, _area_not_within_plane],
)
def test_radarcollection_area_corners(sicd_con, invalidate_func):
    invalidate_func(sicd_con.sicdroot)
    sicd_con.check("check_area_corners")
    assert sicd_con.failures()


def test_area_plane_valid_smoke(sicd_con):
    sicd_con.check("check_area_plane_valid")
    assert not sicd_con.failures()

    # Force the projection path and still PASS
    sicd_con.xmlhelp.set(
        "./{*}RadarCollection/{*}Area/{*}Plane/{*}RefPt/{*}ECF",
        np.array([6378138.0, 0.0, 0.0]),
    )
    sicd_con.check("check_area_plane_valid")
    assert not sicd_con.failures()

    # Force the projection path and FAIL
    sicd_con.xmlhelp.set(
        "./{*}RadarCollection/{*}Area/{*}Plane/{*}RefPt/{*}ECF",
        np.array([6378138.0, 25000.0, 0.0]),
    )
    sicd_con.check("check_area_plane_valid")
    assert sicd_con.failures()


def test_scp_ecf_llh(sicd_con):
    _change_node(sicd_con.sicdroot, "./{*}GeoData/{*}SCP/{*}ECF/{*}X", 1)
    sicd_con.check("check_scp_ecf_llh")
    assert sicd_con.failures()


def _invalid_num_icps(xml):
    last_corner = xml.find("./{*}GeoData/{*}ImageCorners/{*}ICP[last()]")
    last_corner.getparent().remove(last_corner)


def _misaligned_image_corners(xml):
    last_corner = xml.find("./{*}GeoData/{*}ImageCorners/{*}ICP[last()]")
    latitude = float(last_corner.findtext("./{*}Lat"))
    _change_node(last_corner, "./{*}Lat", -1.0 * latitude)


def _subimage_image_corners(xml):
    xml.find("./{*}ImageData/{*}FirstRow").text = str(
        int(xml.find("./{*}ImageData/{*}NumRows").text) // 2
    )


@pytest.mark.parametrize(
    "invalidate_func",
    [_invalid_num_icps, _misaligned_image_corners, _subimage_image_corners],
)
def test_image_corners(sicd_con, invalidate_func):
    invalidate_func(sicd_con.sicdroot)
    sicd_con.check("check_image_corners")
    assert sicd_con.failures()


def _tx_polarization_sequence_mismatch(xml):
    _change_node(xml, "./{*}RadarCollection/{*}TxPolarization", "SEQUENCE")


@pytest.mark.parametrize(
    "invalidate_func",
    [
        _tx_polarization_sequence_mismatch,
    ],
)
def test_tx_polarization(sicd_con, invalidate_func):
    invalidate_func(sicd_con.sicdroot)
    sicd_con.check("check_tx_polarization")
    assert sicd_con.failures()


def test_grid_normal_away_from_earth(sicd_con):
    sicd_con.xmlhelp.set(
        "./{*}Grid/{*}Row/{*}UVectECF",
        -1.0 * sicd_con.xmlhelp.load("./{*}Grid/{*}Row/{*}UVectECF"),
    )
    sicd_con.check("check_grid_normal_away_from_earth")
    assert sicd_con.failures()


def test_grid_shadows_downward(sicd_con):
    for d in ("Row", "Col"):
        sicd_con.xmlhelp.set(
            f"./{{*}}Grid/{{*}}{d}/{{*}}UVectECF",
            -1.0 * sicd_con.xmlhelp.load(f"./{{*}}Grid/{{*}}{d}/{{*}}UVectECF"),
        )
    sicd_con.check("check_grid_shadows_downward")
    assert sicd_con.failures()


def test_check_grid_sign(sicd_con):
    col_sgn = sicd_con.xmlhelp.load("./{*}Grid/{*}Col/{*}Sgn")
    sicd_con.xmlhelp.set("./{*}Grid/{*}Row/{*}Sgn", -col_sgn)
    sicd_con.check("check_grid_sign")
    testing.assert_failures(sicd_con, "Row and Col grid signs match")


def test_grid_uvect_orthogonal(sicd_con):
    sicd_con.xmlhelp.set(
        "./{*}Grid/{*}Row/{*}UVectECF",
        sicd_con.xmlhelp.load("./{*}Grid/{*}Col/{*}UVectECF"),
    )
    sicd_con.check("check_grid_uvect_orthogonal")
    assert sicd_con.failures()


@pytest.mark.parametrize("direction", ["Row", "Col"])
class TestGridNode:
    def test_deltak1_mismatch_with_poly(self, direction, sicd_con):
        _change_node(
            sicd_con.sicdroot, f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaK1", 10000
        )
        sicd_con.check(f"check_deltakpoly_{direction.lower()}")
        assert sicd_con.failures()

    def test_deltak2_mismatch_with_poly(self, direction, sicd_con):
        _change_node(
            sicd_con.sicdroot, f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaK2", 10000
        )
        sicd_con.check(f"check_deltakpoly_{direction.lower()}")
        assert sicd_con.failures()

    def test_deltak1_mismatch_with_ss(self, direction, sicd_con):
        dk1 = sicd_con.xmlhelp.load(f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaK1")
        _change_node(
            sicd_con.sicdroot, f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaK1", dk1 * 2
        )
        sicd_con.check(f"check_deltak_wrt_ss_{direction.lower()}")
        testing.assert_failures(sicd_con, f"{direction} DeltaKs must agree with SS")

    def test_deltak2_mismatch_with_ss(self, direction, sicd_con):
        _change_node(
            sicd_con.sicdroot, f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaK2", 10000
        )
        sicd_con.check(f"check_deltak_wrt_ss_{direction.lower()}")
        testing.assert_failures(sicd_con, f"{direction} DeltaKs must agree with SS")

    def test_deltak1_mismatch_with_deltak2(self, direction, sicd_con):
        dk1 = sicd_con.xmlhelp.load(f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaK1")
        dk2 = sicd_con.xmlhelp.load(f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaK2")
        _change_node(
            sicd_con.sicdroot, f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaK1", dk2
        )
        _change_node(
            sicd_con.sicdroot, f"./{{*}}Grid/{{*}}{direction}/{{*}}DeltaK2", dk1
        )
        sicd_con.check(f"check_deltak_wrt_ss_{direction.lower()}")
        testing.assert_failures(sicd_con, f"{direction} DeltaK2 >= DeltaK1")

    def test_check_iprbw_to_deltak(self, direction, sicd_con):
        sicd_con.xmlhelp.set(f"./{{*}}Grid/{{*}}{direction}/{{*}}ImpRespBW", 1e6)
        sicd_con.check(f"check_iprbw_to_deltak_{direction.lower()}")
        testing.assert_failures(sicd_con, f"{direction} ImpRespBW <= DeltaK2 - DeltaK1")

    def test_check_iprbw_to_ss(self, direction, sicd_con):
        sicd_con.xmlhelp.set(f"./{{*}}Grid/{{*}}{direction}/{{*}}ImpRespBW", 1e6)
        sicd_con.check(f"check_iprbw_to_ss_{direction.lower()}")
        testing.assert_failures(
            sicd_con, f"{direction} Impulse Response BW is supported by sample spacing"
        )

    def test_check_iprbw_to_ss_osr_too_small(self, direction, sicd_con):
        sicd_con.xmlhelp.set(f"./{{*}}Grid/{{*}}{direction}/{{*}}ImpRespBW", 1e6)
        sicd_con.check(f"check_iprbw_to_ss_osr_{direction.lower()}")
        testing.assert_failures(sicd_con, f"{direction} OSR >= 1.1")

    def test_check_iprbw_to_ss_osr_too_large(self, direction, sicd_con):
        sicd_con.xmlhelp.set(f"./{{*}}Grid/{{*}}{direction}/{{*}}ImpRespBW", 1e-6)
        sicd_con.check(f"check_iprbw_to_ss_osr_{direction.lower()}")
        testing.assert_failures(sicd_con, f"{direction} OSR <= 2.2")

    def test_check_pfa_ipr_bw(self, direction, sicd_con):
        sicd_con.xmlhelp.set(f"./{{*}}Grid/{{*}}{direction}/{{*}}ImpRespBW", 1e6)
        sicd_con.check(f"check_pfa_ipr_bw_{direction.lower()}")
        testing.assert_failures(sicd_con, f"{direction} IPR bandwidth supported by")

    def test_check_wgtfunct_indices_bad_index(self, direction, sicd_con, em):
        grid_node = sicd_con.sicdroot.find(f"./{{*}}Grid/{{*}}{direction}")
        grid_node.append(
            em.WgtFunct(
                em.Wgt("0", index="10"),
                em.Wgt("1", index="2"),
                em.Wgt("2", index="3"),
                em.Wgt("3", index="4"),
                size="4",
            ),
        )
        sicd_con.check(f"check_wgtfunct_indices_{direction.lower()}")
        testing.assert_failures(sicd_con, "Wgt elements are present")

    def test_check_wgtfunct_indices_bad_size(self, direction, sicd_con, em):
        grid_node = sicd_con.sicdroot.find(f"./{{*}}Grid/{{*}}{direction}")
        grid_node.append(
            em.WgtFunct(
                em.Wgt("0", index="10"),
                em.Wgt("1", index="2"),
                em.Wgt("2", index="3"),
                em.Wgt("3", index="4"),
                size="10",
            ),
        )
        sicd_con.check(f"check_wgtfunct_indices_{direction.lower()}")
        testing.assert_failures(sicd_con, "WgtFunct size attribute matches number")

    def test_grid_unit_vectors(self, direction, sicd_con):
        z_val = sicd_con.xmlhelp.load(
            f"./{{*}}Grid/{{*}}{direction}/{{*}}UVectECF/{{*}}Z"
        )
        sicd_con.xmlhelp.set(
            f"./{{*}}Grid/{{*}}{direction}/{{*}}UVectECF/{{*}}Z", -100.0 * z_val
        )
        sicd_con.check(f"check_grid_unit_vector_{direction.lower()}")
        assert sicd_con.failures()

    def test_check_uniform_ipr_width(self, direction, sicd_con, em):
        grid_node = sicd_con.sicdroot.find(f"./{{*}}Grid/{{*}}{direction}")
        grid_node.append(em.WgtType(em.WindowName("UNIFORM")))
        sicd_con.xmlhelp.set(f"./{{*}}Grid/{{*}}{direction}/{{*}}ImpRespWid", 1e6)
        sicd_con.check(f"check_uniform_ipr_width_{direction.lower()}")
        testing.assert_failures(
            sicd_con, f"Grid/{direction} uniform weighted IPR width matches bandwidth"
        )


@pytest.mark.parametrize("antenna", ["Tx", "Rcv", "TwoWay"])
class TestAntennaNode:
    @pytest.mark.parametrize("comptype", ("Array", "Elem"))
    @pytest.mark.parametrize("polytype", ("Gain", "Phase"))
    def test_gainphase_poly_constant(self, antenna, sicd_con, polytype, comptype):
        sicd_con.xmlhelp.set(
            f"./{{*}}Antenna/{{*}}{antenna}/{{*}}{comptype}/{{*}}{polytype}Poly/{{*}}Coef",
            1.0,
        )
        sicd_con.check(f"check_antenna_{comptype.lower()}_gain_phase")
        assert sicd_con.failures()

    def test_bs_poly_constant(self, antenna, sicd_con):
        sicd_con.xmlhelp.set(
            f"./{{*}}Antenna/{{*}}{antenna}/{{*}}GainBSPoly/{{*}}Coef", 1.0
        )
        sicd_con.check("check_antenna_bspoly_gain")
        assert sicd_con.failures()


def test_check_antenna_oneway_apc_tx(sicd_con):
    assert sicd_con.sicdroot.find("./{*}Antenna/{*}Tx") is not None
    apc_poly = sicd_con.sicdroot.find("./{*}Position/{*}TxAPCPoly")

    apc_poly.getparent().remove(apc_poly)
    sicd_con.check("check_antenna_oneway_apc")
    testing.assert_failures(
        sicd_con, "TxAPCPoly must be present when Antenna/Tx is present"
    )


def test_check_antenna_oneway_apc_rcv(sicd_con):
    assert sicd_con.sicdroot.find("./{*}Antenna/{*}Rcv") is not None
    apc_polys = sicd_con.sicdroot.findall("./{*}Position/{*}RcvAPC/{*}RcvAPCPoly")
    for apc_poly in apc_polys:
        apc_poly.getparent().remove(apc_poly)
    sicd_con.check("check_antenna_oneway_apc")
    testing.assert_failures(
        sicd_con, "RcvAPCPoly must be present when Antenna/Rcv is present"
    )


def test_grid_polys(sicd_con):
    _invalidate_2d_poly_coefs(sicd_con.sicdroot.find("./{*}Grid/{*}TimeCOAPoly"))
    for dir in ["Row", "Col"]:
        _invalidate_2d_poly_coefs(
            sicd_con.sicdroot.find(f"./{{*}}Grid/{{*}}{dir}/{{*}}DeltaKCOAPoly")
        )
    sicd_con.check("check_grid_polys")
    details = sicd_con.failures()["check_grid_polys"]["details"]
    assert np.all([det["passed"] is False for det in details])


def test_timeline_polys(sicd_con):
    ipp_sets = sicd_con.sicdroot.findall("./{*}Timeline/{*}IPP/{*}Set")
    for ipp_set in ipp_sets:
        _invalidate_1d_poly_coefs(ipp_set.find("./{*}IPPPoly"))
    sicd_con.check("check_timeline_polys")
    details = sicd_con.failures()["check_timeline_polys"]["details"]
    assert np.all([det["passed"] is False for det in details])


def test_check_segment_start_and_end(sicd_con, em):
    rca_plane = sicd_con.sicdroot.find("./{*}RadarCollection/{*}Area/{*}Plane")
    rca_plane.append(
        em.SegmentList(
            em.Segment(
                em.StartLine(str(0)),
                em.StartSample(str(0)),
                em.EndLine(str(-1)),
                em.EndSample(str(-1)),
            ),
        )
    )

    sicd_con.check("check_segment_start_and_end")
    testing.assert_failures(sicd_con, "SegmentList EndLine >= StartLine")
    testing.assert_failures(sicd_con, "SegmentList EndSample >= StartSample")


def test_check_tx_rf_bandwidth(sicd_con):
    max_coll = sicd_con.xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Max")
    sicd_con.xmlhelp.set(
        "./{*}RadarCollection/{*}Waveform/{*}WFParameters/{*}TxRFBandwidth", max_coll
    )
    sicd_con.check("check_tx_rf_bandwidth")
    testing.assert_failures(sicd_con, "Waveform TxBW must be within the collected BW")
    testing.assert_failures(sicd_con, "Derived BW must be close to the TxRFBW")


def _tx_freq_start_gt_max(xmlhelp):
    max_freq = xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Max")
    xmlhelp.set(
        "./{*}RadarCollection/{*}Waveform/{*}WFParameters/{*}TxFreqStart", max_freq * 2
    )


def _tx_freq_start_lt_min(xmlhelp):
    min_freq = xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Min")
    xmlhelp.set(
        "./{*}RadarCollection/{*}Waveform/{*}WFParameters/{*}TxFreqStart", min_freq / 2
    )


@pytest.mark.parametrize(
    "invalidate_tx_freq_start",
    [
        {
            "func": _tx_freq_start_gt_max,
            "error": "Waveform TxFreqStart <= max collected frequency",
        },
        {
            "func": _tx_freq_start_lt_min,
            "error": "Waveform TxFreqStart >= the min collected frequency",
        },
    ],
)
def test_check_tx_freq_start(sicd_con, invalidate_tx_freq_start):
    invalidate_tx_freq_start["func"](sicd_con.xmlhelp)
    sicd_con.check("check_tx_freq_start")
    testing.assert_failures(sicd_con, invalidate_tx_freq_start["error"])


@pytest.mark.parametrize(
    "invalidate_tx_freq_bounds",
    [
        {
            "func": _tx_freq_start_gt_max,
            "error": "Computed waveform end frequency <= max collected frequency",
        },
        {
            "func": _tx_freq_start_lt_min,
            "error": "Computed waveform end frequency >= min collected frequency",
        },
    ],
)
def test_check_tx_freq_bounds(sicd_con, invalidate_tx_freq_bounds):
    invalidate_tx_freq_bounds["func"](sicd_con.xmlhelp)
    sicd_con.check("check_tx_freq_bounds")
    testing.assert_failures(sicd_con, invalidate_tx_freq_bounds["error"])


def test_position_polys(sicd_con):
    for poly in ["ARPPoly", "GRPPoly", "TxAPCPoly", "RcvAPC/{*}RcvAPCPoly"]:
        _invalidate_xyz_poly_coefs(
            sicd_con.sicdroot.find(f"./{{*}}Position/{{*}}{poly}")
        )
    sicd_con.check("check_position_polys")
    details = sicd_con.failures()["check_position_polys"]["details"]
    assert np.all([det["passed"] is False for det in details])


def test_radiometric_polys(sicd_con):
    for poly in [
        "NoiseLevel/{*}NoisePoly",
        "RCSSFPoly",
        "SigmaZeroSFPoly",
        "BetaZeroSFPoly",
        "GammaZeroSFPoly",
    ]:
        _invalidate_2d_poly_coefs(
            sicd_con.sicdroot.find(f"./{{*}}Radiometric/{{*}}{poly}")
        )
    sicd_con.check("check_radiometric_polys")
    details = sicd_con.failures()["check_radiometric_polys"]["details"]
    assert np.all([det["passed"] is False for det in details])


def _mismatch_chirp(xmlhelp):
    xmlhelp.set(
        "./{*}RadarCollection/{*}Waveform/{*}WFParameters/{*}RcvDemodType", "CHIRP"
    )


def _mismatch_stretch(xmlhelp):
    xmlhelp.set("./{*}RadarCollection/{*}Waveform/{*}WFParameters/{*}RcvFMRate", 0.0)


@pytest.mark.parametrize(
    "invalidate_fmrate",
    [
        {
            "func": _mismatch_chirp,
            "error": "Consistent receive FM rate for chirp/stretch demodulation types",
        },
        {
            "func": _mismatch_stretch,
            "error": "Consistent receive FM rate for chirp/stretch demodulation types",
        },
    ],
)
def test_check_rcv_fmrate(sicd_con, invalidate_fmrate):
    sicd_con.check("check_rcv_fmrate")
    assert not sicd_con.failures()

    invalidate_fmrate["func"](sicd_con.xmlhelp)
    sicd_con.check("check_rcv_fmrate")
    testing.assert_failures(sicd_con, invalidate_fmrate["error"])


def test_antenna_polys(sicd_con):
    for ant in ["Tx", "Rcv", "TwoWay"]:
        for poly in ["EB/{*}DCXPoly", "EB/{*}DCYPoly", "GainBSPoly"]:
            _invalidate_1d_poly_coefs(
                sicd_con.sicdroot.find(f"./{{*}}Antenna/{{*}}{ant}/{{*}}{poly}")
            )

        for poly in [
            "Array/{*}GainPoly",
            "Array/{*}PhasePoly",
            "Elem/{*}GainPoly",
            "Elem/{*}PhasePoly",
        ]:
            _invalidate_2d_poly_coefs(
                sicd_con.sicdroot.find(f"./{{*}}Antenna/{{*}}{ant}/{{*}}{poly}")
            )

        for poly in ["XAxisPoly", "YAxisPoly"]:
            _invalidate_xyz_poly_coefs(
                sicd_con.sicdroot.find(f"./{{*}}Antenna/{{*}}{ant}/{{*}}{poly}")
            )
    sicd_con.check("check_antenna_polys")
    details = sicd_con.failures()["check_antenna_polys"]["details"]
    assert np.all([det["passed"] is False for det in details])


def test_rgazcomp_polys(sicd_con, em):
    sicd_con.sicdroot.append(
        em.RgAzComp(
            em.AzSF("0.0"),
            sksicd.PolyType().make_elem("KazPoly", np.zeros(4)),
        )
    )
    sicd_con.check("check_rgazcomp_polys")
    assert sicd_con.passes()
    _invalidate_1d_poly_coefs(sicd_con.sicdroot.find("./{*}RgAzComp/{*}KazPoly"))
    sicd_con.check("check_rgazcomp_polys")
    assert sicd_con.failures()


def test_check_match_type(sicd_con, em):
    sicd_con.sicdroot.append(
        em.MatchInfo(
            em.NumMatchTypes("2"),
            em.MatchType(
                em.TypeID("BOGUS"),
                em.NumMatchCollections("0"),
                index="1",
            ),
        )
    )
    sicd_con.check("check_match_type")
    testing.assert_failures(sicd_con, "Number of MatchType nodes matches NumMatchTypes")


def test_check_match_type_indices(sicd_con, em):
    sicd_con.sicdroot.append(
        em.MatchInfo(
            em.NumMatchTypes("2"),
            em.MatchType(
                em.TypeID("BOGUS1"),
                em.NumMatchCollections("0"),
                index="1",
            ),
            em.MatchType(
                em.TypeID("BOGUS2"),
                em.NumMatchCollections("0"),
                index="10",
            ),
        )
    )
    sicd_con.check("check_match_type")
    testing.assert_failures(sicd_con, "MatchType indexed 1 to NumMatchTypes")


def test_check_no_match_collections(sicd_con, em):
    sicd_con.sicdroot.append(
        em.MatchInfo(
            em.NumMatchTypes("1"),
            em.MatchType(
                em.TypeID("BOGUS1"),
                em.NumMatchCollections("24"),
            ),
        )
    )
    sicd_con.check("check_match_collection")
    testing.assert_failures(
        sicd_con, "Number of MatchCollection nodes matches NumMatchCollections"
    )


def test_check_match_collection(sicd_con, em):
    sicd_con.sicdroot.append(
        em.MatchInfo(
            em.NumMatchTypes("2"),
            em.MatchType(
                em.TypeID("BOGUS1"),
                em.NumMatchCollections("1"),
                em.MatchCollection(
                    em.CoreName("CORENAME"),
                    index="1",
                ),
            ),
            em.MatchType(
                em.TypeID("BOGUS2"),
                em.NumMatchCollections("2"),
                em.MatchCollection(
                    em.CoreName("CORENAME"),
                    index="1",
                ),
            ),
        )
    )
    sicd_con.check("check_match_collection")
    testing.assert_failures(
        sicd_con, "Number of MatchCollection nodes matches NumMatchCollections"
    )


def test_check_match_collection_indices(sicd_con, em):
    sicd_con.sicdroot.append(
        em.MatchInfo(
            em.NumMatchTypes("2"),
            em.MatchType(
                em.TypeID("BOGUS1"),
                em.NumMatchCollections("1"),
                em.MatchCollection(
                    em.CoreName("CORENAME"),
                    index="1",
                ),
            ),
            em.MatchType(
                em.TypeID("BOGUS2"),
                em.NumMatchCollections("2"),
                em.MatchCollection(
                    em.CoreName("CORENAME1"),
                    index="1",
                ),
                em.MatchCollection(
                    em.CoreName("CORENAME2"),
                    index="10",
                ),
            ),
        )
    )
    sicd_con.check("check_match_collection")
    testing.assert_failures(
        sicd_con, "MatchCollection indexed 1 to NumMatchCollections"
    )


def test_check_valid_data_indices(sicd_con):
    sicd_con.check("check_valid_data_indices")
    assert not sicd_con.failures()

    vertices = sicd_con.sicdroot.findall("./{*}ImageData/{*}ValidData/{*}Vertex")
    vertices[0].set("index", "99")
    sicd_con.check("check_valid_data_indices")
    testing.assert_failures(sicd_con, "GeoData indices equal to ImageData indices")


def test_check_icp_indices(sicd_con):
    sicd_con.check("check_icp_indices")
    assert not sicd_con.failures()

    icp_0 = sicd_con.sicdroot.find("./{*}GeoData/{*}ImageCorners/{*}ICP")
    icp_0.set("index", "10:FRFR")
    sicd_con.check("check_icp_indices")
    testing.assert_failures(sicd_con, "GeoData ICPs indexed correctly")


@pytest.mark.parametrize(
    "seglist_failures",
    [
        {
            "index": "10",
            "size": "1",
            "error": "Segment elements are present",
        },
        {
            "index": "1",
            "size": "10",
            "error": "SegmentList size attribute matches number",
        },
    ],
)
def test_check_segmentlist_indices(sicd_con, seglist_failures, em):
    plane_node = sicd_con.sicdroot.find("./{*}RadarCollection/{*}Area/{*}Plane")
    plane_node.append(
        em.SegmentList(
            em.Segment(
                em.StartLine("0"),
                em.StartSample("0"),
                em.EndLine("100"),
                em.EndSample("100"),
                em.Identifier("1"),
                index=seglist_failures["index"],
            ),
            size=seglist_failures["size"],
        ),
    )
    sicd_con.check("check_segmentlist_indices")
    testing.assert_failures(sicd_con, seglist_failures["error"])


@pytest.mark.parametrize(
    "txsequence_failures",
    [
        {
            "index": "10",
            "size": "1",
            "error": "TxStep elements are present",
        },
        {
            "index": "1",
            "size": "10",
            "error": "TxSequence size attribute matches number",
        },
    ],
)
def test_check_txsequence_indices(sicd_con, txsequence_failures, em):
    rc_node = sicd_con.sicdroot.find("./{*}RadarCollection")
    rc_node.append(
        em.TxSequence(
            em.TxStep("25", index=txsequence_failures["index"]),
            size=txsequence_failures["size"],
        ),
    )
    sicd_con.check("check_txsequence_indices")
    testing.assert_failures(sicd_con, txsequence_failures["error"])


@pytest.mark.parametrize(
    "other_index_failures",
    [
        {
            "check": "check_waveform_params_indices",
            "node": "./{*}RadarCollection/{*}Waveform/{*}WFParameters",
            "error": "WFParameters elements are present",
        },
        {
            "check": "check_ipp_set_indices",
            "node": "./{*}Timeline/{*}IPP/{*}Set",
            "error": "Set elements are present",
        },
        {
            "check": "check_rcv_channel_indices",
            "node": "./{*}RadarCollection/{*}RcvChannels/{*}ChanParameters",
            "error": "ChanParameters elements are present",
        },
        {
            "check": "check_rcvapc_indices",
            "node": "./{*}Position/{*}RcvAPC/{*}RcvAPCPoly",
            "error": "RcvAPCPoly elements are present",
        },
    ],
)
def test_check_indices_bad_index(sicd_con, other_index_failures):
    sicd_con.check(other_index_failures["check"])
    assert not sicd_con.failures()

    index_0 = sicd_con.sicdroot.find(other_index_failures["node"])
    index_0.set("index", "10")
    sicd_con.check(other_index_failures["check"])
    testing.assert_failures(sicd_con, other_index_failures["error"])


@pytest.mark.parametrize(
    "size_failures",
    [
        {
            "check": "check_waveform_params_indices",
            "node": "./{*}RadarCollection/{*}Waveform",
            "error": "Waveform size attribute matches number",
        },
        {
            "check": "check_ipp_set_indices",
            "node": "./{*}Timeline/{*}IPP",
            "error": "IPP size attribute matches number",
        },
        {
            "check": "check_rcv_channel_indices",
            "node": "./{*}RadarCollection/{*}RcvChannels",
            "error": "RcvChannels size attribute matches number",
        },
        {
            "check": "check_rcvapc_indices",
            "node": "./{*}Position/{*}RcvAPC",
            "error": "RcvAPC size attribute matches number",
        },
    ],
)
def test_check_indices_bad_size(sicd_con, size_failures):
    sicd_con.check(size_failures["check"])
    assert not sicd_con.failures()

    sicd_con.sicdroot.find(size_failures["node"]).set("size", "20")
    sicd_con.check(size_failures["check"])
    testing.assert_failures(sicd_con, size_failures["error"])


def test_check_waveform_params_rcvwinlen(sicd_con):
    sicd_con.check("check_waveform_params")
    assert not sicd_con.failures()

    wf_params_node = sicd_con.sicdroot.find(
        "./{*}RadarCollection/{*}Waveform/{*}WFParameters"
    )
    wf_params_node.find("./{*}RcvWindowLength").text = "0.0"
    sicd_con.check("check_waveform_params")
    testing.assert_failures(sicd_con, "RcvWindowLength > zero")


def test_check_waveform_params_txfmrate(sicd_con):
    sicd_con.check("check_waveform_params")
    assert not sicd_con.failures()

    wf_params_node = sicd_con.sicdroot.find(
        "./{*}RadarCollection/{*}Waveform/{*}WFParameters"
    )
    wf_params_node.find("./{*}TxFMRate").text = "0.0"
    sicd_con.check("check_waveform_params")
    testing.assert_failures(sicd_con, "TxFMRate not zero")


def test_rgazcomp_ifa(sicd_con, em):
    sicd_con.sicdroot.append(
        em.RgAzComp(
            em.AzSF("0.0"),
            sksicd.PolyType().make_elem("KazPoly", np.zeros(4)),
        )
    )

    sicd_con.sicdroot.find("./{*}ImageFormation/{*}ImageFormAlgo").text = "RGAZCOMP"
    sicd_con.check("check_valid_ifa")
    assert not sicd_con.failures()


def test_check_validdata_first_vertex(sicd_con):
    sicd_con.check("check_validdata_first_vertex")
    assert not sicd_con.failures()

    vertices = sicd_con.xmlhelp.load("./{*}ImageData/{*}ValidData")
    vertices[0] = [np.max(vertices[:, 0]), np.max(vertices[:, 1])]
    sicd_con.xmlhelp.set("./{*}ImageData/{*}ValidData", vertices)
    sicd_con.check("check_validdata_first_vertex")
    testing.assert_failures(sicd_con, "First ValidData Vertex is min row -> min col")


def test_check_amptable_bad_index(sicd_con, em):
    img_data_node = sicd_con.sicdroot.find("./{*}ImageData")
    img_data_node.append(
        em.AmpTable(
            em.Amplitude("1", index="256"),
            em.Amplitude("2", index="1"),
            em.Amplitude("3", index="2"),
            em.Amplitude("4", index="3"),
            size="256",
        )
    )
    sicd_con.check("check_amptable")
    testing.assert_failures(sicd_con, "AmpTable indexed 0 to 255")


def test_check_multiple_geoinfo_lines(sicd_con, em):
    geo_data_node = sicd_con.sicdroot.find("./{*}GeoData")
    geo_data_node.append(
        em.GeoInfo(
            em.Line(
                em.Endpoint(
                    em.Lat("60.0"),
                    em.Lon("120.0"),
                    index="10",
                ),
                em.Endpoint(
                    em.Lat("70.0"),
                    em.Lon("130.0"),
                    index="2",
                ),
                size="2",
            ),
        )
    )
    geo_data_node.append(
        em.GeoInfo(
            em.Line(
                em.Endpoint(
                    em.Lat("65.0"),
                    em.Lon("125.0"),
                    index="1",
                ),
                em.Endpoint(
                    em.Lat("75.0"),
                    em.Lon("135.0"),
                    index="2",
                ),
                size="2",
            ),
        )
    )
    sicd_con.check("check_geoinfo_line")
    testing.assert_failures(sicd_con, "Endpoint elements are present")


def test_check_nested_geoinfo_lines(sicd_con, em):
    geo_data_node = sicd_con.sicdroot.find("./{*}GeoData")
    geo_data_node.append(
        em.GeoInfo(
            em.Line(
                em.Endpoint(
                    em.Lat("60.0"),
                    em.Lon("120.0"),
                    index="10",
                ),
                em.Endpoint(
                    em.Lat("70.0"),
                    em.Lon("130.0"),
                    index="2",
                ),
                size="2",
            ),
            em.GeoInfo(
                em.Line(
                    em.Endpoint(
                        em.Lat("65.0"),
                        em.Lon("125.0"),
                        index="1",
                    ),
                    em.Endpoint(
                        em.Lat("75.0"),
                        em.Lon("135.0"),
                        index="2",
                    ),
                    size="2",
                ),
            ),
        )
    )
    sicd_con.check("check_geoinfo_line")
    testing.assert_failures(sicd_con, "Endpoint elements are present")


def test_check_multiple_geoinfo_polygons(sicd_con, em):
    geo_data_node = sicd_con.sicdroot.find("./{*}GeoData")
    geo_data_node.append(
        em.GeoInfo(
            em.Polygon(
                em.Vertex(em.Lat("60.0"), em.Lon("120.0"), index="10"),
                em.Vertex(em.Lat("70.0"), em.Lon("130.0"), index="2"),
                em.Vertex(em.Lat("80.0"), em.Lon("140'0"), index="3"),
                size="3",
            ),
        )
    )
    geo_data_node.append(
        em.GeoInfo(
            em.Polygon(
                em.Vertex(em.Lat("65.0"), em.Lon("125.0"), index="10"),
                em.Vertex(em.Lat("75.0"), em.Lon("135.0"), index="2"),
                em.Vertex(em.Lat("85.0"), em.Lon("145'0"), index="3"),
                size="3",
            ),
        )
    )
    sicd_con.check("check_geoinfo_polygon")
    testing.assert_failures(sicd_con, "Vertex elements are present")


def test_check_nested_geoinfo_polygons(sicd_con, em):
    geo_data_node = sicd_con.sicdroot.find("./{*}GeoData")
    geo_data_node.append(
        em.GeoInfo(
            em.Polygon(
                em.Vertex(em.Lat("65.0"), em.Lon("125.0"), index="10"),
                em.Vertex(em.Lat("75.0"), em.Lon("135.0"), index="2"),
                em.Vertex(em.Lat("85.0"), em.Lon("145'0"), index="3"),
                size="3",
            ),
            em.GeoInfo(
                em.Polygon(
                    em.Vertex(em.Lat("65.0"), em.Lon("125.0"), index="1"),
                    em.Vertex(em.Lat("75.0"), em.Lon("135.0"), index="2"),
                    em.Vertex(em.Lat("85.0"), em.Lon("145'0"), index="3"),
                    size="3",
                ),
            ),
        )
    )
    sicd_con.check("check_geoinfo_polygon")
    testing.assert_failures(sicd_con, "Vertex elements are present")


def test_check_geoinfo_polygon_bad_size(sicd_con, em):
    geo_data_node = sicd_con.sicdroot.find("./{*}GeoData")
    geo_data_node.append(
        em.GeoInfo(
            em.Polygon(
                em.Vertex(em.Lat("60.0"), em.Lon("120.0"), index="1"),
                em.Vertex(em.Lat("70.0"), em.Lon("130.0"), index="2"),
                em.Vertex(em.Lat("80.0"), em.Lon("140'0"), index="3"),
                size="4",
            ),
        )
    )
    sicd_con.check("check_geoinfo_polygon")
    testing.assert_failures(sicd_con, "Polygon size attribute matches number")


def test_check_no_validdata_presence(sicd_con):
    id_validdata_node = sicd_con.sicdroot.find("./{*}ImageData/{*}ValidData")
    id_validdata_node.getparent().remove(id_validdata_node)
    gd_validdata_node = sicd_con.sicdroot.find("./{*}GeoData/{*}ValidData")
    gd_validdata_node.getparent().remove(gd_validdata_node)

    sicd_con.check("check_validdata_presence")
    assert not sicd_con.failures()


def _validdata_missing_from_imagedata(xml):
    validdata_node = xml.find("./{*}ImageData/{*}ValidData")
    validdata_node.getparent().remove(validdata_node)


def _validdata_missing_from_geodata(xml):
    validdata_node = xml.find("./{*}GeoData/{*}ValidData")
    validdata_node.getparent().remove(validdata_node)


@pytest.mark.parametrize(
    "missing_validdata",
    [
        {
            "func": _validdata_missing_from_imagedata,
            "error": "ValidData in both GeoData and ImageData or neither",
        },
        {
            "func": _validdata_missing_from_geodata,
            "error": "ValidData in both GeoData and ImageData or neither",
        },
    ],
)
def test_check_validdata_presence(sicd_con, missing_validdata):
    missing_validdata["func"](sicd_con.sicdroot)
    sicd_con.check("check_validdata_presence")
    testing.assert_failures(sicd_con, missing_validdata["error"])


def test_check_validdata_winding(sicd_con):
    sicd_con.check("check_validdata_winding")
    assert not sicd_con.failures()

    vertices = sicd_con.xmlhelp.load("./{*}ImageData/{*}ValidData")
    vertices = vertices[::-1]
    sicd_con.xmlhelp.set("./{*}ImageData/{*}ValidData", vertices)
    sicd_con.check("check_validdata_winding")
    testing.assert_failures(sicd_con, "Clockwise ValidData")


def test_check_validdata_simpleness(sicd_con):
    sicd_con.check("check_validdata_simpleness")
    assert not sicd_con.failures()

    vertices = sicd_con.xmlhelp.load("./{*}ImageData/{*}ValidData")
    vertices[[2, 3]] = vertices[[3, 2]]
    sicd_con.xmlhelp.set("./{*}ImageData/{*}ValidData", vertices)
    sicd_con.check("check_validdata_simpleness")
    testing.assert_failures(sicd_con, "Simple ValidData")


def test_check_pfa_grid_type(sicd_con):
    sicd_con.check("check_pfa_grid_type")
    assert sicd_con.passes()

    sicd_con.xmlhelp.set("./{*}Grid/{*}Type", "RGZERO")
    sicd_con.check("check_pfa_grid_type")
    testing.assert_failures(sicd_con, "PFA grid type must be RGAZIM")


def _bad_kaz1(xmlhelp):
    xmlhelp.set("./{*}PFA/{*}Kaz1", -10000.0)


def _bad_kaz2(xmlhelp):
    xmlhelp.set("./{*}PFA/{*}Kaz2", 10000.0)


@pytest.mark.parametrize(
    "invalidate_kaz",
    [
        {"func": _bad_kaz1, "error": "PFA Kaz1 within half of 1/Grid.Col.SS of KCtr"},
        {"func": _bad_kaz2, "error": "PFA Kaz2 within half of 1/Grid.Col.SS of KCtr"},
    ],
)
def test_check_pfa_spot_kaz_to_grid(sicd_con, invalidate_kaz):
    sicd_con.check("check_pfa_spot_kaz_to_grid")
    assert not sicd_con.failures()

    invalidate_kaz["func"](sicd_con.xmlhelp)
    sicd_con.check("check_pfa_spot_kaz_to_grid")
    testing.assert_failures(sicd_con, invalidate_kaz["error"])


def _bad_krg1(xmlhelp):
    xmlhelp.set("./{*}PFA/{*}Krg1", -10000.0)


def _bad_krg2(xmlhelp):
    xmlhelp.set("./{*}PFA/{*}Krg2", 10000.0)


@pytest.mark.parametrize(
    "invalidate_krg",
    [
        {"func": _bad_krg1, "error": "PFA Krg1 within half of 1/Grid.Row.SS of KCtr"},
        {"func": _bad_krg2, "error": "PFA Krg2 within half of 1/Grid.Row.SS of KCtr"},
    ],
)
def test_check_pfa_krg_to_grid(sicd_con, invalidate_krg):
    sicd_con.check("check_pfa_krg_to_grid")
    assert not sicd_con.failures()

    invalidate_krg["func"](sicd_con.xmlhelp)
    sicd_con.check("check_pfa_krg_to_grid")
    testing.assert_failures(sicd_con, invalidate_krg["error"])


@pytest.mark.parametrize(
    "poly_to_invalidate",
    ("{*}PolarAngPoly", "{*}SpatialFreqSFPoly", "{*}STDeskew/{*}STDSPhasePoly"),
)
def test_pfa_polys(sicd_con, em, poly_to_invalidate):
    assert sicd_con.sicdroot.find("./{*}PFA/{*}STDeskew") is None
    # Add STDSPhasePoly node since example xml does not have it
    sicd_con.sicdroot.find("./{*}PFA").append(
        em.STDeskew(
            em.Applied("true"),
            sksicd.Poly2dType().make_elem("STDSPhasePoly", np.zeros((2, 3))),
        )
    )
    sicd_con.check("check_pfa_polys")
    assert sicd_con.passes()
    _invalidate_1d_poly_coefs(sicd_con.sicdroot.find("./{*}PFA/" + poly_to_invalidate))
    sicd_con.check("check_pfa_polys")
    assert sicd_con.failures()


def test_pfa_stds_kcoa(sicd_con, em):
    # Make sure STDSPhasePoly does not exist, so we don't overwrite it
    assert sicd_con.sicdroot.find("./{*}PFA/{*}STDeskew/{*}STDSPhasePoly") is None

    # Add non-zero STDSPhasePoly node since example xml does not have it
    pfa_node = sicd_con.sicdroot.find("./{*}PFA")
    pfa_node.append(
        em.STDeskew(
            em.Applied(),
            em.STDSPhasePoly(),
        )
    )
    sicd_con.xmlhelp.set("./{*}PFA/{*}STDeskew/{*}Applied", True)
    sicd_con.xmlhelp.set(
        "./{*}PFA/{*}STDeskew/{*}STDSPhasePoly", [[1.0, 0.0], [1.0, 0.0]]
    )

    # Use non-zero column DeltaKCOAPoly to force failure
    sicd_con.xmlhelp.set(
        "./{*}Grid/{*}Col/{*}DeltaKCOAPoly", np.array([[1, 0], [1, 0]])
    )

    sicd_con.check("check_pfa_stds_kcoa")
    details = sicd_con.failures()["check_pfa_stds_kcoa"]["details"]
    assert np.all([det["passed"] is False for det in details])

    # Use non-zero row DeltaKCOAPoly to force failure and applied=False for the other path
    sicd_con.xmlhelp.set(
        "./{*}Grid/{*}Row/{*}DeltaKCOAPoly", np.array([[1, 0], [1, 0]])
    )
    sicd_con.xmlhelp.set("./{*}PFA/{*}STDeskew/{*}Applied", "false")

    sicd_con.check("check_pfa_stds_kcoa")
    details = sicd_con.failures()["check_pfa_stds_kcoa"]["details"]
    assert np.all([det["passed"] is False for det in details])


def _min_proc_lt_min_coll(xmlhelp):
    min_coll = xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Min")
    xmlhelp.set("./{*}ImageFormation/{*}TxFrequencyProc/{*}MinProc", min_coll / 2)


def _max_proc_gt_max_coll(xmlhelp):
    max_coll = xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Max")
    xmlhelp.set("./{*}ImageFormation/{*}TxFrequencyProc/{*}MaxProc", max_coll * 2)


def _max_coll_lt_min_coll(xmlhelp):
    min_coll = xmlhelp.load("./{*}RadarCollection/{*}TxFrequency/{*}Min")
    xmlhelp.set("./{*}RadarCollection/{*}TxFrequency/{*}Max", min_coll / 2)


def _max_proc_lt_min_proc(xmlhelp):
    min_proc = xmlhelp.load("./{*}ImageFormation/{*}TxFrequencyProc/{*}MinProc")
    xmlhelp.set("./{*}ImageFormation/{*}TxFrequencyProc/{*}MaxProc", min_proc / 2)


def _zero_min(xmlhelp):
    xmlhelp.set("./{*}RadarCollection/{*}TxFrequency/{*}Min", 0.0)


@pytest.mark.parametrize(
    "invalidate_proc_freq",
    [
        {
            "func": _min_proc_lt_min_coll,
            "error": "Min processed frequency >= min collected frequency",
        },
        {
            "func": _max_proc_gt_max_coll,
            "error": "Max processed frequency <= max collected frequency",
        },
        {
            "func": _max_coll_lt_min_coll,
            "error": "Max collected frequency > min collected",
        },
        {
            "func": _max_proc_lt_min_proc,
            "error": "Max processed frequency > min processed",
        },
        {"func": _zero_min, "error": "Min collected frequency > 0.0"},
    ],
)
def test_check_proc_freq(sicd_con, invalidate_proc_freq):
    invalidate_proc_freq["func"](sicd_con.xmlhelp)
    sicd_con.check("check_proc_freq")
    testing.assert_failures(sicd_con, invalidate_proc_freq["error"])


@pytest.fixture
def sicd_con_bad_inca(sicd_con, em):
    # Add RMA/INCA nodes since example xml does not have them
    assert sicd_con.sicdroot.find("./{*}RMA/{*}INCA") is None
    sicd_con.sicdroot.append(
        em.RMA(
            em.RMAlgoType("RG_DOP"),
            em.ImageType("INCA"),
            em.INCA(
                sksicd.PolyType().make_elem("TimeCAPoly", np.ones(4)),
                em.R_CA_SCP("10000.0"),
                em.FreqZero("0.0"),
                sksicd.Poly2dType().make_elem("DRateSFPoly", np.ones((4, 3))),
                sksicd.Poly2dType().make_elem("DopCentroidPoly", np.ones((5, 4))),
                em.DopCentroidCOA("false"),
            ),
        )
    )
    sicd_con.check("check_inca")
    assert sicd_con.failures()
    return sicd_con


@pytest.mark.parametrize(
    "poly_to_invalidate", ("{*}TimeCAPoly", "{*}DRateSFPoly", "{*}DopCentroidPoly")
)
def test_check_rma_inca_polys(sicd_con_bad_inca, poly_to_invalidate):
    sicd_con = sicd_con_bad_inca
    sicd_con.check("check_rma_inca_polys")
    assert sicd_con.passes()
    _invalidate_1d_poly_coefs(
        sicd_con.sicdroot.find("./{*}RMA/{*}INCA/" + poly_to_invalidate)
    )
    sicd_con.check("check_rma_inca_polys")
    assert sicd_con.failures()


def test_segment_bounds(sicd_con, em):
    rca_plane = sicd_con.sicdroot.find("./{*}RadarCollection/{*}Area/{*}Plane")
    assert rca_plane.find("./{*}SegmentList") is None

    first_line = sicd_con.xmlhelp.load(
        "./{*}RadarCollection/{*}Area/{*}Plane/{*}XDir/{*}FirstLine"
    )
    first_sample = sicd_con.xmlhelp.load(
        "./{*}RadarCollection/{*}Area/{*}Plane/{*}YDir/{*}FirstSample"
    )
    num_lines = sicd_con.xmlhelp.load(
        "./{*}RadarCollection/{*}Area/{*}Plane/{*}XDir/{*}NumLines"
    )
    num_samples = sicd_con.xmlhelp.load(
        "./{*}RadarCollection/{*}Area/{*}Plane/{*}YDir/{*}NumSamples"
    )

    rca_plane.append(
        em.SegmentList(
            em.Segment(
                em.StartLine(str(first_line)),
                em.StartSample(str(first_sample)),
                em.EndLine(str(int(first_line + num_lines // 2 - 1))),
                em.EndSample(str(int(first_sample + num_samples // 2 - 1))),
            ),
            em.Segment(
                em.StartLine(str(int(first_line + num_lines // 2))),
                em.StartSample(str(int(first_sample + num_samples // 2))),
                em.EndLine(str(int(first_line + num_lines - 1))),
                em.EndSample(str(int(first_sample + num_samples - 1))),
            ),
        )
    )
    sicd_con.check("check_segmentlist_bounds")
    assert sicd_con.passes()

    rca_plane.find("./{*}XDir/{*}NumLines").text = str(num_lines - 10)
    sicd_con.check("check_segmentlist_bounds")
    testing.assert_failures(
        sicd_con, "All segments within the segment_list are bounded"
    )


def test_segment_identifier(sicd_con, em):
    imform = sicd_con.sicdroot.find("./{*}ImageFormation")
    assert imform.find("./{*}SegmentIdentifier") is None
    rca_plane = sicd_con.sicdroot.find("./{*}RadarCollection/{*}Area/{*}Plane")
    assert rca_plane.find("./{*}SegmentList") is None

    rca_plane.append(
        em.SegmentList(
            em.Segment(em.Identifier("SegmentID 1")),
            em.Segment(em.Identifier("SegmentID 2")),
        )
    )
    sicd_con.check("check_segment_identifier")
    testing.assert_failures(sicd_con, "SegmentIdentifier is included")

    segid = em.SegmentIdentifier("not found ID")
    imform.append(segid)
    sicd_con.check("check_segment_identifier")
    testing.assert_failures(sicd_con, "SegmentList has SegmentIdentifier")
    segid.text = "SegmentID 2"
    sicd_con.check("check_segment_identifier")
    assert sicd_con.passes()


def test_check_segment_unique_ids(sicd_con, em):
    rca_plane = sicd_con.sicdroot.find("./{*}RadarCollection/{*}Area/{*}Plane")
    rca_plane.append(
        em.SegmentList(
            em.Segment(em.Identifier("seg1")),
            em.Segment(em.Identifier("seg1")),
        )
    )
    sicd_con.check("check_segment_unique_ids")
    testing.assert_failures(sicd_con, "SegmentList segments have unique identifiers")


def test_check_image_formation_timeline(sicd_con):
    sicd_con.xmlhelp.set(
        "./{*}ImageFormation/{*}TStartProc",
        sicd_con.xmlhelp.load("./{*}ImageFormation/{*}TEndProc") + 1,
    )
    sicd_con.check("check_image_formation_timeline")
    assert sicd_con.failures()


def test_check_rcvapcindex(sicd_con):
    # Invalid APCIndex
    sicd_con.xmlhelp.set(
        "./{*}RadarCollection/{*}RcvChannels/{*}ChanParameters/{*}RcvAPCIndex",
        100,
    )
    sicd_con.check("check_rcvapcindex")
    assert sicd_con.failures()

    # No APCIndex with APCPolys is OK
    rcvapcindex = sicd_con.sicdroot.find(
        "./{*}RadarCollection/{*}RcvChannels/{*}ChanParameters/{*}RcvAPCIndex"
    )
    rcvapcindex.getparent().remove(rcvapcindex)
    sicd_con.check("check_rcvapcindex")
    assert not sicd_con.failures()


def test_check_rcvapcindex_nopolys(sicd_con):
    # APCIndex with no APCPolys
    rcvapcnode = sicd_con.sicdroot.find("./{*}Position/{*}RcvAPC")
    rcvapcnode.getparent().remove(rcvapcnode)
    sicd_con.check("check_rcvapcindex")
    assert sicd_con.failures()


def test_check_chanindex(sicd_con):
    sicd_con.check("check_chanindex")
    assert sicd_con.passes()

    rcv_channels = sicd_con.sicdroot.findall("./{*}ImageFormation/{*}RcvChanProc")
    # Change the first ChanIndex
    rcv_channels[0].find("./{*}ChanIndex").text = "99"
    sicd_con.check("check_chanindex")
    testing.assert_failures(sicd_con, "ChanIndex 99 must reference a ChanParameters")


def test_check_collection_duration(sicd_con):
    sicd_con.xmlhelp.set("./{*}Timeline/{*}CollectDuration", 0.0)
    sicd_con.check("check_collection_duration")
    testing.assert_failures(sicd_con, "CollectionDuration > zero")


def test_check_nitf_imseg(example_sicd_file, tmp_path):
    example_sicd_file.seek(0)
    with sksicd.NitfReader(example_sicd_file) as r:
        sicd_meta = r.metadata

    # Use SICD v1.4.0 FFDD Example 2 parameters to force segmentation
    sicd_meta.xmltree.find("{*}ImageData/{*}NumRows").text = "30000"
    sicd_meta.xmltree.find("{*}ImageData/{*}NumCols").text = "90000"
    sicd_meta.xmltree.find("{*}ImageData/{*}PixelType").text = "RE32F_IM32F"
    assert sksicd.image_segment_sizing_calculations(sicd_meta.xmltree)[0] == 3
    tmp_sicd = tmp_path / "forced_segmentation.sicd"
    with open(tmp_sicd, "wb") as f, sksicd.NitfWriter(f, sicd_meta):
        pass  # don't currently care about the pixels

    with tmp_sicd.open("rb") as f:
        sicd_con = SicdConsistency.from_file(f)
    sicd_con.check("check_nitf_imseg")
    assert sicd_con.passes() and not sicd_con.failures()

    # monkey with the IID1s
    with tmp_sicd.open("rb+") as fd:
        ntf = jbpy.Jbp()
        ntf.load(fd)
        for imseg in ntf["ImageSegments"]:
            imseg["subheader"]["IID1"].value = "SICD000"
            imseg["subheader"]["IID1"].dump(fd, seek_first=True)
    with tmp_sicd.open("rb") as f:
        sicd_con = SicdConsistency.from_file(f)
    sicd_con.check("check_nitf_imseg")
    testing.assert_failures(sicd_con, "Sequential IID1")


def test_check_error_components_posvel_stddev(sicd_con, em):
    p2 = em.P2("0.2")
    assert sicd_con.sicdroot.find("./{*}ErrorStatistics") is None
    sicd_con.sicdroot.append(
        em.ErrorStatistics(
            em.Components(
                em.PosVelErr(
                    em.P1("0.1"),
                    p2,
                    em.P3("0.3"),
                    em.V1("0.4"),
                    em.V2("0.5"),
                    em.V3("0.6"),
                )
            )
        )
    )
    sicd_con.check("check_error_components_posvel_stddev")
    assert sicd_con.passes()
    p2.text = "-1.0"
    sicd_con.check("check_error_components_posvel_stddev")
    testing.assert_failures(sicd_con, "PosVelErr P2 >= 0.0")


def test_check_error_components_posvel_corr(sicd_con, em):
    p1v1 = em.P1V1("0.17")
    assert sicd_con.sicdroot.find("./{*}ErrorStatistics") is None
    sicd_con.sicdroot.append(
        em.ErrorStatistics(
            em.Components(
                em.PosVelErr(
                    em.CorrCoefs(
                        em.P1P2("0.12"),
                        em.P1P3("0.13"),
                        p1v1,
                        em.P1V2("0.18"),
                        em.P1V3("0.19"),
                        em.P2P3("0.23"),
                        em.P2V1("0.27"),
                        em.P2V2("-0.28"),
                        em.P2V3("-0.29"),
                        em.P3V1("-0.37"),
                        em.P3V2("-0.38"),
                        em.P3V3("-0.39"),
                        em.V1V2("-0.78"),
                        em.V1V3("-0.79"),
                        em.V2V3("-0.89"),
                    )
                )
            )
        )
    )
    sicd_con.check("check_error_components_posvel_corr")
    assert sicd_con.passes()
    p1v1.text = "-1.1"
    sicd_con.check("check_error_components_posvel_corr")
    testing.assert_failures(sicd_con, "CorrCoefs P1V1 <= 1.0")


def _bad_composite_rg(xmlhelp):
    xmlhelp.set("./{*}ErrorStatistics/{*}CompositeSCP/{*}Rg", -1.0)


def _bad_composite_az(xmlhelp):
    xmlhelp.set("./{*}ErrorStatistics/{*}CompositeSCP/{*}Az", -1.0)


def _bad_composite_rgaz(xmlhelp):
    xmlhelp.set("./{*}ErrorStatistics/{*}CompositeSCP/{*}RgAz", 2.0)


@pytest.mark.parametrize(
    "invalidate_composite_scp",
    [
        {"func": _bad_composite_rg, "error": "CompositeSCP Rg >= 0.0"},
        {"func": _bad_composite_az, "error": "CompositeSCP Az >= 0.0"},
        {"func": _bad_composite_rgaz, "error": "CompositeSCP RgAz <= 1.0"},
    ],
)
def test_check_error_composite(sicd_con, em, invalidate_composite_scp):
    assert sicd_con.sicdroot.find("./{*}ErrorStatistics") is None
    sicd_con.sicdroot.append(
        em.ErrorStatistics(
            em.CompositeSCP(
                em.Rg("1.0"),
                em.Az("2.0"),
                em.RgAz("0.0"),
            )
        )
    )
    sicd_con.check("check_error_composite")
    assert sicd_con.passes()

    invalidate_composite_scp["func"](sicd_con.xmlhelp)
    sicd_con.check("check_error_composite")
    testing.assert_failures(sicd_con, invalidate_composite_scp["error"])


def test_check_error_radarsensor_rangebias(sicd_con, em):
    assert sicd_con.sicdroot.find("./{*}ErrorStatistics") is None
    sicd_con.sicdroot.append(
        em.ErrorStatistics(
            em.Components(
                em.RadarSensor(
                    em.RangeBias("1.0"),
                )
            )
        )
    )
    sicd_con.check("check_error_radarsensor_rangebias")
    assert sicd_con.passes()

    sicd_con.xmlhelp.set(
        "./{*}ErrorStatistics/{*}Components/{*}RadarSensor/{*}RangeBias", -1.0
    )
    sicd_con.check("check_error_radarsensor_rangebias")
    testing.assert_failures(sicd_con, "RangeBias >= 0.0")


def test_check_txsequence_waveform_index(sicd_con, em):
    assert sicd_con.sicdroot.find("./{*}RadarCollection/{*}TxSequence") is None
    rc_node = sicd_con.sicdroot.find("./{*}RadarCollection")
    rc_node.append(em.TxSequence(em.TxStep(em.WFIndex("1"))))
    sicd_con.check("check_txsequence_waveform_index")
    assert sicd_con.passes()

    sicd_con.xmlhelp.set(
        "./{*}RadarCollection/{*}TxSequence/{*}TxStep/{*}WFIndex", "100"
    )
    sicd_con.check("check_txsequence_waveform_index")
    testing.assert_failures(sicd_con, "WFIndex 100 must reference a WFParameters")


def test_smart_open_http(example_sicd):
    with tests.utils.static_http_server(example_sicd.parent) as server_url:
        assert not main([f"{server_url}/{example_sicd.name}"])


def test_smart_open_contract(example_sicd, monkeypatch):
    mock_open = unittest.mock.MagicMock(side_effect=tests.utils.simple_open_read)
    monkeypatch.setattr(sarkit.verification._sicdcheck, "open", mock_open)
    assert not main([str(example_sicd)])
    mock_open.assert_called_once_with(str(example_sicd), "rb")
