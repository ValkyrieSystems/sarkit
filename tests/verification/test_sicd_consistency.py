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

    def test_grid_unit_vectors(self, direction, sicd_con):
        z_val = sicd_con.xmlhelp.load(
            f"./{{*}}Grid/{{*}}{direction}/{{*}}UVectECF/{{*}}Z"
        )
        sicd_con.xmlhelp.set(
            f"./{{*}}Grid/{{*}}{direction}/{{*}}UVectECF/{{*}}Z", -100.0 * z_val
        )
        sicd_con.check(f"check_grid_unit_vector_{direction.lower()}")
        assert sicd_con.failures()


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


def test_smart_open_http(example_sicd):
    with tests.utils.static_http_server(example_sicd.parent) as server_url:
        assert not main([f"{server_url}/{example_sicd.name}"])


def test_smart_open_contract(example_sicd, monkeypatch):
    mock_open = unittest.mock.MagicMock(side_effect=tests.utils.simple_open_read)
    monkeypatch.setattr(sarkit.verification._sicdcheck, "open", mock_open)
    assert not main([str(example_sicd)])
    mock_open.assert_called_once_with(str(example_sicd), "rb")
