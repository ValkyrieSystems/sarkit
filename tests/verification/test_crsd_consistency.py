import itertools
import os
import pathlib
import re
import types
import unittest.mock

import lxml.builder
import numpy as np
import pytest
from lxml import etree

import sarkit.crsd as skcrsd
import sarkit.verification._crsdcheck
import sarkit.wgs84
import tests.utils
from sarkit.verification._crsd_consistency import CrsdConsistency
from sarkit.verification._crsdcheck import main

from . import testing

DATAPATH = pathlib.Path(__file__).parents[2] / "data"

good_crsd_xml_path = DATAPATH / "example-crsd-1.0.xml"


def _repack_support_arrays(crsd_etree):
    offset = 0
    for array in crsd_etree.findall("{*}Data/{*}Support/{*}SupportArray"):
        array.find("{*}ArrayByteOffset").text = str(offset)
        offset += (
            int(array.findtext("{*}NumRows"))
            * int(array.findtext("{*}NumCols"))
            * int(array.findtext("{*}BytesPerElement"))
        )
    return offset


def _make_elem(name, text="", **attrs):
    retval = etree.Element(name, **attrs)
    retval.text = text
    return retval


def _remove(root, pattern):
    if (elem := root.find(pattern)) is not None:
        elem.getparent().remove(elem)
    else:
        print(f"Cannot find {pattern=}")


def assert_failures(crsd_con, pattern):
    testing.assert_failures(crsd_con, pattern)


def assert_not_failures(crsd_con, pattern):
    pattern = re.compile(pattern)
    failure_details = itertools.chain(
        *[x["details"] for x in crsd_con.failures(omit_passed_sub=True).values()]
    )
    failure_messages = [x["details"] for x in failure_details]

    # this construction can help improve the error message for determining why pattern is not present
    assert crsd_con.all() and not any(list(map(pattern.search, failure_messages)))


@pytest.fixture(scope="session")
def example_crsdsar_file(example_crsdsar):
    assert not main([str(example_crsdsar), "-v"])
    with example_crsdsar.open("rb") as f:
        yield f


@pytest.fixture(scope="session")
def example_crsdtx_file(example_crsdtx):
    assert not main([str(example_crsdtx), "-vvv"])
    with example_crsdtx.open("rb") as f:
        yield f


@pytest.fixture(scope="session")
def example_crsdrcv_file(example_crsdrcv):
    assert not main([str(example_crsdrcv), "-vvv"])
    with example_crsdrcv.open("rb") as f:
        yield f


@pytest.fixture(scope="session")
def example_crsdrcvcompressed_file(tmp_path_factory, example_crsdrcv_file):
    example_crsdrcv_file.seek(0)
    with skcrsd.Reader(example_crsdrcv_file) as r:
        channel_id = r.metadata.xmltree.findtext(
            "{*}Channel/{*}Parameters/{*}Identifier"
        )
        pvps = r.read_pvps(channel_id)
        support_arrays = {
            said.text: r.read_support_array(said.text, masked=False)
            for said in r.metadata.xmltree.findall("{*}SupportArray/*/{*}Identifier")
        }

    new_meta = r.metadata
    data_rcv = new_meta.xmltree.find("{*}Data/{*}Receive")
    assert data_rcv.find("{*}Data/{*}SignalCompression") is None
    ns = etree.QName(new_meta.xmltree.getroot()).namespace
    em = lxml.builder.ElementMaker(namespace=ns, nsmap={None: ns})
    compressed_data = b"ultra-compressed"
    data_rcv.find("{*}NumCRSDChannels").addnext(
        em.SignalCompression(
            em.Identifier("is constant!"),
            em.CompressedSignalSize(str(len(compressed_data))),
        )
    )
    tmp_crsd = tmp_path_factory.mktemp("data") / "faux-compressed.crsd"
    with tmp_crsd.open("wb") as f, skcrsd.Writer(f, new_meta) as w:
        w.write_pvp(channel_id, pvps)
        for sa_id, arr in support_arrays.items():
            w.write_support_array(sa_id, arr)
        w.write_signal_compressed(np.frombuffer(compressed_data, np.uint8))
    assert not main([str(tmp_crsd)])
    with tmp_crsd.open("rb") as f:
        yield f


@pytest.fixture(scope="session", params=("sar", "tx", "rcv"))
def example_crsd_file(request):
    param = request.param
    yield request.getfixturevalue(f"example_crsd{param}_file")


@pytest.fixture(scope="module")
def good_xml():
    return etree.parse(good_crsd_xml_path)


@pytest.fixture
def good_xml_root(good_xml):
    return copy_xml(good_xml)


@pytest.fixture
def crsd_con(example_crsd_file):
    return CrsdConsistency.from_file(example_crsd_file, thorough=True)


def copy_xml(elem):
    return etree.fromstring(etree.tostring(elem))


def test_from_file_crsd(example_crsd_file):
    crsdcon = CrsdConsistency.from_file(example_crsd_file)
    crsdcon.check()
    assert not crsdcon.failures()


def test_from_file_xml():
    crsdcon = CrsdConsistency.from_file(str(good_crsd_xml_path))
    assert isinstance(crsdcon, CrsdConsistency)
    crsdcon.check()
    assert not crsdcon.failures()


@pytest.mark.parametrize("xml_file", (DATAPATH / "syntax_only/crsd").glob("*.xml"))
def test_smoketest(xml_file):
    main([str(xml_file)])


def test_main_with_ignore(good_xml_root, tmp_path):
    good_xml_root.find("./{*}SceneCoordinates/{*}EarthModel").text += "1"
    slightly_bad_xml = os.path.join(tmp_path, "slightly_bad.xml")
    etree.ElementTree(good_xml_root).write(str(slightly_bad_xml))
    assert main([slightly_bad_xml])
    assert not main([slightly_bad_xml, "--ignore", "check_against_schema"])


def test_main_schema_args(crsd_con):
    good_schema = crsd_con.schema

    assert not main(
        [
            str(good_crsd_xml_path),
            "--schema",
            str(good_schema),
        ]
    )  # pass with actual schema

    assert main(
        [
            str(good_crsd_xml_path),
            "--schema",
            str(good_crsd_xml_path),
        ]
    )  # fails with bogus schema


def test_thorough(example_crsd_file):
    con = CrsdConsistency.from_file(example_crsd_file, thorough=True)
    con.check()
    num_skips_thorough = len(con.skips(include_partial=True))

    con = CrsdConsistency.from_file(example_crsd_file)
    con.check()
    num_skips_default = len(con.skips(include_partial=True))
    assert num_skips_thorough < num_skips_default


def test_header_filetype(crsd_con):
    crsd_con.file_type_header += "MAKEBAD"

    crsd_con.check("check_file_type_header")
    assert crsd_con.failures()


def test_header_kvp_missing_sb_size(crsd_con):
    del crsd_con.kvp_list["SUPPORT_BLOCK_SIZE"]

    crsd_con.check("check_header_kvp_list")
    assert crsd_con.failures()


def test_header_kvp_missing_pxpb_size(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        del crsd_con.kvp_list["PPP_BLOCK_SIZE"]
    else:
        del crsd_con.kvp_list["PVP_BLOCK_SIZE"]

    crsd_con.check("check_header_kvp_list")
    assert crsd_con.failures()


def test_header_bad_class(crsd_con):
    crsd_con.kvp_list["CLASSIFICATION"] = "FAKE_CLASSIFICATION"

    crsd_con.check("check_classification_and_release_info")
    assert crsd_con.failures()


def test_header_bad_release(crsd_con):
    crsd_con.kvp_list["RELEASE_INFO"] = "FAKE_RELEASE"

    crsd_con.check("check_classification_and_release_info")
    assert crsd_con.failures()


# only applicable to CRSDsar
def test_sensorname_monostatic(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find("{*}TransmitInfo/{*}SensorName").text += "_bad"
    crsd_con.check("check_sensorname")
    assert_failures(crsd_con, "Transmit and receive sensor names are the same")


# only applicable to CRSDsar
def test_eventname_monostatic(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find("{*}TransmitInfo/{*}EventName").text += "_bad"
    crsd_con.check("check_eventname")
    assert_failures(crsd_con, "Transmit and receive event names are the same")


# only applicable to CRSDsar
def test_sensorname_eventname_bistatic(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find("{*}SARInfo/{*}CollectType").text = "BISTATIC"
    crsd_con.crsdroot.find("{*}TransmitInfo/{*}SensorName").text += "_other"
    crsd_con.crsdroot.find("{*}TransmitInfo/{*}EventName").text += "_other"
    crsd_con.check("check_sensorname")
    # We're looking for skips, so we have to check all instead of passes
    assert crsd_con.all() and not crsd_con.failures()
    crsd_con.check("check_eventname")
    assert crsd_con.all() and not crsd_con.failures()


@pytest.mark.parametrize("name", ["TxTime1", "TxTime2"])
def test_global_txtime12(crsd_con, name):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find(f"{{*}}Global/{{*}}Transmit/{{*}}{name}").text = "-1.0"
    crsd_con.check("check_global_txtime12")
    assert_failures(crsd_con, f"{name} matches")


@pytest.mark.parametrize("name", ["FxMin", "FxMax"])
def test_global_fxminmax(crsd_con, name):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find(f"{{*}}Global/{{*}}Transmit/{{*}}{name}").text = "0.0"
    crsd_con.check("check_global_fxminmax")
    assert_failures(crsd_con, f"{name} matches")


@pytest.mark.parametrize("name", ["RcvStartTime1", "RcvStartTime2"])
def test_global_rcvstarttime12(crsd_con, name):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    crsd_con.crsdroot.find(f"{{*}}Global/{{*}}Receive/{{*}}{name}").text = "0.0"
    crsd_con.check("check_global_rcvstarttime12")
    assert_failures(crsd_con, f"{name} matches")


@pytest.mark.parametrize("name", ["FrcvMin", "FrcvMax"])
def test_global_frcvminmax(crsd_con, name):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    crsd_con.crsdroot.find(f"{{*}}Global/{{*}}Receive/{{*}}{name}").text = "0.0"
    crsd_con.check("check_global_frcvminmax")
    assert_failures(crsd_con, f"{name} matches")


# only applicable to CRSDsar
def test_ref_tx_id_sar(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    ref_chan_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    crsd_con.crsdroot.find(
        f'{{*}}Channel/{{*}}Parameters[{{*}}Identifier="{ref_chan_id}"]/{{*}}SARImage/{{*}}TxId'
    ).text += "_bad"
    crsd_con.check("check_reftxid")
    assert_failures(crsd_con, "reference channel")


def test_xm_id_lfm(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find("{*}TxSequence/{*}TxWFType").text = "LFM"
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}RefPulseIndex").addnext(
        _make_elem(f"{{{elem_ns}}}XMId", "Some ID")
    )
    crsd_con.check("check_xm_id", allow_prefix=True)
    assert_failures(crsd_con, "XMId is not present")


def test_xm_id_notlfm_missing_xmid(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    # test XML has TxWFType == LFM w XM
    assert crsd_con.crsdroot.find("{*}TxSequence/{*}TxWFType").text == "LFM w XM"
    _remove(crsd_con.crsdroot, "{*}TxSequence/{*}Parameters/{*}XMId")
    crsd_con.check("check_xm_id", allow_prefix=True)
    assert_failures(crsd_con, "XMId is present")


def test_ref_pulse_index(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find(
        "{*}TxSequence/{*}Parameters/{*}RefPulseIndex"
    ).text += "99999999"
    crsd_con.check("check_ref_pulse_index", allow_prefix=True)
    assert_failures(crsd_con, "extant pulse")


def test_txrefpoint(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    zcoord = crsd_con.crsdroot.find(
        "{*}TxSequence/{*}Parameters/{*}TxRefPoint/{*}ECF/{*}Z"
    )
    zcoord.text = str(float(zcoord.text) + 10)
    crsd_con.check("check_txrefpoint", allow_prefix=True)
    assert_failures(crsd_con, "IAC maps to ECF")


def test_ref_rad_intensity(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    ref_rad_int = crsd_con.crsdroot.find(
        "{*}TxSequence/{*}Parameters/{*}TxRefRadIntensity"
    )
    ref_rad_int.text = str(float(ref_rad_int.text) + 1)
    crsd_con.check("check_ref_rad_intensity", allow_prefix=True)
    assert_failures(crsd_con, "matches the PPP")


def test_txtime1(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    txtime1 = crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}TxTime1")
    txtime1.text = str(float(txtime1.text) + 1)
    crsd_con.check("check_txtime12", allow_prefix=True)
    assert_failures(crsd_con, "matches the first PPP")


def test_txtime2(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    txtime2 = crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}TxTime2")
    txtime2.text = str(float(txtime2.text) + 1)
    crsd_con.check("check_txtime12", allow_prefix=True)
    assert_failures(crsd_con, "matches the last PPP")


def test_txmt_min(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    txmt_min = crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}TXmtMin")
    txmt_min.text = str(float(txmt_min.text) + 1)
    crsd_con.check("check_txmt_minmax", allow_prefix=True)
    assert_failures(crsd_con, "matches minimum")


def test_txmt_max(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    txmt_max = crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}TXmtMax")
    txmt_max.text = str(float(txmt_max.text) + 1)
    crsd_con.check("check_txmt_minmax", allow_prefix=True)
    assert_failures(crsd_con, "matches maximum")


def test_fxc(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    fxc = crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}FxC")
    fxc.text = str(float(fxc.text) + 100e6)
    crsd_con.check("check_tx_frequency_band", allow_prefix=True)
    assert_failures(crsd_con, "FxC matches the PPPs")


def test_fxbw(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    fxbw = crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}FxBW")
    fxbw.text = str(float(fxbw.text) + 100e6)
    crsd_con.check("check_tx_frequency_band", allow_prefix=True)
    assert_failures(crsd_con, "FxBW matches the PPPs")


def test_fxbw_not_fixed(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}FxBWFixed").text = "false"
    crsd_con.check("check_tx_frequency_band", allow_prefix=True)
    assert_failures(crsd_con, "does change")


def test_fxbw_fixed_fx1(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppps = crsd_con._get_sequence_ppps(sequence_id)
    ppps["FX1"][0] *= 2
    crsd_con.check("check_tx_frequency_band", allow_prefix=True)
    assert_failures(crsd_con, "FX1 does not change")


def test_fxbw_fixed_fx2(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppps = crsd_con._get_sequence_ppps(sequence_id)
    ppps["FX2"][0] *= 2
    crsd_con.check("check_tx_frequency_band", allow_prefix=True)
    assert_failures(crsd_con, "FX2 does not change")


def test_xmindex_ppp_presence(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find("{*}TxSequence/{*}TxWFType").text = "LFM"
    crsd_con.check("check_xmindex_ppp_presence")
    assert_failures(crsd_con, "is not LFM")


def test_xmindex_ppp_presence_not_present(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    _remove(crsd_con.crsdroot, "{*}PPP/{*}XMIndex")
    crsd_con.check("check_xmindex_ppp_presence")
    assert_failures(crsd_con, "is LFM")


def test_xmindex_value(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["XMIndex"][0] = -3
    crsd_con.check("check_xmindex_value", allow_prefix=True)
    assert_failures(crsd_con, ">= 0")


def test_xmindex_value_too_big(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["XMIndex"][0] = 2**62
    crsd_con.check("check_xmindex_value", allow_prefix=True)
    assert_failures(crsd_con, "< NumXM")


def test_fx12_ppp(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["FX1"][0] = ppp["FX2"][0]
    crsd_con.check("check_fx12_ppp", allow_prefix=True)
    assert_failures(crsd_con, "FX1 < FX2")


def test_txposveltime_matched(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["TxPos"][:] = ppp["TxPos"][::-1]
    crsd_con.check("check_txposveltime", allow_prefix=True)
    assert_failures(crsd_con, "matched set")


def test_txposveltime_sized(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["TxPos"][:] *= 2
    ppp["TxVel"][:] *= 2  # to keep them a matched set
    crsd_con.check("check_txposveltime", allow_prefix=True)
    assert_failures(crsd_con, "near Earth")


def test_txtime_not_increasing(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["TxTime"][1] = ppp["TxTime"][0]
    crsd_con.check("check_txtime", allow_prefix=True)
    assert_failures(crsd_con, "TxTime is strictly increasing")


def test_txtime_frac_negative(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["TxTime"]["Frac"][1] = -0.5
    crsd_con.check("check_txtime", allow_prefix=True)
    assert_failures(crsd_con, "TxTime.Frac is non-negative")


def test_txtime_frac_too_big(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["TxTime"]["Frac"][1] = 1.5
    crsd_con.check("check_txtime", allow_prefix=True)
    assert_failures(crsd_con, "TxTime.Frac is less than 1")


def test_txoverlap(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["TXmt"][0] = 1.0
    crsd_con.check("check_txoverlap", allow_prefix=True)
    assert_failures(crsd_con, "overlap")


def test_phix0_intfrac_negative(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["PhiX0"]["Frac"][1] = -0.5
    crsd_con.check("check_phix0_intfrac", allow_prefix=True)
    assert_failures(crsd_con, "PhiX0.Frac is non-negative")


def test_phix0_intfrac_too_big(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["PhiX0"]["Frac"][1] = 1.5
    crsd_con.check("check_phix0_intfrac", allow_prefix=True)
    assert_failures(crsd_con, "PhiX0.Frac is less than 1")


@pytest.mark.parametrize("axis", ["X", "Y"])
def test_tx_acxy_not_unit(crsd_con, axis):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp[f"TxAC{axis}"][0] *= 3
    crsd_con.check("check_tx_acxy", allow_prefix=True)
    assert_failures(crsd_con, f"TxAC{axis} is unit length")


def test_txacxy_not_ortho(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["TxACX"][0] = ppp["TxACY"][0]
    crsd_con.check("check_tx_acxy", allow_prefix=True)
    assert_failures(crsd_con, "are orthogonal")


def test_txeb(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["TxEB"][0] = [0.8, 0.8]
    crsd_con.check("check_txeb", allow_prefix=True)
    assert_failures(crsd_con, "unit length")


def test_fxresponseindex_negative(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["FxResponseIndex"][0] = -1
    crsd_con.check("check_fxresponseindex", allow_prefix=True)
    assert_failures(crsd_con, "non-negative")


def test_fxresponseindex_too_big(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["FxResponseIndex"][0] = 2**62  # this _should_ be big enough
    crsd_con.check("check_fxresponseindex", allow_prefix=True)
    assert_failures(crsd_con, "less than NumFXR")


def test_fxrate_xm_zero(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find("{*}TxSequence/{*}TxWFType").text = "XM"
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["FxRate"][:] = 0
    crsd_con.check("check_fxrate", allow_prefix=True)
    assert crsd_con.passes() and not crsd_con.failures()


def test_fxrate_xm_nonzero(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find("{*}TxSequence/{*}TxWFType").text = "XM"
    crsd_con.check("check_fxrate", allow_prefix=True)
    assert_failures(crsd_con, "is zero")


def test_fxfreq0(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    sequence_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
    ppp = crsd_con._get_sequence_ppps(sequence_id)
    ppp["FxFreq0"][3] = 0
    crsd_con.check("check_fxfreq0", allow_prefix=True)
    assert_failures(crsd_con, "is positive")


def test_xm_pulse_length_too_short(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}TXmtMin").text = "1e-6"
    crsd_con.check("check_xm_pulse_length", allow_prefix=True)
    assert_failures(crsd_con, "500 XM samples")


def test_xm_pulse_length_too_long(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    xm_id = crsd_con.crsdroot.findtext("{*}TxSequence/{*}Parameters/{*}XMId")
    crsd_con.crsdroot.find(
        f'{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId="{xm_id}"]/{{*}}NumCols'
    ).text = "2000"
    crsd_con.check("check_xm_pulse_length", allow_prefix=True)
    assert_failures(crsd_con, "pulse fits")


def test_refvectorindex_notexist(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    crsd_con.crsdroot.find("{*}Channel/{*}Parameters/{*}RefVectorIndex").text += "999"
    crsd_con.check("check_refvectorindex", allow_prefix=True)
    assert_failures(crsd_con, "extant vector")


def test_refvectorindex_notnormal(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["SIGNAL"] = 3
    crsd_con.check("check_refvectorindex", allow_prefix=True)
    assert_failures(crsd_con, "normal signal")


# only applicable to CRSDsar
def test_refvectorindex_sar_notx(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["TxPulseIndex"] = -1
    crsd_con.check("check_refvectorindex", allow_prefix=True)
    assert_failures(crsd_con, "with a transmit pulse")


@pytest.fixture
def ant_patched_crsd_con(crsd_con, monkeypatch):
    def func(*, data=(1.0, 0.0), data_filler=None):
        gp_dtype = np.dtype([("Gain", np.float32), ("Phase", np.float32)])
        data = np.array(data, dtype=gp_dtype)

        def dummy(identifier):
            data_node = crsd_con.crsdroot.find(
                f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{identifier}']"
            )
            num_rows = int(data_node.findtext("{*}NumRows"))
            num_cols = int(data_node.findtext("{*}NumCols"))
            retval = np.zeros(
                (num_rows, num_cols),
                dtype=[("Gain", np.float32), ("Phase", np.float32)],
            )
            if data_filler is None:
                retval[num_rows // 2, num_cols // 2] = data
            else:
                data_filler(retval)
            return skcrsd.mask_support_array(
                retval,
                crsd_con.crsdroot.findtext(
                    f"{{*}}SupportArray/{{*}}GainPhaseArray[{{*}}Identifier='{identifier}']/{{*}}NODATA"
                ),
            )

        monkeypatch.setattr(crsd_con, "_get_support_array", dummy)
        return crsd_con, data

    return func


@pytest.mark.parametrize("name,val", [("Gain", (1.0, 0.0)), ("Phase", (0.0, 1.0))])
def test_check_ant_gain_phase(ant_patched_crsd_con, name, val):
    crsd_con, _ = ant_patched_crsd_con(data=val)
    crsd_con.check("check_ant_gain_phase")
    assert_failures(crsd_con, f"has zero {name}")


@pytest.mark.parametrize("name,val", [("Gain", (1.0, 0.0)), ("Phase", (0.0, 1.0))])
def test_check_ant_gain_phase_nodata(ant_patched_crsd_con, name, val):
    crsd_con, data_value = ant_patched_crsd_con(data=val)
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    for gp_node in crsd_con.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
        _remove(gp_node, "{*}NODATA")
        gp_node.append(_make_elem(f"{{{elem_ns}}}NODATA", data_value.tobytes().hex()))
    crsd_con.check("check_ant_gain_phase")
    assert_failures(crsd_con, r"has data in sample at \(0, 0\)")
    assert_not_failures(crsd_con, f"has zero {name}")


@pytest.mark.parametrize("var", ["X0", "Y0", "XSS", "YSS"])
@pytest.mark.parametrize("name,val", [("Gain", (1.0, 0.0)), ("Phase", (0.0, 1.0))])
def test_check_ant_gain_phase_want_contains(ant_patched_crsd_con, var, name, val):
    crsd_con, _ = ant_patched_crsd_con(data=val)
    for gp in crsd_con.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
        gp.find("{*}" + var).text = "1e-10"
    crsd_con.check("check_ant_gain_phase")
    assert_failures(crsd_con, r"contains \(0, 0\)")
    assert_not_failures(crsd_con, f"has zero {name}")


@pytest.mark.parametrize("var", ["X0", "Y0", "XSS", "YSS"])
def test_ant_gp_extent(crsd_con, var):
    for gp in crsd_con.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
        gp.find("{*}" + var).text = "1.1"
    crsd_con.check("check_ant_gp_extent")
    assert_failures(crsd_con, "within unit square")


@pytest.mark.parametrize("var", ["NumRows", "NumCols"])
def test_ant_gp_size(crsd_con, var):
    for gp in crsd_con.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
        identifier = gp.findtext("{*}Identifier")
        data_node = crsd_con.crsdroot.find(
            f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{identifier}']"
        )
        data_node.find("{*}" + var).text = "1"
    crsd_con.check("check_ant_gp_size")
    assert_failures(crsd_con, "2x2")


def test_ant_gp_nan_pass(ant_patched_crsd_con):
    crsd_con, _ = ant_patched_crsd_con(data=(1.0, 0.0))
    crsd_con.check("check_ant_gp_nan")
    assert crsd_con.passes() and not crsd_con.failures()


def test_ant_gp_nan_pass_nodata(ant_patched_crsd_con):
    def filler(arr):
        # set one element to not have a NaN
        arr[0, 1] = (1.0, 0.0)

    crsd_con, data_value = ant_patched_crsd_con(data=(np.nan, 0.0))
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    for gp_node in crsd_con.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
        _remove(gp_node, "{*}NODATA")
        gp_node.append(_make_elem(f"{{{elem_ns}}}NODATA", data_value.tobytes().hex()))
    crsd_con.check("check_ant_gp_nan")
    assert crsd_con.passes() and not crsd_con.failures()


def test_ant_gp_nan_fail_nodata(ant_patched_crsd_con):
    def filler(arr):
        # set one element to have a NaN and is not NODATA
        arr[0, 1] = (np.nan, np.nan)

    crsd_con, data_value = ant_patched_crsd_con(data=(np.nan, 0.0), data_filler=filler)
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    for gp_node in crsd_con.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
        _remove(gp_node, "{*}NODATA")
        gp_node.append(_make_elem(f"{{{elem_ns}}}NODATA", data_value.tobytes().hex()))
    crsd_con.check("check_ant_gp_nan")
    assert_failures(crsd_con, "NaN")


def test_ant_gp_nan_fail(ant_patched_crsd_con):
    crsd_con, _ = ant_patched_crsd_con(data=(np.nan, 0.0))
    crsd_con.check("check_ant_gp_nan")
    assert_failures(crsd_con, "NaN")


@pytest.mark.parametrize("antdir", ["X", "Y"])
def test_check_ant_gain_phase_not_sample(crsd_con, antdir):
    d0 = crsd_con.crsdroot.find("{*}SupportArray/{*}GainPhaseArray/{*}" + f"{antdir}0")
    dss = crsd_con.crsdroot.find(
        "{*}SupportArray/{*}GainPhaseArray/{*}" + f"{antdir}SS"
    )
    d0.text = str(float(d0.text) + float(dss.text) / 2)
    crsd_con.check("check_ant_gain_phase")
    assert_not_failures(crsd_con, r"has zero {name}")


def test_reffreqfixed_true(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RefFreq"][0] += 100e6
    crsd_con.check("check_reffreqfixed", allow_prefix=True)
    assert_failures(crsd_con, "PVP is fixed")


def test_reffreqfixed_false(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    crsd_con.crsdroot.find("{*}Channel/{*}Parameters/{*}RefFreqFixed").text = "false"
    crsd_con.check("check_reffreqfixed", allow_prefix=True)
    assert_failures(crsd_con, "PVP is not fixed")


@pytest.mark.parametrize("name", ["FRCV1", "FRCV2"])
def test_frcvfixed_fixed_frcv1(crsd_con, name):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp[name][0] += 100e6
    crsd_con.check("check_frcvfixed", allow_prefix=True)
    assert_failures(crsd_con, f"{name} PVP is fixed")


def test_frcvfixed_false(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    crsd_con.crsdroot.find("{*}Channel/{*}Parameters/{*}FrcvFixed").text = "false"
    crsd_con.check("check_frcvfixed", allow_prefix=True)
    assert_failures(crsd_con, "Either FRCV1 or FRCV2 is not fixed")


def test_signalnormal_false(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    crsd_con.crsdroot.find("{*}Channel/{*}Parameters/{*}SignalNormal").text = "false"
    crsd_con.check("check_signalnormal", allow_prefix=True)
    assert_failures(crsd_con, "SIGNAL PVP is not always 1")


def test_signalnormal_true(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["SIGNAL"][0] = 2
    crsd_con.check("check_signalnormal", allow_prefix=True)
    assert_failures(crsd_con, "SIGNAL PVP is always 1")


def test_inst_osr(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    crsd_con.crsdroot.find(
        "{*}Channel/{*}Parameters/{*}BWInst"
    ).text = crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Fs")
    crsd_con.check("check_inst_osr", allow_prefix=True)
    assert_failures(crsd_con, "sufficient oversample")


def test_rcvstart12_order(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    rcvstart1 = crsd_con.crsdroot.find("{*}Channel/{*}Parameters/{*}RcvStartTime1")
    rcvstart2 = crsd_con.crsdroot.find("{*}Channel/{*}Parameters/{*}RcvStartTime2")
    rcvstart1.text, rcvstart2.text = rcvstart2.text, rcvstart1.text
    crsd_con.check("check_rcvstart12", allow_prefix=True)
    assert_failures(crsd_con, "not greater than")


def test_rcvstart12_1pvp(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RcvStart"]["Int"][0] -= 1
    crsd_con.check("check_rcvstart12", allow_prefix=True)
    assert_failures(crsd_con, "first PVP")


def test_rcvstart12_2pvp(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RcvStart"]["Int"][-1] += 1
    crsd_con.check("check_rcvstart12", allow_prefix=True)
    assert_failures(crsd_con, "last PVP")


@pytest.mark.parametrize("name, offset", [["FRCV1", -1], ["FRCV2", +1]])
def test_fcvminmax_min(crsd_con, name, offset):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp[name][0] += offset
    crsd_con.check("check_frcvminmax", allow_prefix=True)
    assert_failures(crsd_con, name)


def test_rcvstart_not_increasing(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RcvStart"][1] = pvp["RcvStart"][0]
    crsd_con.check("check_rcvstart", allow_prefix=True)
    assert_failures(crsd_con, "RcvStart is strictly increasing")


def test_rcvstart_frac_negative(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RcvStart"]["Frac"][1] = -0.5
    crsd_con.check("check_rcvstart", allow_prefix=True)
    assert_failures(crsd_con, "RcvStart.Frac is non-negative")


def test_rcvstart_frac_too_big(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RcvStart"]["Frac"][1] = 1.5
    crsd_con.check("check_rcvstart", allow_prefix=True)
    assert_failures(crsd_con, "RcvStart.Frac is less than 1")


def test_rcvstart_sample(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    fs = float(crsd_con.crsdroot.findtext("{*}Channel/{*}Parameters/{*}Fs"))
    pvp["RcvStart"]["Frac"][1] += 0.5 / fs
    crsd_con.check("check_rcvstart_sample", allow_prefix=True)
    assert_failures(crsd_con, "integer multiple")


def test_rcvstart_overlap(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    fs = crsd_con.xmlhelp.load("{*}Channel/{*}Parameters/{*}Fs")
    crsd_con.xmlhelp.set("{*}Channel/{*}Parameters/{*}Fs", fs / 2e6)
    crsd_con.check("check_rcvstart_overlap", allow_prefix=True)
    assert_failures(crsd_con, "overlap")


def test_rcvposveltime_matched(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RcvPos"][:] = pvp["RcvPos"][::-1]
    crsd_con.check("check_rcvposveltime", allow_prefix=True)
    assert_failures(crsd_con, "matched set")


def test_refphi0_intfrac_negative(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RefPhi0"]["Frac"][1] = -0.5
    crsd_con.check("check_refphi0_intfrac", allow_prefix=True)
    assert_failures(crsd_con, "RefPhi0.Frac is non-negative")


def test_refphi0_intfrac_too_big(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RefPhi0"]["Frac"][1] = 1.5
    crsd_con.check("check_refphi0_intfrac", allow_prefix=True)
    assert_failures(crsd_con, "RefPhi0.Frac is less than 1")


def test_signal_pvp(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["SIGNAL"][1] = 4
    crsd_con.check("check_signal_pvp", allow_prefix=True)
    assert_failures(crsd_con, "in range")


def test_ampsf_pvp(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["AmpSF"][1] = -0.1
    crsd_con.check("check_ampsf_pvp", allow_prefix=True)
    assert_failures(crsd_con, "is positive")


# only applicable to CRSDsar
@pytest.mark.parametrize("bad_value", [-2, 2**62])
def test_txpulseindex(example_crsdsar_file, bad_value):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["TxPulseIndex"][2] = bad_value
    crsd_con.check("check_txpulseindex_pvp", allow_prefix=True)
    assert_failures(crsd_con, "is either -1 or a valid index")


# only applicable to CRSDsar
def test_txpulseindex_signal1(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["TxPulseIndex"][2] = -1
    pvp["SIGNAL"][2] = 1
    crsd_con.check("check_txpulseindex_pvp", allow_prefix=True)
    assert_failures(crsd_con, "is not -1 when SIGNAL is 1")


# only applicable to CRSDsar
def test_txpulseindex_signal2(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["SIGNAL"][2] = 2
    crsd_con.check("check_txpulseindex_pvp", allow_prefix=True)
    assert_failures(crsd_con, "is -1 when SIGNAL is 2")


@pytest.mark.parametrize("axis", ["X", "Y"])
def test_rcv_acxy_not_unit(crsd_con, axis):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp[f"RcvAC{axis}"][0] *= 3
    crsd_con.check("check_rcv_acxy", allow_prefix=True)
    assert_failures(crsd_con, f"RcvAC{axis} is unit length")


def test_rcvacxy_not_ortho(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RcvACX"][0] = pvp["RcvACY"][0]
    crsd_con.check("check_rcv_acxy", allow_prefix=True)
    assert_failures(crsd_con, "are orthogonal")


def test_rcveb(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["RcvEB"][0] = [0.8, 0.8]
    crsd_con.check("check_rcveb", allow_prefix=True)
    assert_failures(crsd_con, "unit length")


def test_dfic0_ficrate_zero_ficrate(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["DFIC0"][0] = 1
    crsd_con.check("check_dfic0_ficrate", allow_prefix=True)
    assert_failures(crsd_con, "DFIC0 = 0")


def test_dfic0_ficrate_nonzero_ficrate(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    channel_id = crsd_con.crsdroot.findtext("{*}Channel/{*}RefChId")
    pvp = crsd_con._get_channel_pvps(channel_id)
    pvp["DFIC0"][0] = 1
    pvp["FICRate"][0] = 1
    crsd_con.check("check_dfic0_ficrate", allow_prefix=True)
    assert_failures(crsd_con, "for some point")


@pytest.fixture
def dwell_array_patched_crsd_con(example_crsdsar_file, monkeypatch):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    # switch dwell to array
    sardwell = crsd_con.crsdroot.find(
        "{*}Channel/{*}Parameters/{*}SARImage/{*}DwellTimes"
    )
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    arr = etree.Element(f"{{{elem_ns}}}Array")
    arr.append(_make_elem(f"{{{elem_ns}}}DTAId", "dwell array"))
    sardwell.append(arr)
    _remove(sardwell, "{*}Polynomials")

    # use a grid such that 0, 0 is not on a sample but is in the middle
    # and none of the other grid parameters are the same for x and y

    # add dwell array to /Data
    num_arr = crsd_con.xmlhelp.load("{*}Data/{*}Support/{*}NumSupportArrays") + 1
    crsd_con.xmlhelp.set("{*}Data/{*}Support/{*}NumSupportArrays", num_arr)
    dataarr = etree.Element(f"{{{elem_ns}}}SupportArray")
    dataarr.append(_make_elem(f"{{{elem_ns}}}SAId", "dwell array"))
    dataarr.append(_make_elem(f"{{{elem_ns}}}NumRows", "54"))
    dataarr.append(_make_elem(f"{{{elem_ns}}}NumCols", "52"))
    dataarr.append(_make_elem(f"{{{elem_ns}}}BytesPerElement", "8"))
    dataarr.append(_make_elem(f"{{{elem_ns}}}ArrayByteOffset", "0"))
    crsd_con.crsdroot.find("{*}Data/{*}Support").append(dataarr)
    _repack_support_arrays(crsd_con.crsdroot)

    # add dwell array to /SupportArray
    dta = etree.Element(f"{{{elem_ns}}}DwellTimeArray")
    dta.append(_make_elem(f"{{{elem_ns}}}Identifier", "dwell array"))
    dta.append(_make_elem(f"{{{elem_ns}}}ElementFormat", "COD=F4;DT=F4;"))
    dta.append(_make_elem(f"{{{elem_ns}}}X0", "-291.5"))
    dta.append(_make_elem(f"{{{elem_ns}}}Y0", "-255.0"))
    dta.append(_make_elem(f"{{{elem_ns}}}XSS", "11.0"))
    dta.append(_make_elem(f"{{{elem_ns}}}YSS", "10.0"))
    crsd_con.crsdroot.find("{*}SupportArray/{*}XMArray").addnext(dta)

    # monkeypatch in a dwell
    cod = crsd_con.xmlhelp.load("{*}ReferenceGeometry/{*}SARImage/{*}CODTime")
    dt = crsd_con.xmlhelp.load("{*}ReferenceGeometry/{*}SARImage/{*}DwellTime")
    old_get_support_array = crsd_con._get_support_array

    def dummy(identifier):
        if identifier != "dwell array":
            return old_get_support_array(identifier)
        data_node = crsd_con.crsdroot.find(
            f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{identifier}']"
        )
        num_rows = int(data_node.findtext("{*}NumRows"))
        num_cols = int(data_node.findtext("{*}NumCols"))
        retval = np.empty((num_rows, num_cols), dtype=[("COD", ">f4"), ("DT", ">f4")])
        # Add a gradient to each of the arrays so that we can make sure we're linearly interpolating
        retval["COD"] = (
            cod
            + np.linspace(-1, 1, num_rows)[:, np.newaxis]
            + np.linspace(-2, 2, num_cols)
        )
        retval["DT"] = (
            dt
            + np.linspace(-1, 1, num_rows)[:, np.newaxis]
            + np.linspace(-2, 2, num_cols)
        )
        return retval

    monkeypatch.setattr(crsd_con, "_get_support_array", dummy)

    # make sure it looks fairly reasonable, the support block size is now wrong, but we can just
    # ignore that check
    crsd_con.check(ignore_patterns=["check_support_block_size_and_packing"])
    assert crsd_con.passes() and not crsd_con.failures()
    return crsd_con


# only applicable to CRSDsar
def test_dwell_array_coverage_x2y2(dwell_array_patched_crsd_con):
    crsd_con = dwell_array_patched_crsd_con
    crsd_con.crsdroot.find("{*}SupportArray/{*}DwellTimeArray/{*}XSS").text = "1.0"
    crsd_con.check("check_dwell_array_coverage", allow_prefix=True)
    assert_failures(crsd_con, "covers X2Y2")


# only applicable to CRSDsar
def test_dwell_array_coverage_x1y1(dwell_array_patched_crsd_con):
    crsd_con = dwell_array_patched_crsd_con
    crsd_con.crsdroot.find("{*}SupportArray/{*}DwellTimeArray/{*}X0").text = "0.0"
    crsd_con.check("check_dwell_array_coverage", allow_prefix=True)
    assert_failures(crsd_con, "covers X1Y1")


# only applicable to CRSDsar
@pytest.mark.parametrize("name", ["COD", "Dwell"])
def test_check_num_coddwell_times(example_crsdsar_file, name):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find(f"{{*}}DwellPolynomials/{{*}}Num{name}Times").text = "2"
    crsd_con.check(f"check_num{name.lower()}times")
    assert_failures(crsd_con, "Num.* is correct")


def test_scene_iarp_ecf_llh_mismatch(crsd_con):
    crsd_con.crsdroot.find("{*}SceneCoordinates/{*}IARP/{*}LLH/{*}HAE").text = "5e4"
    crsd_con.check("check_scene_iarp", allow_prefix=True)
    assert_failures(crsd_con, "ECF matches LLH")


def test_scene_iarp_not_near_earth(crsd_con):
    crsd_con.crsdroot.find("{*}SceneCoordinates/{*}IARP/{*}LLH/{*}HAE").text = "5e5"
    llh = crsd_con.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}LLH")
    crsd_con.xmlhelp.set(
        "{*}SceneCoordinates/{*}IARP/{*}ECF", sarkit.wgs84.geodetic_to_cartesian(llh)
    )
    crsd_con.check("check_scene_iarp", allow_prefix=True)
    assert_failures(crsd_con, "near Earth's surface")


@pytest.mark.parametrize("vec", ["uIAX", "uIAY"])
def test_scene_planar_axes_unit(crsd_con, vec):
    crsd_con.crsdroot.find(
        f"{{*}}SceneCoordinates/{{*}}ReferenceSurface/{{*}}Planar/{{*}}{vec}/{{*}}X"
    ).text = "1.1"
    crsd_con.check("check_scene_planar_axes", allow_prefix=True)
    assert_failures(crsd_con, f"{vec} is unit length")


def test_scene_planar_axes_orthogonal(crsd_con):
    iax = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"
    )
    crsd_con.xmlhelp.set(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY", iax
    )
    crsd_con.check("check_scene_planar_axes", allow_prefix=True)
    assert_failures(crsd_con, "are orthogonal")


def test_scene_planar_axes_iaz(crsd_con):
    iax = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"
    )
    iay = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"
    )
    crsd_con.xmlhelp.set(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY", iax
    )
    crsd_con.xmlhelp.set(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX", iay
    )
    crsd_con.check("check_scene_planar_axes", allow_prefix=True)
    assert_failures(crsd_con, "uIAZ is upward")


def _replace_plane_with_hae(crsd_con):
    plane_iax = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"
    )
    plane_iay = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"
    )
    iarp_ecf = crsd_con.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}ECF")
    iarp_llh = crsd_con.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}LLH")
    hae_iax = np.deg2rad(
        (sarkit.wgs84.cartesian_to_geodetic(iarp_ecf + plane_iax) - iarp_llh)[:2]
    )
    hae_iay = np.deg2rad(
        (sarkit.wgs84.cartesian_to_geodetic(iarp_ecf + plane_iay) - iarp_llh)[:2]
    )
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    hae_node = etree.Element(f"{{{elem_ns}}}HAE")
    hae_node.append(etree.Element(f"{{{elem_ns}}}uIAXLL"))
    hae_node.append(etree.Element(f"{{{elem_ns}}}uIAYLL"))
    crsd_con.crsdroot.find("{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar").addnext(
        hae_node
    )
    _remove(crsd_con.crsdroot, "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar")
    crsd_con.xmlhelp.set_elem(hae_node.find("{*}uIAXLL"), hae_iax)
    crsd_con.xmlhelp.set_elem(hae_node.find("{*}uIAYLL"), hae_iay)
    crsd_con.check("check_against_schema")
    assert crsd_con.passes() and not crsd_con.failures()


def test_check_refgeom_point_ecf(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        crsd_con.crsdroot.find(
            "{*}TxSequence/{*}Parameters/{*}TxRefPoint/{*}ECF/{*}X"
        ).text = "123.0"
    else:
        crsd_con.crsdroot.find(
            "{*}Channel/{*}Parameters/{*}RcvRefPoint/{*}ECF/{*}X"
        ).text = "123.0"
    crsd_con.check("check_refgeom")
    assert_failures(crsd_con, "RefPoint/ECF matches")


@pytest.mark.parametrize("name", ["AmpH", "AmpV", "PhaseH", "PhaseV"])
def test_check_txpolarization(crsd_con, name):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("Test not applicable to CRSDrcv")
    crsd_con.crsdroot.find(
        "{*}TxSequence/{*}Parameters/{*}TxPolarization/{*}" + name
    ).text = "0.12345"
    crsd_con.check("check_txpolarization", allow_prefix=True)
    assert_failures(crsd_con, f"{name[-1]} component is correct")
    if "Amp" in name:
        assert_failures(crsd_con, "normalized")


@pytest.mark.parametrize("name", ["AmpH", "AmpV", "PhaseH", "PhaseV"])
def test_check_rcvpolarization(crsd_con, name):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("Test not applicable to CRSDtx")
    crsd_con.crsdroot.find(
        "{*}Channel/{*}Parameters/{*}RcvPolarization/{*}" + name
    ).text = "0.12345"
    crsd_con.check("check_rcvpolarization", allow_prefix=True)
    assert_failures(crsd_con, f"{name[-1]} component is correct")
    if "Amp" in name:
        assert_failures(crsd_con, "normalized")


@pytest.mark.parametrize("name", ["AmpH", "AmpV", "PhaseH", "PhaseV"])
def test_check_sartxpolarization(example_crsdsar_file, name):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find(
        "{*}Channel/{*}Parameters/{*}SARImage/{*}TxPolarization/{*}" + name
    ).text = "0.12345"
    crsd_con.check("check_sartxpolarization", allow_prefix=True)
    assert_failures(crsd_con, f"{name[-1]} component is correct")
    if "Amp" in name:
        assert_failures(crsd_con, "normalized")


# only in CRSDsar
def test_check_sarimage_refgeom(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find("{*}Channel/{*}Parameters/{*}RefVectorIndex").text = "3"
    crsd_con.check("check_refgeom")
    # Almost all the fields should be wrong if we use the wrong vector
    assert_failures(crsd_con, "ReferenceTime")
    assert_failures(crsd_con, "ARPPos")
    assert_failures(crsd_con, "Incidence")
    assert_failures(crsd_con, "Azimuth")


# only in CRSDsar
@pytest.mark.parametrize("name", ["CODTime", "DwellTime"])
def test_check_sarimage_refgeom_dwell_poly(example_crsdsar_file, name):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find("{*}ReferenceGeometry/{*}SARImage/{*}" + name).text = "0"
    crsd_con.check("check_refgeom")
    assert_failures(crsd_con, name)


# only in CRSDsar
@pytest.mark.parametrize("name", ["CODTime", "DwellTime"])
def test_check_sarimage_refgeom_dwell_array(dwell_array_patched_crsd_con, name):
    crsd_con = dwell_array_patched_crsd_con
    crsd_con.crsdroot.find("{*}ReferenceGeometry/{*}SARImage/{*}" + name).text = "0"
    crsd_con.check("check_refgeom")
    assert_failures(crsd_con, name)


# only applies to CRSDsar
def test_check_tx_refgeom_sarpass(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    # the reference geometry does not depend of RefPulseIndex for CRSDsar
    crsd_con.crsdroot.find("{*}TxSequence/{*}Parameters/{*}RefPulseIndex").text = "3"
    crsd_con.check("check_refgeom")
    assert crsd_con.passes() and not crsd_con.failures()


@pytest.mark.parametrize(
    "example_file_fixture,key",
    [
        ("example_crsdsar_file", "RefVectorPulseIndex"),
        ("example_crsdtx_file", "RefPulseIndex"),
    ],
)
def test_check_tx_refgeom(request, example_file_fixture, key):
    example_file = request.getfixturevalue(example_file_fixture)
    crsd_con = CrsdConsistency.from_file(example_file)
    crsd_con.crsdroot.find(".//{*}" + key).text = "3"
    crsd_con.check("check_refgeom")
    assert_failures(crsd_con, "Time")
    assert_failures(crsd_con, "APCPos")
    assert_failures(crsd_con, "Incidence")


def test_check_rcv_refgeom(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("test not applicable with CRSDtx")
    crsd_con.crsdroot.find(".//{*}RefVectorIndex").text = "3"
    crsd_con.check("check_refgeom")
    assert_failures(crsd_con, "Time")
    assert_failures(crsd_con, "APCPos")
    assert_failures(crsd_con, "Incidence")


def test_scene_hae_axes_pass(crsd_con):
    _replace_plane_with_hae(crsd_con)
    crsd_con.check("check_scene_hae_axes", allow_prefix=True)
    assert crsd_con.passes() and not crsd_con.failures()


@pytest.mark.parametrize("vec", ["uIAXLL", "uIAYLL"])
def test_scene_hae_axes_unit(crsd_con, vec):
    _replace_plane_with_hae(crsd_con)
    hae_uvec = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}" + vec
    )
    crsd_con.xmlhelp.set(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}" + vec, hae_uvec * 1.1
    )
    crsd_con.check("check_against_schema")
    assert crsd_con.passes() and not crsd_con.failures()
    crsd_con.check("check_scene_hae_axes", allow_prefix=True)
    assert_failures(crsd_con, f"{vec} is unit length")


def test_scene_hae_axes_orthogonal(crsd_con):
    _replace_plane_with_hae(crsd_con)
    hae_iax = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAXLL"
    )
    crsd_con.xmlhelp.set(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAYLL", hae_iax
    )
    crsd_con.check("check_against_schema")
    assert crsd_con.passes() and not crsd_con.failures()
    crsd_con.check("check_scene_hae_axes", allow_prefix=True)
    assert_failures(crsd_con, "are orthogonal")


def test_scene_hae_axes_iaz(crsd_con):
    _replace_plane_with_hae(crsd_con)
    iax = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAXLL"
    )
    iay = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAYLL"
    )
    crsd_con.xmlhelp.set(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAYLL", iax
    )
    crsd_con.xmlhelp.set(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAXLL", iay
    )
    crsd_con.check("check_against_schema")
    assert crsd_con.passes() and not crsd_con.failures()
    crsd_con.check("check_scene_hae_axes", allow_prefix=True)
    assert_failures(crsd_con, "uIAZ is upward")


@pytest.mark.parametrize(
    "path,name",
    [
        ("{*}SceneCoordinates/{*}ImageArea/{*}Polygon", "scene_imagearea"),
        ("{*}SceneCoordinates/{*}ExtendedArea/{*}Polygon", "scene_extendedarea"),
        (
            "{*}Channel/{*}Parameters/{*}SARImage/{*}ImageArea/{*}Polygon",
            "channel_imagearea",
        ),
    ],
)
def test_imagearea_polygon_clockwise(crsd_con, path, name):
    if ("SAR" in path or "Extended" in path) and crsd_con.crsd_type != "CRSDsar":
        pytest.skip("Test only applicable to CRSDsar")
    polygon = crsd_con.xmlhelp.load(path)
    crsd_con.xmlhelp.set(path, polygon[::-1])
    crsd_con.check(f"check_{name}_polygon", allow_prefix=True)
    assert_failures(crsd_con, "is clockwise")


@pytest.mark.parametrize(
    "path,name",
    [
        ("{*}SceneCoordinates/{*}ImageArea/{*}Polygon", "scene_imagearea"),
        ("{*}SceneCoordinates/{*}ExtendedArea/{*}Polygon", "scene_extendedarea"),
        (
            "{*}Channel/{*}Parameters/{*}SARImage/{*}ImageArea/{*}Polygon",
            "channel_imagearea",
        ),
    ],
)
def test_imagearea_polygon_simple(crsd_con, path, name):
    if ("SAR" in path or "Extended" in path) and crsd_con.crsd_type != "CRSDsar":
        pytest.skip("Test only applicable to CRSDsar")
    polygon = crsd_con.xmlhelp.load(path)
    polygon[:2] = polygon[1::-1]
    crsd_con.xmlhelp.set(path, polygon)
    crsd_con.check(f"check_{name}_polygon", allow_prefix=True)
    assert_failures(crsd_con, "is simple")


@pytest.mark.parametrize(
    "path,name",
    [
        ("{*}SceneCoordinates/{*}ImageArea/{*}Polygon", "scene_imagearea"),
        ("{*}SceneCoordinates/{*}ExtendedArea/{*}Polygon", "scene_extendedarea"),
        (
            "{*}Channel/{*}Parameters/{*}SARImage/{*}ImageArea/{*}Polygon",
            "channel_imagearea",
        ),
    ],
)
def test_imagearea_polygon_bounded(crsd_con, path, name):
    if ("SAR" in path or "Extended" in path) and crsd_con.crsd_type != "CRSDsar":
        pytest.skip("Test only applicable to CRSDsar")
    polygon = crsd_con.xmlhelp.load(path)
    polygon *= 0.9
    crsd_con.xmlhelp.set(path, polygon)
    crsd_con.check(f"check_{name}_polygon", allow_prefix=True)
    assert_failures(crsd_con, "bounded")


# SegmentList is only in CRSDsar
def test_segment_list_num_segments(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find(
        "{*}SceneCoordinates/{*}ImageGrid/{*}SegmentList/{*}NumSegments"
    ).text += "9"
    crsd_con.check("check_segment_list")
    assert_failures(crsd_con, "NumSegments is correct")


@pytest.mark.parametrize("field", ["StartLine", "StartSample", "EndLine", "EndSample"])
def test_segment_list_in_grid_low(example_crsdsar_file, field):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find(
        "{*}SceneCoordinates/{*}ImageGrid/{*}SegmentList/{*}Segment/{*}" + field
    ).text = "-1"
    crsd_con.check("check_segment_list")
    assert_failures(crsd_con, f"{field} is within grid")


@pytest.mark.parametrize("field", ["StartLine", "StartSample", "EndLine", "EndSample"])
def test_segment_list_in_grid_high(example_crsdsar_file, field):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find(
        "{*}SceneCoordinates/{*}ImageGrid/{*}SegmentList/{*}Segment/{*}" + field
    ).text = "500"
    crsd_con.check("check_segment_list")
    assert_failures(crsd_con, f"{field} is within grid")


@pytest.mark.parametrize("field", ["EndLine", "EndSample"])
def test_segment_list_size(example_crsdsar_file, field):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find(
        "{*}SceneCoordinates/{*}ImageGrid/{*}SegmentList/{*}Segment/{*}" + field
    ).text = "0"
    crsd_con.check("check_segment_list")
    assert_failures(crsd_con, "Start.* less than End")


@pytest.mark.parametrize("field", ["StartLine", "StartSample", "EndLine", "EndSample"])
def test_segment_list_polygon_bound(example_crsdsar_file, field):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find(
        "{*}SceneCoordinates/{*}ImageGrid/{*}SegmentList/{*}Segment/{*}" + field
    ).text = "10"
    crsd_con.check("check_segment_list")
    assert_failures(crsd_con, "bounded by segment limits")


def test_segment_list_polygon_simple(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    polygon = crsd_con.xmlhelp.load(".//{*}SegmentList//{*}SegmentPolygon")
    polygon[:2] = polygon[1::-1]
    crsd_con.xmlhelp.set(".//{*}SegmentList//{*}SegmentPolygon", polygon)
    crsd_con.check("check_segment_list")
    assert_failures(crsd_con, "is simple")


def test_segment_list_polygon_clockwise(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    polygon = crsd_con.xmlhelp.load(".//{*}SegmentList//{*}SegmentPolygon")
    crsd_con.xmlhelp.set(".//{*}SegmentList//{*}SegmentPolygon", polygon[::-1])
    crsd_con.check("check_segment_list")
    assert_failures(crsd_con, "is clockwise")


def test_num_support_arrays(crsd_con):
    _replace_plane_with_hae(crsd_con)
    num_support = crsd_con.crsdroot.find("{*}Data/{*}Support/{*}NumSupportArrays")
    num_support.text = str(int(num_support.text) + 1)
    crsd_con.check("check_numsupportarrays")
    assert_failures(crsd_con, "NumSupportArrays is correct")


def test_support_array_bytes_per_element(crsd_con):
    bpe = crsd_con.crsdroot.find(
        "{*}Data/{*}Support/{*}SupportArray/{*}BytesPerElement"
    )
    bpe.text = str(int(bpe.text) + 1)
    crsd_con.check("check_support_array_bytes_per_element")
    assert crsd_con.failures()


def test_num_txsequence(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("test not applicable with CRSDrcv")
    num_seq = crsd_con.crsdroot.find("{*}Data/{*}Transmit/{*}NumTxSequences")
    num_seq.text = str(int(num_seq.text) + 1)
    crsd_con.check("check_numtxsequences")
    assert_failures(crsd_con, "NumTxSequences is correct")


def test_numbytesppp(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("test not applicable with CRSDrcv")
    nbppp = crsd_con.crsdroot.find("{*}Data/{*}Transmit/{*}NumBytesPPP")
    nbppp.text = str(int(nbppp.text) + 1)
    crsd_con.check("check_numbytesppp")
    assert crsd_con.failures()


def test_ppp_min_offset(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("test not applicable with CRSDrcv")
    ppp_offsets = crsd_con.crsdroot.findall("{*}PPP//{*}Offset")
    min_offset = min(ppp_offsets, key=lambda x: int(x.text))
    min_offset.text = "1"
    crsd_con.check("check_ppp_min_offset")
    assert_failures(crsd_con, "First PPP has offset zero")


def test_ppp_unique_names(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("test not applicable with CRSDrcv")
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    added_ppp = _make_elem(f"{{{elem_ns}}}AddedPPP")
    added_ppp.append(_make_elem(f"{{{elem_ns}}}Name", "XMIndex"))
    added_ppp.append(_make_elem(f"{{{elem_ns}}}Offset", "0"))
    added_ppp.append(_make_elem(f"{{{elem_ns}}}Size", "1"))
    added_ppp.append(_make_elem(f"{{{elem_ns}}}Format", "I8"))
    crsd_con.crsdroot.find("{*}PPP").append(added_ppp)
    crsd_con.check("check_ppp_unique_names")
    assert_failures(crsd_con, "PPP names are unique")
    assert_failures(crsd_con, "XMIndex is not an AddedPPP")


def test_num_channnels(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("test not applicable with CRSDtx")
    num_chan = crsd_con.crsdroot.find("{*}Data/{*}Receive/{*}NumCRSDChannels")
    num_chan.text = str(int(num_chan.text) + 1)
    crsd_con.check("check_numcrsdchannels")
    assert_failures(crsd_con, "NumCRSDChannels is correct")


def test_numbytespvp(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("test not applicable with CRSDtx")
    nbpvp = crsd_con.crsdroot.find("{*}Data/{*}Receive/{*}NumBytesPVP")
    nbpvp.text = str(int(nbpvp.text) + 1)
    crsd_con.check("check_numbytespvp")
    assert crsd_con.failures()


def test_pvp_min_offset(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("test not applicable with CRSDtx")
    pvp_offsets = crsd_con.crsdroot.findall("{*}PVP//{*}Offset")
    min_offset = min(pvp_offsets, key=lambda x: int(x.text))
    min_offset.text = "1"
    crsd_con.check("check_pvp_min_offset")
    assert_failures(crsd_con, "First PVP has offset zero")


def test_pvp_unique_names(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("test not applicable with CRSDtx")
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    added_pvp = _make_elem(f"{{{elem_ns}}}AddedPVP")
    added_pvp.append(_make_elem(f"{{{elem_ns}}}Name", "TxPulseIndex"))
    added_pvp.append(_make_elem(f"{{{elem_ns}}}Offset", "0"))
    added_pvp.append(_make_elem(f"{{{elem_ns}}}Size", "1"))
    added_pvp.append(_make_elem(f"{{{elem_ns}}}Format", "I8"))
    crsd_con.crsdroot.find("{*}PVP").append(added_pvp)
    crsd_con.check("check_pvp_unique_names")
    if crsd_con.crsd_type == "CRSDsar":
        assert_failures(crsd_con, "PVP names are unique")
    assert_failures(crsd_con, "TxPulseIndex is not an AddedPVP")


def test_support_array_nodata_fail(crsd_con):
    ant_sa = crsd_con.crsdroot.find("{*}SupportArray/{*}GainPhaseArray")
    if ant_sa.find("{*}NODATA") is None:
        elem_ns = etree.QName(ant_sa).namespace
        ant_sa.append(etree.Element(f"{{{elem_ns}}}NODATA"))
    ant_sa.find("{*}NODATA").text = "F" * 13
    crsd_con.check("check_support_array_nodata")
    assert_failures(crsd_con, "right size")


def test_ns_fxr(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("test not applicable to CRSDrcv")
    fx_id = crsd_con.crsdroot.findtext(
        "{*}SupportArray/{*}FxResponseArray/{*}Identifier"
    )
    crsd_con.crsdroot.find(
        f'{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId="{fx_id}"]/{{*}}NumCols'
    ).text = "1"
    crsd_con.check("check_ns_fxr")
    assert_failures(crsd_con, "at least 3")


def test_check_maxxmbw(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("test not applicable to CRSDrcv")
    crsd_con.crsdroot.find("{*}SupportArray/{*}XMArray/{*}MaxXMBW").text = "100e9"
    crsd_con.check("check_maxxmbw")
    assert_failures(crsd_con, "OSR")


# only applicable to CRSDsar
def test_error_mono_bistatic_mono_used(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    crsd_con.crsdroot.find("{*}SARInfo/{*}CollectType").text = "BISTATIC"
    crsd_con.check("check_error_mono_bistatic")
    assert_failures(crsd_con, "Bistatic ErrorParameters branch is used")


def _make_bistatic_error(crsd_con):
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    # mock out a bistatic branch that matches the schema
    bistatic = etree.Element(f"{{{elem_ns}}}Bistatic")
    posvel = etree.Element(f"{{{elem_ns}}}PosVelError")
    posvel.append(_make_elem(f"{{{elem_ns}}}TxFrame", "ECF"))
    posvel.append(_make_elem(f"{{{elem_ns}}}TxPVCov", size1="6", size2="6"))
    posvel.append(_make_elem(f"{{{elem_ns}}}RcvFrame", "ECF"))
    posvel.append(_make_elem(f"{{{elem_ns}}}RcvPVCov", size1="6", size2="6"))
    posvel.append(_make_elem(f"{{{elem_ns}}}TxRcvPVCov", size1="6", size2="6"))
    bistatic.append(posvel)
    radarsensor = etree.Element(f"{{{elem_ns}}}RadarSensor")
    radarsensor.append(_make_elem(f"{{{elem_ns}}}TimeFreqCov", size1="4", size2="4"))
    bistatic.append(radarsensor)
    crsd_con.crsdroot.find("{*}ErrorParameters/{*}SARImage").append(bistatic)
    _remove(crsd_con.crsdroot, "{*}ErrorParameters/{*}SARImage/{*}Monostatic")

    # make sure we match the schema
    crsd_con.crsdroot.find("{*}SARInfo/{*}CollectType").text = "BISTATIC"
    crsd_con.check()
    assert crsd_con.passes() and not crsd_con.failures()


# only applicable to CRSDsar
def test_error_mono_bistatic_bistatic_used(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    _make_bistatic_error(crsd_con)

    # make sure the test catches if there is mismatch
    crsd_con.crsdroot.find("{*}SARInfo/{*}CollectType").text = "MONOSTATIC"
    crsd_con.check("check_error_mono_bistatic")
    assert_failures(crsd_con, "Monostatic ErrorParameters branch is used")


@pytest.mark.parametrize("short", ["ACF", "APC", "APAT"])
def test_check_antenna_ids_count(crsd_con, short):
    crsd_con.crsdroot.find(f"{{*}}Antenna/{{*}}Num{short}s").text += "1"
    crsd_con.check(f"check_num{short.lower()}s")
    assert_failures(crsd_con, f"Num{short}s is correct")


def test_check_ant_pol_ref(crsd_con):
    crsd_con.crsdroot.find("{*}Antenna/{*}AntPattern/{*}AntPolRef/{*}AmpY").text = "0.5"
    crsd_con.check("check_ant_pol_ref")
    assert_failures(crsd_con, "unit amplitude")


# only need one flavor (and a dummy at that) for this test
@pytest.mark.parametrize(
    "matrix, error",
    [
        ([[1, 2], [3, 4]], "TheName is symmetric"),
        ([[1, 4], [4, 4]], "TheName has only non-negative eigenvalues"),
    ],
)
def test_assert_symmetric_positive_semidefinite(example_crsdsar_file, matrix, error):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    the_matrix = np.array([[1, 1], [1, 2]])  # a matrix that should pass the checks

    def check_dummy(self):
        self.assert_symmetric_positive_semidefinite(the_matrix, "TheName")

    crsd_con.check_dummy = types.MethodType(check_dummy, crsd_con)
    crsd_con.funcs.append(crsd_con.check_dummy)

    # make sure the good matrix doesn't error out
    crsd_con.check("check_dummy")
    assert crsd_con.passes() and not crsd_con.failures()

    # replace the matrix with one that errors out
    the_matrix[...] = matrix
    crsd_con.check("check_dummy")
    assert_failures(crsd_con, error)


@pytest.mark.parametrize("name", ["PVCov", "TimeFreqCov"])
def test_check_error_cov_single_sensor(crsd_con, name):
    path = f"{{*}}ErrorParameters//{{*}}{name}"
    cov = crsd_con.xmlhelp.load(path)
    # This will make the matrix non-symmetric and have bad (likely complex) eigenvalues
    cov[0, 1] = cov[1, 0] + 1
    crsd_con.xmlhelp.set(path, cov)
    crsd_con.check("check_error_cov_single_sensor")
    assert_failures(crsd_con, f"{name} is symmetric")


# only applicable to CRSDsar
@pytest.mark.parametrize("name", ["TxPVCov", "RcvPVCov", "TimeFreqCov"])
def test_check_error_cov_bistatic_parts(example_crsdsar_file, name):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    _make_bistatic_error(crsd_con)
    path = f"{{*}}ErrorParameters//{{*}}{name}"
    cov = crsd_con.xmlhelp.load(path)
    # This will make the matrix non-symmetric and have bad (likely complex) eigenvalues
    cov[0, 1] = cov[1, 0] + 1
    crsd_con.xmlhelp.set(path, cov)
    crsd_con.check("check_error_cov_bistatic")
    assert_failures(crsd_con, f"{name} is symmetric")


# only applicable to CRSDsar
def test_check_error_cov_bistatic_full(example_crsdsar_file):
    crsd_con = CrsdConsistency.from_file(example_crsdsar_file)
    _make_bistatic_error(crsd_con)
    path = "{*}ErrorParameters//{*}TxRcvPVCov"
    cov = crsd_con.xmlhelp.load(path)
    # This will make the matrix have bad (likely complex) eigenvalues
    cov[0, 1] = 1e9
    crsd_con.xmlhelp.set(path, cov)
    crsd_con.check("check_error_cov_bistatic")
    assert_failures(crsd_con, "Full PVCov .* has only non-negative eigenvalues")


def test_check_block_order(crsd_con):
    crsd_con.kvp_list["SIGNAL_BLOCK_BYTE_OFFSET"] = "0"
    crsd_con.check("check_block_order")
    assert crsd_con.failures()


def test_check_post_header_section_terminator(crsd_con):
    crsd_con.kvp_list["XML_BLOCK_BYTE_OFFSET"] = "5"
    crsd_con.check("check_post_header_section_terminator_and_pad")
    assert crsd_con.failures()


def test_check_post_header_pad(crsd_con):
    crsd_con.kvp_list["XML_BLOCK_BYTE_OFFSET"] = str(
        int(crsd_con.kvp_list["XML_BLOCK_BYTE_OFFSET"]) + 1
    )
    crsd_con.check("check_post_header_section_terminator_and_pad")
    assert crsd_con.failures()


def test_check_post_xml_section_terminator(crsd_con):
    crsd_con.kvp_list["XML_BLOCK_SIZE"] = str(
        int(crsd_con.kvp_list["XML_BLOCK_SIZE"]) - 1
    )
    crsd_con.check("check_post_xml_section_terminator")
    assert crsd_con.failures()


def test_check_pad_before_binary_blocks(crsd_con):
    crsd_con.kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"] = str(
        int(crsd_con.kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"]) + 196
    )
    crsd_con.check("check_pad_before_binary_blocks")
    assert crsd_con.failures()


def test_check_support_block_size(crsd_con):
    crsd_con.kvp_list["SUPPORT_BLOCK_SIZE"] = str(
        int(crsd_con.kvp_list["SUPPORT_BLOCK_SIZE"]) - 1
    )
    crsd_con.check("check_support_block_size_and_packing")
    assert crsd_con.failures()


def test_check_support_block_packing(good_xml_root):
    root = copy_xml(good_xml_root)
    root.find("{*}Data/{*}Support/{*}SupportArray/{*}ArrayByteOffset").text += "1"
    crsd_con = CrsdConsistency(root)
    crsd_con.check("check_support_block_size_and_packing")
    assert crsd_con.failures()


def test_check_ppp_block_size(crsd_con):
    if crsd_con.crsd_type == "CRSDrcv":
        pytest.skip("test not applicable with CRSDrcv")
    crsd_con.kvp_list["PPP_BLOCK_SIZE"] = str(
        int(crsd_con.kvp_list["PPP_BLOCK_SIZE"]) - 1
    )
    crsd_con.check("check_ppp_block_size_and_packing")
    assert crsd_con.failures()


def test_check_ppp_block_packing(good_xml_root):
    root = copy_xml(good_xml_root)
    if root.find("{*}Data/{*}Transmit") is None:
        pytest.skip("test not applicable with CRSDrcv")
    root.find("{*}Data/{*}Transmit/{*}TxSequence/{*}PPPArrayByteOffset").text += "1"
    crsd_con = CrsdConsistency(root)
    crsd_con.check("check_ppp_block_size_and_packing")
    assert crsd_con.failures()


def test_check_pvp_block_size(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("test not applicable with CRSDtx")
    crsd_con.kvp_list["PVP_BLOCK_SIZE"] = str(
        int(crsd_con.kvp_list["PVP_BLOCK_SIZE"]) - 1
    )
    crsd_con.check("check_pvp_block_size_and_packing")
    assert crsd_con.failures()


def test_check_pvp_block_packing(good_xml_root):
    root = copy_xml(good_xml_root)
    if root.find("{*}Data/{*}Receive") is None:
        pytest.skip("test not applicable with CRSDtx")
    root.find("{*}Data/{*}Receive/{*}Channel/{*}PVPArrayByteOffset").text += "1"
    crsd_con = CrsdConsistency(root)
    crsd_con.check("check_pvp_block_size_and_packing")
    assert crsd_con.failures()


def test_check_signal_block_size_header(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        pytest.skip("test not applicable with CRSDtx")
    crsd_con.kvp_list["SIGNAL_BLOCK_SIZE"] = str(
        int(crsd_con.kvp_list["SIGNAL_BLOCK_SIZE"]) - 1
    )
    crsd_con.check("check_signal_block_size_and_packing")
    assert crsd_con.failures()


def test_check_signal_block_packing(good_xml_root):
    root = copy_xml(good_xml_root)
    if root.find("{*}Data/{*}Receive") is None:
        pytest.skip("test not applicable with CRSDtx")
    root.find("{*}Data/{*}Receive/{*}Channel/{*}SignalArrayByteOffset").text += "1"
    crsd_con = CrsdConsistency(root)
    crsd_con.check("check_signal_block_size_and_packing")
    assert crsd_con.failures()


def test_end_of_file_at_last_block(crsd_con):
    if crsd_con.crsd_type == "CRSDtx":
        crsd_con.kvp_list["PPP_BLOCK_SIZE"] = "0"
    else:
        crsd_con.kvp_list["SIGNAL_BLOCK_SIZE"] = "0"

    crsd_con.check("check_end_of_file_at_last_block")
    assert crsd_con.failures()


def test_assert_iac_matches_ecf_planar(crsd_con):
    iarp = crsd_con.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}ECF")
    uiax = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"
    )
    uiay = crsd_con.xmlhelp.load(
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"
    )
    crsd_con.assert_iac_matches_ecf([0, 0], iarp)

    crsd_con.assert_iac_matches_ecf([10, 0], iarp + 10 * uiax)
    crsd_con.assert_iac_matches_ecf([0, 10], iarp + 10 * uiay)
    crsd_con.assert_iac_matches_ecf([0, 10], iarp, tol=11)
    with pytest.raises(AssertionError):
        crsd_con.assert_iac_matches_ecf([10, 0], iarp + 10 * uiay)
    with pytest.raises(AssertionError):
        crsd_con.assert_iac_matches_ecf([10, 0], iarp)


def test_assert_iac_matches_ecf_hae(crsd_con):
    _remove(crsd_con.crsdroot, "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar")
    elem_ns = etree.QName(crsd_con.crsdroot).namespace
    hae = etree.Element(f"{{{elem_ns}}}HAE")
    iarp_llh = crsd_con.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}LLH")
    iarp_ecf = crsd_con.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}ECF")
    hae.append(etree.Element(f"{{{elem_ns}}}uIAXLL"))
    hae.append(etree.Element(f"{{{elem_ns}}}uIAYLL"))
    crsd_con.crsdroot.find("{*}SceneCoordinates/{*}ReferenceSurface").append(hae)
    iaxll = np.array([-2e-5, 1e-5])
    iayll = np.array([1.0e-5, 2e-5])
    crsd_con.xmlhelp.set_elem(hae.find("{*}uIAXLL"), iaxll)
    crsd_con.xmlhelp.set_elem(hae.find("{*}uIAYLL"), iayll)
    crsd_con.assert_iac_matches_ecf(
        [0, 0], crsd_con.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}ECF")
    )
    pt_llh = iarp_llh.copy()
    pt_llh[:2] += 10 * iaxll
    pt_ecf = sarkit.wgs84.geodetic_to_cartesian(pt_llh)
    crsd_con.assert_iac_matches_ecf([10, 0], pt_ecf)

    pt_llh2 = iarp_llh.copy()
    pt_llh2[:2] += 10 * iayll
    pt_ecf2 = sarkit.wgs84.geodetic_to_cartesian(pt_llh2)
    crsd_con.assert_iac_matches_ecf([0, 10], pt_ecf2)
    with pytest.raises(AssertionError):
        crsd_con.assert_iac_matches_ecf([10, 0], iarp_ecf)
    with pytest.raises(AssertionError):
        crsd_con.assert_iac_matches_ecf([0, 10], pt_ecf)


def test_check_compressed_signal_block_offsets(example_crsdrcvcompressed_file):
    crsd_con = CrsdConsistency.from_file(example_crsdrcvcompressed_file)
    crsd_con.xmlhelp.set("{*}Data/{*}Receive/{*}Channel/{*}SignalArrayByteOffset", 24)
    crsd_con.check("check_compressed_signal_block")
    assert_failures(crsd_con, "SignalArrayByteOffset is 0")


def test_check_compressed_signal_block_size(example_crsdrcvcompressed_file):
    crsd_con = CrsdConsistency.from_file(example_crsdrcvcompressed_file)
    css = crsd_con.crsdroot.find(
        "{*}Data/{*}Receive/{*}SignalCompression/{*}CompressedSignalSize"
    )
    crsd_con.xmlhelp.set_elem(css, crsd_con.xmlhelp.load_elem(css) + 1)
    crsd_con.check("check_compressed_signal_block")
    assert_failures(crsd_con, "SIGNAL_BLOCK_SIZE is set equal to CompressedSignalSize")


def test_smart_open_http(example_crsdsar):
    with tests.utils.static_http_server(example_crsdsar.parent) as server_url:
        assert not main([f"{server_url}/{example_crsdsar.name}", "--thorough"])


def test_smart_open_contract(example_crsdsar, monkeypatch):
    mock_open = unittest.mock.MagicMock(side_effect=tests.utils.simple_open_read)
    monkeypatch.setattr(sarkit.verification._crsdcheck, "open", mock_open)
    assert not main([str(example_crsdsar), "--thorough"])
    mock_open.assert_called_once_with(str(example_crsdsar), "rb")
