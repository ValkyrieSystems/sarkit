import copy
import pathlib

import numpy as np
import pytest
from lxml import etree

import sarkit.cphd as skcphd
import sarkit.crsd as skcrsd
import sarkit.sicd as sksicd
import sarkit.sidd as sksidd
from sarkit import _constants

DATAPATH = pathlib.Path(__file__).parents[1] / "data"

good_cphd_xml_path = DATAPATH / "example-cphd-1.0.1.xml"
good_crsd_xml_path = DATAPATH / "example-crsd-1.0.xml"
good_sicd_xml_path = DATAPATH / "example-sicd-1.2.1.xml"
good_sidd_xml_path = DATAPATH / "example-sidd-3.0.0.xml"


@pytest.fixture(scope="session")
def example_cphd(tmp_path_factory):
    cphd_etree = etree.parse(good_cphd_xml_path)
    xmlhelp = skcphd.XmlHelper(cphd_etree)

    cphd_plan = skcphd.Metadata(
        xmltree=cphd_etree,
    )
    pvp_dtype = skcphd.get_pvp_dtype(cphd_etree)

    assert int(cphd_etree.findtext("{*}Data/{*}NumCPHDChannels")) == 1
    assert cphd_etree.findtext("./{*}Data/{*}SignalArrayFormat") == "CF8"
    rng = np.random.default_rng(123456)
    num_vectors = xmlhelp.load(".//{*}Data/{*}Channel/{*}NumVectors")
    num_samples = xmlhelp.load(".//{*}Data/{*}Channel/{*}NumSamples")
    signal = (
        rng.random((num_vectors, num_samples, 2), dtype=np.float32)
        .view(np.complex64)
        .squeeze()
    )

    pvps = np.zeros((num_vectors), dtype=pvp_dtype)
    pvps["TxTime"] = np.linspace(
        xmlhelp.load(".//{*}TxTime1"),
        xmlhelp.load(".//{*}TxTime2"),
        num_vectors,
        endpoint=True,
    )
    arppos = xmlhelp.load(".//{*}ARPPos")
    arpvel = xmlhelp.load(".//{*}ARPVel")
    t_ref = xmlhelp.load(".//{*}ReferenceTime")

    arppoly = np.stack([(arppos - t_ref * arpvel), arpvel])

    fx1 = xmlhelp.load(".//{*}FxMin")
    fx2 = xmlhelp.load(".//{*}FxMax")
    pvps["FX1"][:] = fx1
    pvps["FX2"][:] = fx2
    pvps["SC0"] = fx1
    pvps["SCSS"] = (fx2 - fx1) / (num_samples - 1)
    pvps["TOA1"][:] = xmlhelp.load(".//{*}TOAMin")
    pvps["TOA2"][:] = xmlhelp.load(".//{*}TOAMax")

    pvps["TxPos"] = np.polynomial.polynomial.polyval(pvps["TxTime"], arppoly).T
    pvps["TxVel"] = np.polynomial.polynomial.polyval(
        pvps["TxTime"], np.polynomial.polynomial.polyder(arppoly)
    ).T

    pvps["RcvTime"] = (
        pvps["TxTime"]
        + 2.0 * xmlhelp.load(".//{*}SlantRange") / _constants.speed_of_light
    )
    pvps["RcvPos"] = np.polynomial.polynomial.polyval(pvps["RcvTime"], arppoly).T
    pvps["RcvVel"] = np.polynomial.polynomial.polyval(
        pvps["RcvTime"], np.polynomial.polynomial.polyder(arppoly)
    ).T

    srp = xmlhelp.load(".//{*}SRP/{*}ECF")
    pvps["SRPPos"] = srp

    tmp_cphd = (
        tmp_path_factory.mktemp("data") / good_cphd_xml_path.with_suffix(".cphd").name
    )
    with open(tmp_cphd, "wb") as f, skcphd.Writer(f, cphd_plan) as cw:
        cw.write_pvp("1", pvps)
        cw.write_signal("1", signal)
    yield tmp_cphd


@pytest.fixture(scope="session")
def example_crsdsar(tmp_path_factory):
    crsd_etree = etree.parse(good_crsd_xml_path)
    xmlhelp = skcrsd.XmlHelper(crsd_etree)

    pvp_dtype = skcrsd.get_pvp_dtype(crsd_etree)

    assert crsd_etree.findtext("./{*}Data/{*}Receive/{*}SignalArrayFormat") == "CI2"
    signal_dtype = skcrsd.binary_format_string_to_dtype(
        crsd_etree.findtext("./{*}Data/{*}Receive/{*}SignalArrayFormat")
    )
    rng = np.random.default_rng(123456)
    num_pulses = xmlhelp.load("./{*}Data/{*}Transmit/{*}TxSequence/{*}NumPulses")
    num_vectors = xmlhelp.load("./{*}Data/{*}Receive/{*}Channel/{*}NumVectors")
    num_samples = xmlhelp.load("./{*}Data/{*}Receive/{*}Channel/{*}NumSamples")
    signal = (
        rng.integers(-128, 127, (num_vectors, num_samples, 2), dtype=np.int8)
        .view(signal_dtype)
        .squeeze()
    )

    pvps = np.zeros((num_vectors), dtype=pvp_dtype)
    ppps = np.zeros(num_pulses, dtype=skcrsd.get_ppp_dtype(crsd_etree))
    tx_ref_time = xmlhelp.load("{*}ReferenceGeometry/{*}TxParameters/{*}Time")
    txtime = np.interp(
        np.arange(num_pulses),
        [
            0,
            xmlhelp.load(".//{*}RefVectorPulseIndex"),
            num_pulses - 1,
        ],
        [
            xmlhelp.load(".//{*}TxTime1"),
            tx_ref_time,
            xmlhelp.load(".//{*}TxTime2"),
        ],
    )
    ppps["TxTime"]["Int"] = np.floor(txtime)
    ppps["TxTime"]["Frac"] = txtime % 1

    txpos = xmlhelp.load("{*}ReferenceGeometry/{*}TxParameters/{*}APCPos")
    txvel = xmlhelp.load("{*}ReferenceGeometry/{*}TxParameters/{*}APCVel")

    tx_pos_poly = np.stack([(txpos - tx_ref_time * txvel), txvel])

    fx1 = xmlhelp.load(".//{*}FxMin")
    fx2 = xmlhelp.load(".//{*}FxMax")
    ppps["FX1"][:] = fx1
    ppps["FX2"][:] = fx2
    ppps["TXmt"][:] = xmlhelp.load(".//{*}TXmtMin")
    ppps["TxRadInt"][:] = xmlhelp.load(".//{*}TxRefRadIntensity")
    ppps["FxRate"][:] = 1e12
    ppps["FxFreq0"][:] = xmlhelp.load(".//{*}FxC")

    ppps["TxPos"] = np.polynomial.polynomial.polyval(txtime, tx_pos_poly).T
    ppps["TxPos"][xmlhelp.load(".//{*}RefVectorPulseIndex")] = txpos
    ppps["TxVel"] = txvel
    ppps["TxACX"][...] = [1, 0, 0]
    ppps["TxACY"][...] = [0, 1, 0]

    rcvstart = np.interp(
        np.arange(num_vectors),
        [
            0,
            xmlhelp.load(".//{*}RefVectorIndex"),
            num_vectors - 1,
        ],
        [
            xmlhelp.load(".//{*}RcvStartTime1"),
            xmlhelp.load("./{*}ReferenceGeometry/{*}RcvParameters/{*}Time"),
            xmlhelp.load(".//{*}RcvStartTime2"),
        ],
    )
    fs = xmlhelp.load("{*}Channel/{*}Parameters/{*}Fs")
    rcvstart = np.round((rcvstart - rcvstart[0]) * fs) / fs + rcvstart[0]
    pvps["RcvStart"]["Int"] = np.floor(rcvstart)
    pvps["RcvStart"]["Frac"] = rcvstart % 1

    rcvpos = xmlhelp.load("{*}ReferenceGeometry/{*}RcvParameters/{*}APCPos")
    rcvvel = xmlhelp.load("{*}ReferenceGeometry/{*}RcvParameters/{*}APCVel")
    rcv_ref_time = xmlhelp.load("{*}ReferenceGeometry/{*}RcvParameters/{*}Time")

    rcv_pos_poly = np.stack([(rcvpos - rcv_ref_time * rcvvel), rcvvel])
    pvps["RcvPos"] = np.polynomial.polynomial.polyval(rcvstart, rcv_pos_poly).T
    pvps["RcvPos"][xmlhelp.load(".//{*}RefVectorIndex")] = rcvpos
    pvps["RcvVel"] = rcvvel
    pvps["SIGNAL"] = 1
    pvps["RefFreq"] = xmlhelp.load("{*}Channel/{*}Parameters/{*}F0Ref")
    pvps["TxPulseIndex"] = np.arange(pvps.size)
    pvps["FRCV1"] = xmlhelp.load(".//{*}FrcvMin")
    pvps["FRCV2"] = xmlhelp.load(".//{*}FrcvMax")
    pvps["AmpSF"] = 1.0
    pvps["RcvACX"][...] = [1, 0, 0]
    pvps["RcvACY"][...] = [0, 1, 0]
    pvps["DFIC0"][1] = -10
    pvps["FICRate"][1] = 10

    tmp_crsd = (
        tmp_path_factory.mktemp("data") / good_crsd_xml_path.with_suffix(".crsd").name
    )
    sequence_id = crsd_etree.findtext("{*}TxSequence/{*}Parameters/{*}Identifier")
    channel_id = crsd_etree.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    new_meta = skcrsd.Metadata(
        xmltree=crsd_etree,
    )
    with open(tmp_crsd, "wb") as f, skcrsd.Writer(f, new_meta) as cw:
        cw.write_ppp(sequence_id, ppps)
        cw.write_pvp(channel_id, pvps)
        cw.write_signal(channel_id, signal)
    yield tmp_crsd


def _remove(root, pattern):
    if (elem := root.find(pattern)) is not None:
        elem.getparent().remove(elem)
    else:
        print(f"Cannot find {pattern=}")


def _replace_error(crsd_etree, sensor_type):
    sar_error = crsd_etree.find("{*}ErrorParameters/{*}SARImage")
    elem_ns = etree.QName(sar_error).namespace
    retval = copy.deepcopy(sar_error.find("{*}Monostatic"))
    retval.tag = f"{{{elem_ns}}}{sensor_type}Sensor"
    sar_error.addnext(retval)
    helper = skcrsd.XmlHelper(crsd_etree)
    ndx = {"Tx": 0, "Rcv": 1}[sensor_type]
    helper.set_elem(
        retval.find(".//{*}TimeFreqCov"),
        skcrsd.MtxType((3, 3)).parse_elem(retval.find(".//{*}TimeFreqCov"))[
            [ndx, 2], :
        ][:, [ndx, 2]],
    )
    time_decorr = copy.deepcopy(retval.find(f".//{{*}}{{{sensor_type}}}TimeDecorr"))
    if time_decorr is not None:
        time_decorr.tag = f"{{{elem_ns}}}TimeDecorr"
        _remove(retval, "{*}TxTimeDecorr")
        _remove(retval, "{*}RcvTimeDecorr")
        retval.find(".//{*}ClockFreqDecorr").addprevious(time_decorr)
    sar_error.getparent().remove(sar_error)


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


@pytest.fixture(scope="session")
def example_crsdtx(tmp_path_factory, example_crsdsar):
    with example_crsdsar.open("rb") as f, skcrsd.Reader(f) as cr:
        crsd_etree = cr.metadata.xmltree
        sequence_id = crsd_etree.findtext("{*}TxSequence/{*}Parameters/{*}Identifier")
        ppps = cr.read_ppps(sequence_id)
    crsd_etree.find(".//{*}RefPulseIndex").text = crsd_etree.find(
        ".//{*}RefVectorPulseIndex"
    ).text
    ns = etree.QName(crsd_etree.getroot()).namespace
    crsd_etree.getroot().tag = f"{{{ns}}}CRSDtx"
    _remove(crsd_etree, "{*}SARInfo")
    _remove(crsd_etree, "{*}ReceiveInfo")
    _remove(crsd_etree, "{*}Global/{*}Receive")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ExtendedArea")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ImageGrid")
    _remove(crsd_etree, "{*}Data/{*}Receive")
    _remove(crsd_etree, "{*}Channel")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}SARImage")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}RcvParameters")
    _remove(crsd_etree, "{*}DwellPolynomials")
    _remove(crsd_etree, "{*}PVP")
    _replace_error(crsd_etree, "Tx")
    tmp_crsd = (
        tmp_path_factory.mktemp("data") / good_crsd_xml_path.with_suffix(".crsd").name
    )

    new_meta = skcrsd.Metadata(
        xmltree=crsd_etree,
    )
    with open(tmp_crsd, "wb") as f, skcrsd.Writer(f, new_meta) as cw:
        cw.write_ppp(sequence_id, ppps)
    yield tmp_crsd


@pytest.fixture(scope="session")
def example_crsdrcv(tmp_path_factory, example_crsdsar):
    with example_crsdsar.open("rb") as f, skcrsd.Reader(f) as cr:
        crsd_etree = cr.metadata.xmltree
        channel_id = crsd_etree.findtext("{*}Channel/{*}Parameters/{*}Identifier")
        pvps = cr.read_pvps(channel_id)
        signal = cr.read_signal(channel_id)
    ns = etree.QName(crsd_etree.getroot()).namespace
    crsd_etree.getroot().tag = f"{{{ns}}}CRSDrcv"
    _remove(crsd_etree, "{*}SARInfo")
    _remove(crsd_etree, "{*}TransmitInfo")
    _remove(crsd_etree, "{*}Global/{*}Transmit")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ExtendedArea")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ImageGrid")
    _remove(crsd_etree, "{*}Data/{*}Transmit")
    _remove(crsd_etree, "{*}TxSequence")
    _remove(crsd_etree, "{*}Channel/{*}Parameters/{*}SARImage")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}SARImage")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}TxParameters")
    _remove(crsd_etree, "{*}DwellPolynomials")
    fx_ids = [
        x.text
        for x in crsd_etree.findall("{*}SupportArray/{*}FxResponseArray/{*}Identifier")
    ]
    xm_ids = [
        x.text for x in crsd_etree.findall("{*}SupportArray/{*}XMArray/{*}Identifier")
    ]
    _remove(crsd_etree, "{*}SupportArray/{*}FxResponseArray")
    _remove(crsd_etree, "{*}SupportArray/{*}XMArray")
    for x in fx_ids + xm_ids:
        _remove(
            crsd_etree,
            f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{x}']",
        )
    nsa = crsd_etree.find("{*}Data/{*}Support/{*}NumSupportArrays")
    nsa.text = str(int(nsa.text) - len(fx_ids + xm_ids))
    _repack_support_arrays(crsd_etree)
    _remove(crsd_etree, "{*}PPP")
    tx_pulse_index_offset = int(crsd_etree.findtext("{*}PVP/{*}TxPulseIndex/{*}Offset"))
    _remove(crsd_etree, "{*}PVP/{*}TxPulseIndex")
    for pvp_offset in crsd_etree.findall("{*}PVP/*/{*}Offset"):
        if int(pvp_offset.text) > tx_pulse_index_offset:
            pvp_offset.text = str(int(pvp_offset.text) - 8)
    crsd_etree.find("{*}Data/{*}Receive/{*}NumBytesPVP").text = str(
        int(crsd_etree.findtext("{*}Data/{*}Receive/{*}NumBytesPVP")) - 8
    )
    _replace_error(crsd_etree, "Rcv")
    new_pvp_dtype = skcrsd.get_pvp_dtype(crsd_etree)
    new_pvps = np.zeros(pvps.shape, new_pvp_dtype)
    for field in new_pvp_dtype.fields:
        new_pvps[field] = pvps[field]
    tmp_crsd = (
        tmp_path_factory.mktemp("data") / good_crsd_xml_path.with_suffix(".crsd").name
    )

    new_meta = skcrsd.Metadata(
        xmltree=crsd_etree,
    )
    with open(tmp_crsd, "wb") as f, skcrsd.Writer(f, new_meta) as cw:
        cw.write_pvp(channel_id, new_pvps)
        cw.write_signal(channel_id, signal)
    yield tmp_crsd


@pytest.fixture(scope="session")
def example_sicd(tmp_path_factory):
    sicd_etree = etree.parse(good_sicd_xml_path)
    tmp_sicd = (
        tmp_path_factory.mktemp("data") / good_sicd_xml_path.with_suffix(".sicd").name
    )
    sec = {"security": {"clas": "U"}}
    sicd_meta = sksicd.NitfMetadata(
        xmltree=sicd_etree,
        file_header_part={"ostaid": "nowhere"} | sec,
        im_subheader_part={"isorce": "this sensor"} | sec,
        de_subheader_part=sec,
    )
    with open(tmp_sicd, "wb") as f, sksicd.NitfWriter(f, sicd_meta):
        pass  # don't currently care about the pixels
    yield tmp_sicd


@pytest.fixture(scope="session")
def example_sidd(tmp_path_factory):
    sidd_etree = etree.parse(good_sidd_xml_path)
    tmp_sidd = (
        tmp_path_factory.mktemp("data") / good_sidd_xml_path.with_suffix(".sidd").name
    )
    sec = {"security": {"clas": "U"}}
    sidd_meta = sksidd.NitfMetadata(
        file_header_part={"ostaid": "nowhere"} | sec,
        images=[
            sksidd.NitfProductImageMetadata(
                xmltree=sidd_etree,
                im_subheader_part=sec,
                de_subheader_part=sec,
            )
        ],
    )
    with tmp_sidd.open("wb") as f, sksidd.NitfWriter(f, sidd_meta):
        pass  # don't currently care about the pixels
    yield tmp_sidd
