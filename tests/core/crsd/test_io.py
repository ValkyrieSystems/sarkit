import pathlib
import uuid

import lxml.builder
import lxml.etree
import numpy as np
import pytest
import smart_open

import sarkit.crsd as skcrsd
import tests.utils

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def _random_array(shape, dtype, reshape=True):
    rng = np.random.default_rng()
    retval = np.frombuffer(
        rng.bytes(np.prod(shape) * dtype.itemsize), dtype=dtype
    ).copy()

    def _zerofill(arr):
        if arr.dtype.names is None:
            arr[~np.isfinite(arr)] = 0
        else:
            for name in arr.dtype.names:
                _zerofill(arr[name])

    _zerofill(retval)
    return retval.reshape(shape) if reshape else retval


def _random_support_array(crsd_xmltree, sa_id):
    xmlhelp = skcrsd.XmlHelper(crsd_xmltree)
    data_sa_elem = crsd_xmltree.find(
        f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{sa_id}']"
    )
    sa_id = xmlhelp.load_elem(data_sa_elem.find("./{*}SAId"))
    nrows = xmlhelp.load_elem(data_sa_elem.find("./{*}NumRows"))
    ncols = xmlhelp.load_elem(data_sa_elem.find("./{*}NumCols"))
    sa_elem = crsd_xmltree.find(f"./{{*}}SupportArray/*[{{*}}Identifier='{sa_id}']")
    format_str = sa_elem.findtext("{*}ElementFormat")
    return _random_array(
        (nrows, ncols), skcrsd.binary_format_string_to_dtype(format_str)
    )


def test_roundtrip(tmp_path, caplog):
    basis_etree = lxml.etree.parse(DATAPATH / "example-crsd-1.0.xml")
    basis_version = lxml.etree.QName(basis_etree.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=skcrsd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(basis_etree)
    xmlhelp = skcrsd.XmlHelper(basis_etree)
    channel_ids = [
        x.text for x in basis_etree.findall("./{*}Channel/{*}Parameters/{*}Identifier")
    ]
    assert len(channel_ids) == 1

    sequence_ids = [
        x.text
        for x in basis_etree.findall("./{*}TxSequence/{*}Parameters/{*}Identifier")
    ]
    assert len(sequence_ids) == 1

    signal_dtype = skcrsd.binary_format_string_to_dtype(
        basis_etree.findtext("./{*}Data/{*}Receive/{*}SignalArrayFormat")
    )
    num_pulses = xmlhelp.load("./{*}Data/{*}Transmit/{*}TxSequence/{*}NumPulses")
    num_vectors = xmlhelp.load("./{*}Data/{*}Receive/{*}Channel/{*}NumVectors")
    num_samples = xmlhelp.load("./{*}Data/{*}Receive/{*}Channel/{*}NumSamples")
    basis_signal = _random_array((num_vectors, num_samples), signal_dtype)

    pvps = np.zeros(num_vectors, dtype=skcrsd.get_pvp_dtype(basis_etree))
    for f, (dt, _) in pvps.dtype.fields.items():
        pvps[f] = _random_array(num_vectors, dtype=dt, reshape=False)

    ppps = np.zeros(num_pulses, dtype=skcrsd.get_ppp_dtype(basis_etree))
    for f, (dt, _) in ppps.dtype.fields.items():
        ppps[f] = _random_array(num_pulses, dtype=dt, reshape=False)

    support_arrays = {}
    for data_sa_elem in basis_etree.findall("./{*}Data/{*}Support/{*}SupportArray"):
        sa_id = xmlhelp.load_elem(data_sa_elem.find("./{*}SAId"))
        support_arrays[sa_id] = _random_support_array(basis_etree, sa_id)

    crsd_metadata = skcrsd.Metadata(
        file_header_part=skcrsd.FileHeaderPart(
            additional_kvps={"k1": "v1", "k2": "v2"},
        ),
        xmltree=basis_etree,
    )
    out_crsd = tmp_path / "out.crsd"
    with open(out_crsd, "wb") as f:
        with skcrsd.Writer(f, crsd_metadata) as writer:
            writer.write_signal(channel_ids[0], basis_signal)
            writer.write_pvp(channel_ids[0], pvps)
            for k, v in support_arrays.items():
                writer.write_support_array(k, v)
            writer.write_ppp(sequence_ids[0], ppps)

    with open(out_crsd, "rb") as f, skcrsd.Reader(f) as reader:
        read_sig, read_pvp = reader.read_channel(channel_ids[0])
        read_support_arrays = {}
        for sa_id in reader.metadata.xmltree.findall(
            "./{*}SupportArray/*/{*}Identifier"
        ):
            read_support_arrays[sa_id.text] = reader.read_support_array(sa_id.text)
        read_ppp = reader.read_ppps(sequence_ids[0])

    assert crsd_metadata.file_header_part == reader.metadata.file_header_part
    assert np.array_equal(basis_signal, read_sig)
    assert np.array_equal(pvps, read_pvp)
    assert np.array_equal(ppps, read_ppp)
    assert support_arrays.keys() == read_support_arrays.keys()
    assert all(
        np.array_equal(support_arrays[f], read_support_arrays[f])
        for f in support_arrays
    )
    assert lxml.etree.tostring(
        reader.metadata.xmltree, method="c14n"
    ) == lxml.etree.tostring(basis_etree, method="c14n")
    assert not caplog.text


def test_roundtrip_compressed(tmp_path):
    basis_etree = lxml.etree.parse(DATAPATH / "example-crsd-1.0.xml")
    basis_version = lxml.etree.QName(basis_etree.getroot()).namespace
    data_rcv = basis_etree.find("{*}Data/{*}Receive")
    assert data_rcv is not None
    assert data_rcv.find("{*}SignalCompression") is None
    channel_ids = [
        x.text for x in basis_etree.findall("./{*}Channel/{*}Parameters/{*}Identifier")
    ]
    assert len(channel_ids) == 1
    ch_id = channel_ids[0]

    em = lxml.builder.ElementMaker(namespace=basis_version, nsmap={None: basis_version})
    signal_block_str = "the identifier is the signal block!"
    data_rcv.find("{*}NumCRSDChannels").addnext(
        em.SignalCompression(
            em.Identifier(signal_block_str),
            em.CompressedSignalSize(str(len(signal_block_str))),
        )
    )
    schema = lxml.etree.XMLSchema(file=skcrsd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(basis_etree)
    xmlhelp = skcrsd.XmlHelper(basis_etree)
    num_vectors = int(data_rcv.findtext("{*}Channel/{*}NumVectors"))
    pvps = np.zeros(num_vectors, dtype=skcrsd.get_pvp_dtype(basis_etree))

    support_arrays = {}
    for data_sa_elem in basis_etree.findall("./{*}Data/{*}SupportArray"):
        sa_id = xmlhelp.load_elem(data_sa_elem.find("./{*}Identifier"))
        support_arrays[sa_id] = _random_support_array(basis_etree, sa_id)

    meta = skcrsd.Metadata(
        xmltree=basis_etree,
    )
    out_crsd = tmp_path / "out.crsd"
    with open(out_crsd, "wb") as f, skcrsd.Writer(f, meta) as writer:
        with pytest.raises(RuntimeError, match="Signal block is compressed.*"):
            writer.write_signal(
                ch_id, np.frombuffer(signal_block_str.encode(), dtype=np.uint8)
            )

        writer.write_signal_compressed(
            np.frombuffer(signal_block_str.encode(), dtype=np.uint8)
        )
        writer.write_pvp(ch_id, pvps)
        for k, v in support_arrays.items():
            writer.write_support_array(k, v)

    with open(out_crsd, "rb") as f, skcrsd.Reader(f) as reader:
        with pytest.raises(RuntimeError, match="Signal block is compressed.*"):
            reader.read_signal(ch_id)

        sig_array = reader.read_signal_compressed()

    assert sig_array.tobytes().decode() == signal_block_str


@pytest.mark.parametrize("is_masked", (True, False))
@pytest.mark.parametrize("nodata_in_xml", (True, False))
def test_write_support_array(is_masked, nodata_in_xml, tmp_path):
    basis_etree = lxml.etree.parse(DATAPATH / "example-crsd-1.0.xml")
    elem_ns = lxml.etree.QName(basis_etree.getroot()).namespace
    em = lxml.builder.ElementMaker(namespace=elem_ns, nsmap={None: elem_ns})
    sa_id = str(uuid.uuid4())
    sa_elem = em.AddedSupportArray(
        em.Identifier(sa_id),
        em.ElementFormat("a=CI4;b=CI4;"),
        em.X0("0.1"),
        em.Y0("0.2"),
        em.XSS("0.3"),
        em.YSS("0.4"),
        em.NODATA(),  # placeholder
        em.XUnits(""),
        em.YUnits(""),
        em.ZUnits(""),
    )
    data_sa_elem = em.SupportArray(
        em.SAId(sa_id),
        em.NumRows("24"),
        em.NumCols("8"),
        em.BytesPerElement(
            str(skcrsd.binary_format_string_to_dtype("a=CI4;b=CI4;").itemsize)
        ),
        em.ArrayByteOffset(
            str(
                max(
                    (
                        int(x.findtext("{*}ArrayByteOffset"))
                        + int(x.findtext("{*}NumRows"))
                        * int(x.findtext("{*}NumCols"))
                        * int(x.findtext("{*}BytesPerElement"))
                        for x in basis_etree.findall("{*}Data/{*}SupportArray")
                    ),
                    default=0,
                )
            )
        ),
    )
    basis_etree.find("{*}Data/{*}Support").append(data_sa_elem)
    basis_etree.find("{*}SupportArray").append(sa_elem)
    basis_array = _random_support_array(basis_etree, sa_id)
    basis_array = basis_array.astype(basis_array.dtype.newbyteorder(">"))

    nodata_elem = sa_elem.find("{*}NODATA")
    nodata_hex_str = basis_array.flat[0].tobytes().hex()
    nodata_elem.text = nodata_hex_str
    if not nodata_in_xml:
        sa_elem.remove(nodata_elem)

    mx = skcrsd.mask_support_array(basis_array, nodata_hex_str)
    if not is_masked:
        mx = mx.filled(0)

    meta = skcrsd.Metadata(
        xmltree=basis_etree,
    )
    out_crsd = tmp_path / "out.crsd"
    with open(out_crsd, "wb") as f, skcrsd.Writer(f, meta) as writer:
        if is_masked and not nodata_in_xml:
            with pytest.raises(ValueError, match="nodata.*does not match.*"):
                writer.write_support_array(sa_id, mx)
            return
        writer.write_support_array(sa_id, mx)

    with open(out_crsd, "rb") as f, skcrsd.Reader(f) as reader:
        read_sa = reader.read_support_array(sa_id)
        assert np.array_equal(mx, read_sa)


def test_remote_read(example_crsdsar):
    with tests.utils.static_http_server(example_crsdsar.parent) as server_url:
        with smart_open.open(
            f"{server_url}/{example_crsdsar.name}", mode="rb"
        ) as file_object:
            with skcrsd.Reader(file_object) as r:
                ch_id = r.metadata.xmltree.findtext(
                    "{*}Channel/{*}Parameters/{*}Identifier"
                )
                _, _ = r.read_channel(ch_id)
                seq_id = r.metadata.xmltree.findtext(
                    "{*}TxSequence/{*}Parameters/{*}Identifier"
                )
                _ = r.read_ppps(seq_id)
