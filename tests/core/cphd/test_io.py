import pathlib
import uuid

import lxml.builder
import lxml.etree
import numpy as np
import pytest
import smart_open

import sarkit.cphd as skcphd
import tests.utils

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def test_version_info():
    actual_order = [x["version"] for x in skcphd.VERSION_INFO.values()]
    expected_order = sorted(actual_order, key=lambda x: x.split("."))
    assert actual_order == expected_order

    for urn, info in skcphd.VERSION_INFO.items():
        assert lxml.etree.parse(info["schema"]).getroot().get("targetNamespace") == urn


dtype_binary_mapping = [
    (np.int8, "I1"),
    (np.int16, "I2"),
    (np.int32, "I4"),
    (np.int64, "I8"),
    (np.uint8, "U1"),
    (np.uint16, "U2"),
    (np.uint32, "U4"),
    (np.uint64, "U8"),
    (np.float32, "F4"),
    (np.float64, "F8"),
    (np.complex64, "CF8"),
    (np.complex128, "CF16"),
    (np.dtype([("real", np.int8), ("imag", np.int8)]), "real=I1;imag=I1;"),
    (np.dtype([("real", np.int16), ("imag", np.int16)]), "real=I2;imag=I2;"),
    (np.dtype([("real", np.int32), ("imag", np.int32)]), "real=I4;imag=I4;"),
    (np.dtype([("I", np.int64), ("Q", np.int64)]), "I=I8;Q=I8;"),
    (np.dtype("S30"), "S30"),
    (np.dtype(("f8", 2)), "DCX=F8;DCY=F8;"),
    (np.dtype(("f8", 3)), "X=F8;Y=F8;Z=F8;"),
    (np.dtype([("a", "i8"), ("b", "f8"), ("c", "f8")]), "a=I8;b=F8;c=F8;"),
]


@pytest.mark.parametrize("dtype, format_str", dtype_binary_mapping)
def test_dtype_binary_format(dtype, format_str):
    assert skcphd.dtype_to_binary_format_string(dtype) == format_str
    assert skcphd.binary_format_string_to_dtype(format_str) == dtype


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


def _random_support_array(cphd_xmltree, sa_id):
    xmlhelp = skcphd.XmlHelper(cphd_xmltree)
    data_sa_elem = cphd_xmltree.find(
        f"{{*}}Data/{{*}}SupportArray[{{*}}Identifier='{sa_id}']"
    )
    sa_id = xmlhelp.load_elem(data_sa_elem.find("./{*}Identifier"))
    nrows = xmlhelp.load_elem(data_sa_elem.find("./{*}NumRows"))
    ncols = xmlhelp.load_elem(data_sa_elem.find("./{*}NumCols"))
    sa_elem = cphd_xmltree.find(f"./{{*}}SupportArray/*[{{*}}Identifier='{sa_id}']")
    format_str = sa_elem.findtext("{*}ElementFormat")
    return _random_array(
        (nrows, ncols), skcphd.binary_format_string_to_dtype(format_str)
    )


@pytest.mark.parametrize("with_support_block", (True, False))
def test_roundtrip(tmp_path, with_support_block):
    basis_etree = lxml.etree.parse(DATAPATH / "example-cphd-1.0.1.xml")
    basis_version = lxml.etree.QName(basis_etree.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=skcphd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(basis_etree)
    xmlhelp = skcphd.XmlHelper(basis_etree)
    channel_ids = [
        x.text for x in basis_etree.findall("./{*}Channel/{*}Parameters/{*}Identifier")
    ]
    assert len(channel_ids) == 1

    signal_dtype = skcphd.binary_format_string_to_dtype(
        basis_etree.findtext("./{*}Data/{*}SignalArrayFormat")
    )
    num_vectors = xmlhelp.load("./{*}Data/{*}Channel/{*}NumVectors")
    num_samples = xmlhelp.load(".//{*}Data/{*}Channel/{*}NumSamples")
    basis_signal = _random_array((num_vectors, num_samples), signal_dtype)

    pvps = np.zeros(num_vectors, dtype=skcphd.get_pvp_dtype(basis_etree))
    for f, (dt, _) in pvps.dtype.fields.items():
        pvps[f] = _random_array(num_vectors, dtype=dt, reshape=False)

    assert basis_etree.find("{*}Data/{*}SupportArray") is not None
    if not with_support_block:
        for sa in basis_etree.findall(".//{*}SupportArray"):
            sa.getparent().remove(sa)

    support_arrays = {}
    for data_sa_elem in basis_etree.findall("./{*}Data/{*}SupportArray"):
        sa_id = xmlhelp.load_elem(data_sa_elem.find("./{*}Identifier"))
        support_arrays[sa_id] = _random_support_array(basis_etree, sa_id)

    cphd_metadata = skcphd.Metadata(
        file_header_part=skcphd.FileHeaderPart(
            additional_kvps={"k1": "v1", "k2": "v2"},
        ),
        xmltree=basis_etree,
    )
    out_cphd = tmp_path / "out.cphd"
    with open(out_cphd, "wb") as f:
        with skcphd.Writer(f, cphd_metadata) as writer:
            writer.write_signal(channel_ids[0], basis_signal)
            writer.write_pvp(channel_ids[0], pvps)
            for k, v in support_arrays.items():
                writer.write_support_array(k, v)

    with open(out_cphd, "rb") as f, skcphd.Reader(f) as reader:
        read_sig, read_pvp = reader.read_channel(channel_ids[0])
        read_support_arrays = {}
        for sa_id in reader.metadata.xmltree.findall(
            "./{*}SupportArray/*/{*}Identifier"
        ):
            read_support_arrays[sa_id.text] = reader.read_support_array(sa_id.text)

    assert cphd_metadata.file_header_part == reader.metadata.file_header_part
    assert np.array_equal(basis_signal, read_sig)
    assert np.array_equal(pvps, read_pvp)
    assert support_arrays.keys() == read_support_arrays.keys()
    assert all(
        np.array_equal(support_arrays[f], read_support_arrays[f])
        for f in support_arrays
    )
    assert lxml.etree.tostring(
        reader.metadata.xmltree, method="c14n"
    ) == lxml.etree.tostring(basis_etree, method="c14n")


def test_roundtrip_compressed(tmp_path):
    basis_etree = lxml.etree.parse(DATAPATH / "example-cphd-1.0.1.xml")
    basis_version = lxml.etree.QName(basis_etree.getroot()).namespace
    assert basis_etree.find("{*}Data/{*}SignalCompressionID") is None
    channel_ids = [
        x.text for x in basis_etree.findall("./{*}Channel/{*}Parameters/{*}Identifier")
    ]
    assert len(channel_ids) == 1
    ch_id = channel_ids[0]

    em = lxml.builder.ElementMaker(namespace=basis_version, nsmap={None: basis_version})
    data_chan_elem = basis_etree.find("{*}Data/{*}Channel")
    data_chan_elem.addprevious(
        em.SignalCompressionID("Channel identifier but as bytes!")
    )
    data_chan_elem.append(em.CompressedSignalSize(str(len(ch_id.encode()))))

    schema = lxml.etree.XMLSchema(file=skcphd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(basis_etree)
    xmlhelp = skcphd.XmlHelper(basis_etree)
    num_vectors = xmlhelp.load("./{*}Data/{*}Channel/{*}NumVectors")
    pvps = np.zeros(num_vectors, dtype=skcphd.get_pvp_dtype(basis_etree))

    support_arrays = {}
    for data_sa_elem in basis_etree.findall("./{*}Data/{*}SupportArray"):
        sa_id = xmlhelp.load_elem(data_sa_elem.find("./{*}Identifier"))
        support_arrays[sa_id] = _random_support_array(basis_etree, sa_id)

    meta = skcphd.Metadata(
        xmltree=basis_etree,
    )
    out_cphd = tmp_path / "out.cphd"
    with open(out_cphd, "wb") as f, skcphd.Writer(f, meta) as writer:
        writer.write_signal(ch_id, np.frombuffer(ch_id.encode(), dtype=np.uint8))
        writer.write_pvp(ch_id, pvps)
        for k, v in support_arrays.items():
            writer.write_support_array(k, v)

    with open(out_cphd, "rb") as f, skcphd.Reader(f) as reader:
        sig_array = reader.read_signal(ch_id)

    assert sig_array.tobytes().decode() == ch_id


@pytest.mark.parametrize("is_masked", (True, False))
@pytest.mark.parametrize("nodata_in_xml", (True, False))
def test_write_support_array(is_masked, nodata_in_xml, tmp_path):
    basis_etree = lxml.etree.parse(DATAPATH / "example-cphd-1.0.1.xml")
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
        em.Identifier(sa_id),
        em.NumRows("24"),
        em.NumCols("8"),
        em.BytesPerElement(
            str(skcphd.binary_format_string_to_dtype("a=CI4;b=CI4;").itemsize)
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
    basis_etree.find("{*}Data").append(data_sa_elem)
    basis_etree.find("{*}SupportArray").append(sa_elem)
    basis_array = _random_support_array(basis_etree, sa_id)
    basis_array = basis_array.astype(basis_array.dtype.newbyteorder(">"))

    nodata_elem = sa_elem.find("{*}NODATA")
    nodata_hex_str = basis_array.flat[0].tobytes().hex()
    nodata_elem.text = nodata_hex_str
    if not nodata_in_xml:
        sa_elem.remove(nodata_elem)

    mx = skcphd.mask_support_array(basis_array, nodata_hex_str)
    if not is_masked:
        mx = mx.filled(0)

    cphd_plan = skcphd.Metadata(
        xmltree=basis_etree,
    )
    out_cphd = tmp_path / "out.cphd"
    with open(out_cphd, "wb") as f, skcphd.Writer(f, cphd_plan) as writer:
        if is_masked and not nodata_in_xml:
            with pytest.raises(ValueError, match="nodata.*does not match.*"):
                writer.write_support_array(sa_id, mx)
            return
        writer.write_support_array(sa_id, mx)

    with open(out_cphd, "rb") as f, skcphd.Reader(f) as reader:
        read_sa = reader.read_support_array(sa_id)
        assert np.array_equal(mx, read_sa)


INDICES_TO_CHECK = [None, 0, 7, 9, -9, -7, -1_000_000_000, 1_000_000_000_000]


@pytest.mark.parametrize("start", INDICES_TO_CHECK)
@pytest.mark.parametrize("stop", INDICES_TO_CHECK)
def test_read_partial(example_cphd, start, stop):
    with open(example_cphd, "rb") as file, skcphd.Reader(file) as reader:
        ch_id = reader.metadata.xmltree.findtext("{*}Data/{*}Channel/{*}Identifier")
        all_signal, all_pvp = reader.read_channel(ch_id)
        kwargs = {}

        if start is not None:
            kwargs["start_vector"] = start

        if stop is not None:
            kwargs["stop_vector"] = stop

        partial_signal, partial_pvp = reader.read_channel(ch_id, **kwargs)

    np.testing.assert_array_equal(all_signal[start:stop, :], partial_signal)
    np.testing.assert_array_equal(all_pvp[start:stop], partial_pvp)


def test_remote_read(example_cphd):
    with tests.utils.static_http_server(example_cphd.parent) as server_url:
        with smart_open.open(
            f"{server_url}/{example_cphd.name}", mode="rb"
        ) as file_object:
            with skcphd.Reader(file_object) as r:
                ch_id = r.metadata.xmltree.findtext("{*}Data/{*}Channel/{*}Identifier")
                _, _ = r.read_channel(ch_id)
