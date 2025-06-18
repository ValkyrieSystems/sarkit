import pathlib

import lxml.etree
import numpy as np

import sarkit.crsd as skcrsd

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


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
    rng = np.random.default_rng()

    def _random_array(shape, dtype, reshape=True):
        retval = np.frombuffer(
            rng.bytes(np.prod(shape) * dtype.itemsize), dtype=dtype
        ).copy()
        if dtype.names is None:
            retval[~np.isfinite(retval)] = 0
        else:
            for name in dtype.names:
                retval[name][~np.isfinite(retval[name])] = 0
        return retval.reshape(shape) if reshape else retval

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
        nrows = xmlhelp.load_elem(data_sa_elem.find("./{*}NumRows"))
        ncols = xmlhelp.load_elem(data_sa_elem.find("./{*}NumCols"))
        format_str = basis_etree.findtext(
            f"./{{*}}SupportArray//{{*}}Identifier[.='{sa_id}']/../{{*}}ElementFormat"
        )
        dt = skcrsd.binary_format_string_to_dtype(format_str)
        support_arrays[sa_id] = _random_array((nrows, ncols), dt)

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
