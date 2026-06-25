import pathlib

import lxml.etree
import numpy as np
from numpy.lib import recfunctions as rfn

import sarkit.crsd as skcrsd

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def _last_field(structured_dtype):
    dtype, offset = sorted(
        ((dtype, offset) for (dtype, offset) in structured_dtype.fields.values()),
        key=lambda x: x[1],
    )[-1]

    return dtype, offset


def test_get_ppp_dtype():
    etree = lxml.etree.parse(DATAPATH / "example-crsd-1.0.xml")
    num_bytes_ppp = int(etree.find("./{*}Data/{*}Transmit/{*}NumBytesPPP").text)
    ppp_dtype = skcrsd.get_ppp_dtype(etree)

    dtype, offset = _last_field(ppp_dtype)
    assert ppp_dtype.itemsize == dtype.itemsize + offset  # example has no end pad

    end_pad = 10
    num_bytes_ppp += end_pad
    etree.find("./{*}Data/{*}Transmit/{*}NumBytesPPP").text = str(num_bytes_ppp)
    ppp_dtype2 = skcrsd.get_ppp_dtype(etree)
    dtype, offset = _last_field(ppp_dtype)
    assert ppp_dtype2.itemsize == dtype.itemsize + offset + end_pad


def test_get_pvp_dtype():
    etree = lxml.etree.parse(DATAPATH / "example-crsd-1.0.xml")
    num_bytes_pvp = int(etree.find("./{*}Data/{*}Receive/{*}NumBytesPVP").text)
    pvp_dtype = skcrsd.get_pvp_dtype(etree)

    dtype, offset = _last_field(pvp_dtype)
    assert pvp_dtype.itemsize == dtype.itemsize + offset  # example has no end pad

    end_pad = 10
    num_bytes_pvp += end_pad
    etree.find("./{*}Data/{*}Receive/{*}NumBytesPVP").text = str(num_bytes_pvp)
    pvp_dtype2 = skcrsd.get_pvp_dtype(etree)
    dtype, offset = _last_field(pvp_dtype)
    assert pvp_dtype2.itemsize == dtype.itemsize + offset + end_pad


def test_ppp_dtype_element_roundtrip():
    basis_etree = lxml.etree.parse(DATAPATH / "example-crsd-1.0.xml")
    basis_version = lxml.etree.QName(basis_etree.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=skcrsd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(basis_etree)

    ew = skcrsd.ElementWrapper(basis_etree.getroot())
    del ew["PPP"]
    assert not schema.validate(basis_etree)

    dtype = skcrsd.get_defined_ppp_dtype(basis_version)
    ew["PPP"] = skcrsd.dtype_to_ppp_element(basis_version, dtype)
    schema.assertValid(basis_etree)

    fields = dtype.fields.copy()
    fields["NewField"] = (skcrsd.binary_format_string_to_dtype("S24"), 0)
    new_dtype = rfn.repack_fields(np.dtype(fields))
    ew["PPP"] = skcrsd.dtype_to_ppp_element(basis_version, new_dtype)
    assert ew["PPP"].find("AddedPPP", Name="NewField") is not None
    schema.assertValid(basis_etree)


def test_pvp_dtype_element_roundtrip():
    basis_etree = lxml.etree.parse(DATAPATH / "example-crsd-1.0.xml")
    basis_version = lxml.etree.QName(basis_etree.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=skcrsd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(basis_etree)

    ew = skcrsd.ElementWrapper(basis_etree.getroot())
    del ew["PVP"]
    assert not schema.validate(basis_etree)

    dtype = skcrsd.get_defined_pvp_dtype(basis_version)
    ew["PVP"] = skcrsd.dtype_to_pvp_element(basis_version, dtype)
    schema.assertValid(basis_etree)

    fields = dtype.fields.copy()
    fields["NewField"] = (skcrsd.binary_format_string_to_dtype("S24"), 0)
    new_dtype = rfn.repack_fields(np.dtype(fields))
    ew["PVP"] = skcrsd.dtype_to_pvp_element(basis_version, new_dtype)
    assert ew["PVP"].find("AddedPVP", Name="NewField") is not None
    schema.assertValid(basis_etree)
