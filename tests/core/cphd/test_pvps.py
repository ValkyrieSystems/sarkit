import pathlib

import lxml.etree
import numpy as np
import pytest
from numpy.lib import recfunctions as rfn

import sarkit.cphd as skcphd

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def _last_field(structured_dtype):
    dtype, offset = sorted(
        ((dtype, offset) for (dtype, offset) in structured_dtype.fields.values()),
        key=lambda x: x[1],
    )[-1]

    return dtype, offset


def test_get_pvp_dtype():
    etree = lxml.etree.parse(DATAPATH / "example-cphd-1.1.0.xml")
    num_bytes_pvp = int(etree.find("./{*}Data/{*}NumBytesPVP").text)
    pvp_dtype = skcphd.get_pvp_dtype(etree)

    dtype, offset = _last_field(pvp_dtype)
    assert pvp_dtype.itemsize == dtype.itemsize + offset  # example has no end pad

    end_pad = 10
    num_bytes_pvp += end_pad
    etree.find("./{*}Data/{*}NumBytesPVP").text = str(num_bytes_pvp)
    pvp_dtype2 = skcphd.get_pvp_dtype(etree)
    dtype, offset = _last_field(pvp_dtype)
    assert pvp_dtype2.itemsize == dtype.itemsize + offset + end_pad


@pytest.mark.parametrize(
    "basis_xml",
    [DATAPATH / "example-cphd-1.0.1.xml", DATAPATH / "example-cphd-1.1.0.xml"],
)
def test_pvp_dtype_element_roundtrip(basis_xml):
    basis_etree = lxml.etree.parse(basis_xml)
    basis_version = lxml.etree.QName(basis_etree.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=skcphd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(basis_etree)

    ew = skcphd.ElementWrapper(basis_etree.getroot())
    del ew["PVP"]
    assert not schema.validate(basis_etree)

    dtype = skcphd.get_defined_pvp_dtype(basis_version)
    ew["PVP"] = skcphd.dtype_to_pvp_element(basis_version, dtype)
    schema.assertValid(basis_etree)

    fields = dtype.fields.copy()
    del fields["SIGNAL"]
    del fields["AmpSF"]
    new_dtype = rfn.repack_fields(np.dtype(fields))
    ew["PVP"] = skcphd.dtype_to_pvp_element(basis_version, new_dtype)
    assert "SIGNAL" not in ew["PVP"]
    assert "AmpSF" not in ew["PVP"]
    schema.assertValid(basis_etree)

    fields["NewField"] = (skcphd.binary_format_string_to_dtype("S24"), 0)
    new_dtype = rfn.repack_fields(np.dtype(fields))
    ew["PVP"] = skcphd.dtype_to_pvp_element(basis_version, new_dtype)
    assert ew["PVP"].find("AddedPVP", Name="NewField") is not None
    schema.assertValid(basis_etree)
