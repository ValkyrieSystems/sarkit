import copy
import pathlib
import uuid

import lxml.etree
import numpy as np
import pytest
import scipy.interpolate

import sarkit.crsd as skcrsd

DATAPATH = pathlib.Path(__file__).parents[3] / "data"

good_crsd_xml_path = DATAPATH / "example-crsd-1.0.xml"


def test_compute_dwelltimes():
    crsd_xml = lxml.etree.parse(good_crsd_xml_path)
    assert crsd_xml.find(".//{*}SARImage/{*}DwellTimes/{*}Polynomials") is not None

    dta_id = str(uuid.uuid4())
    elem_format = "COD=F4;DT=F4;"
    dtype = skcrsd.binary_format_string_to_dtype(elem_format)
    dta = np.zeros((24, 8), dtype)
    crsd_with_dta = skcrsd.ElementWrapper(copy.deepcopy(crsd_xml.getroot()))
    for chan_param in crsd_with_dta["Channel"]["Parameters"]:
        chan_param["SARImage"]["DwellTimes"] = {"Array": {"DTAId": dta_id}}
    crsd_with_dta["Data"]["Support"].add(
        "SupportArray",
        {
            "SAId": dta_id,
            "NumRows": dta.shape[0],
            "NumCols": dta.shape[1],
            "BytesPerElement": dtype.itemsize,
            "ArrayByteOffset": 0,
        },  # doesn't matter for this
    )
    crsd_with_dta["SupportArray"].add(
        "DwellTimeArray",
        {
            "Identifier": dta_id,
            "ElementFormat": elem_format,
            "X0": 0,
            "Y0": 0,
            "XSS": 1,
            "YSS": 1,
        },
    )

    ch_id = crsd_xml.findtext("{*}Channel/{*}RefChId")

    # with poly
    skcrsd.compute_dwelltimes_using_poly(ch_id, 0, 0, crsd_xml)
    skcrsd.compute_dwelltimes_using_poly(
        ch_id, np.zeros((2, 4, 5)), np.zeros((2, 1, 5)), crsd_xml
    )
    with pytest.raises(ValueError, match=".*does not use Polynomials.*"):
        skcrsd.compute_dwelltimes_using_poly(
            ch_id, 0, 0, crsd_with_dta.elem.getroottree()
        )

    # with DTA
    skcrsd.compute_dwelltimes_using_dta(
        ch_id, 0, 0, crsd_with_dta.elem.getroottree(), dta
    )
    skcrsd.compute_dwelltimes_using_dta(
        ch_id,
        np.zeros((2, 4, 5)),
        np.zeros((2, 1, 5)),
        crsd_with_dta.elem.getroottree(),
        dta,
    )
    with pytest.raises(ValueError, match=".*does not use a DTA.*"):
        skcrsd.compute_dwelltimes_using_dta(ch_id, 0, 0, crsd_xml, dta)
    with pytest.raises(ValueError, match=".*shape.*does not match.*shape.*"):
        skcrsd.compute_dwelltimes_using_dta(
            ch_id, 0, 0, crsd_with_dta.elem.getroottree(), dta[:-1, :-1]
        )
    with pytest.raises(ValueError, match=".*dtype.*is not compatible with.*dtype.*"):
        skcrsd.compute_dwelltimes_using_dta(
            ch_id, 0, 0, crsd_with_dta.elem.getroottree(), np.zeros(dta.shape, np.uint8)
        )


@pytest.mark.parametrize("use_mask", (True, False))
def test_interpolate_support_array(use_mask):
    dcx_0, dcy_0 = -0.75, -0.8
    num_rows, num_cols = 51, 29

    dcx = np.linspace(dcx_0, -dcx_0, num_rows, endpoint=True)
    dcy = np.linspace(dcy_0, -dcy_0, num_cols, endpoint=True)

    dcx_ss = np.diff(dcx[:2])
    dcy_ss = np.diff(dcy[:2])

    dcxx, dcyy = np.meshgrid(dcx, dcy, indexing="ij")

    sa = dcxx**2 + dcyy**2
    valid = np.sqrt(dcxx**2 + dcyy**2) < 0.9

    rng = np.random.default_rng()
    dcx_i, dcy_i = 2 * rng.random((2, 3, 40, 50)) - 1
    v, dv = skcrsd.interpolate_support_array(
        dcx_i,
        dcy_i,
        dcx_0,
        dcy_0,
        dcx_ss,
        dcy_ss,
        sa,
        dv_sa=valid if use_mask else None,
    )

    sa_masked = sa.copy()
    sa_masked[~valid] = np.nan
    scipy_int = scipy.interpolate.RegularGridInterpolator(
        (dcx, dcy),
        sa_masked if use_mask else sa,
        bounds_error=False,
    )
    v_scipy = scipy_int((dcx_i, dcy_i))
    assert np.allclose(v, v_scipy, equal_nan=True)
    assert np.array_equal(dv, ~np.isnan(v_scipy))


def test_compute_reference_geometry_sar(example_crsdsar):
    with open(example_crsdsar, "rb") as f, skcrsd.Reader(f) as r:
        crsdxml = r.metadata.xmltree
        pvps = r.read_pvps(crsdxml.findtext(".//{*}RefChId"))
        ppps = r.read_ppps(crsdxml.findtext(".//{*}RefTxId"))
    assert crsdxml.find(".//{*}DwellTimes/{*}Array") is None
    crsdxml.find("{*}ReferenceGeometry").clear()
    refgeom = skcrsd.compute_reference_geometry(crsdxml, pvps=pvps, ppps=ppps)
    crsdxml.getroot().replace(crsdxml.find("{*}ReferenceGeometry"), refgeom)
    basis_version = lxml.etree.QName(crsdxml.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=skcrsd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(crsdxml)


def test_compute_reference_geometry_rcv(example_crsdrcv):
    with open(example_crsdrcv, "rb") as f, skcrsd.Reader(f) as r:
        crsdxml = r.metadata.xmltree
        pvps = r.read_pvps(crsdxml.findtext(".//{*}RefChId"))
    crsdxml.find("{*}ReferenceGeometry").clear()
    refgeom = skcrsd.compute_reference_geometry(crsdxml, pvps=pvps)
    crsdxml.getroot().replace(crsdxml.find("{*}ReferenceGeometry"), refgeom)
    basis_version = lxml.etree.QName(crsdxml.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=skcrsd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(crsdxml)


def test_compute_reference_geometry_tx(example_crsdtx):
    with open(example_crsdtx, "rb") as f, skcrsd.Reader(f) as r:
        crsdxml = r.metadata.xmltree
        ppps = r.read_ppps(crsdxml.findtext(".//{*}RefTxId"))
    crsdxml.find("{*}ReferenceGeometry").clear()
    refgeom = skcrsd.compute_reference_geometry(crsdxml, ppps=ppps)
    crsdxml.getroot().replace(crsdxml.find("{*}ReferenceGeometry"), refgeom)
    basis_version = lxml.etree.QName(crsdxml.getroot()).namespace
    schema = lxml.etree.XMLSchema(file=skcrsd.VERSION_INFO[basis_version]["schema"])
    schema.assertValid(crsdxml)
