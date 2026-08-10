import pathlib

import lxml.etree
import numpy as np
import pytest

import sarkit.cphd as skcphd

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def test_compute_dwelltimes_using_poly():
    cphd_xml = lxml.etree.parse(DATAPATH / "example-cphd-1.1.0.xml")
    ref_chid = cphd_xml.findtext(".//{*}RefChId")

    srp_cod, srp_dwell = skcphd.compute_dwelltimes_using_poly(ref_chid, 0, 0, cphd_xml)
    assert srp_cod == pytest.approx(float(cphd_xml.findtext(".//{*}SRPCODTime")))
    assert srp_dwell == pytest.approx(float(cphd_xml.findtext(".//{*}SRPDwellTime")))

    # broadcast smoke test
    skcphd.compute_dwelltimes_using_poly(
        ref_chid, np.zeros((2, 4, 5)), np.zeros((2, 1, 5)), cphd_xml
    )
