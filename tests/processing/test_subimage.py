import pathlib

import numpy as np
import pytest
from lxml import etree

import sarkit._processing as skproc
import sarkit.sicd as sksicd

good_sicd_xml_path = (
    pathlib.Path(__file__).absolute().parent / "../../data" / "example-sicd-1.2.1.xml"
)


@pytest.fixture(scope="module")
def good_xml():
    with open(good_sicd_xml_path, "rb") as infile:
        root = etree.parse(infile)
    return root


@pytest.fixture(scope="module")
def complex_array(good_xml):
    xml_helper = sksicd.XmlHelper(good_xml)
    num_rows = xml_helper.load("./{*}ImageData/{*}NumRows")
    num_cols = xml_helper.load("./{*}ImageData/{*}NumCols")
    shape = (num_rows, num_cols)

    assert xml_helper.load("./{*}ImageData/{*}PixelType") == "RE32F_IM32F"

    rng = np.random.default_rng()
    return rng.random(shape, dtype=np.float32) + 1j * rng.random(
        shape, dtype=np.float32
    )


def test_subimage_smoke(complex_array, good_xml):
    xmlhelp_in = sksicd.XmlHelper(good_xml)
    first_row = 10
    first_col = 13
    num_rows = 11
    num_cols = 19
    out_arr, out_xml = skproc.sicd_subimage(
        complex_array, good_xml, first_row, first_col, num_rows, num_cols
    )
    xmlhelp_out = sksicd.XmlHelper(out_xml)

    assert np.array_equal(
        out_arr,
        complex_array[
            first_row : first_row + num_rows, first_col : first_col + num_cols
        ],
    )

    assert (
        xmlhelp_out.load("./{*}ImageData/{*}FirstRow")
        == xmlhelp_in.load("./{*}ImageData/{*}FirstRow") + first_row
    )
    assert (
        xmlhelp_out.load("./{*}ImageData/{*}FirstCol")
        == xmlhelp_in.load("./{*}ImageData/{*}FirstCol") + first_col
    )
    assert xmlhelp_out.load("./{*}ImageData/{*}NumRows") == num_rows
    assert xmlhelp_out.load("./{*}ImageData/{*}NumCols") == num_cols
    assert not np.array_equal(
        xmlhelp_out.load("./{*}GeoData/{*}ImageCorners"),
        xmlhelp_in.load("./{*}GeoData/{*}ImageCorners"),
    )
