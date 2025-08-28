import pathlib

import lxml.etree
import numpy as np
import pytest

import sarkit._processing as skproc
import sarkit.sicd as sksicd

sicd_xml_path = (
    pathlib.Path(__file__).absolute().parents[2] / "data/example-sicd-1.3.0.xml"
)


def _fake_pixels(sicd_etree):
    xml_helper = sksicd.XmlHelper(sicd_etree)
    num_rows = xml_helper.load("./{*}ImageData/{*}NumRows")
    num_cols = xml_helper.load("./{*}ImageData/{*}NumCols")
    shape = (num_rows, num_cols)

    assert xml_helper.load("./{*}ImageData/{*}PixelType") == "RE32F_IM32F"

    return np.random.default_rng().random(
        shape, dtype=np.float32
    ) + 1j * np.random.default_rng().random(shape, dtype=np.float32)


@pytest.mark.parametrize("dim", ("Row", "Col"))
def test_deskew_smoke(dim):
    xml_in = lxml.etree.parse(sicd_xml_path)
    for el in xml_in.findall(".//{*}DeltaKCOAPoly"):
        el.getparent().remove(el)
    pixels_in = _fake_pixels(xml_in)

    # no-ops when DeltaKCOAPoly is not present
    pixels_out, xml_out = skproc.sicd_deskew(pixels_in, xml_in, dim)
    xmlhelp_out = sksicd.XmlHelper(xml_out)
    assert np.allclose(pixels_in, pixels_out)
    assert not np.count_nonzero(
        xmlhelp_out.load(f"./{{*}}Grid/{{*}}{dim}/{{*}}DeltaKCOAPoly")
    )

    # no-ops when DeltaKCOAPoly is 0
    pixels_out2, xml_out2 = skproc.sicd_deskew(pixels_out, xml_out, dim)
    xmlhelp_out2 = sksicd.XmlHelper(xml_out2)
    assert np.allclose(pixels_out, pixels_out2)
    assert not np.count_nonzero(
        xmlhelp_out2.load(f"./{{*}}Grid/{{*}}{dim}/{{*}}DeltaKCOAPoly")
    )


def test_deskew_nonzero_poly():
    xml_in = lxml.etree.parse(sicd_xml_path)
    pixels_in = _fake_pixels(xml_in)
    pixels_out, xml_out = skproc.sicd_deskew(pixels_in, xml_in, "Row")
    xml_helper = sksicd.XmlHelper(xml_out)
    output_row_deltakcoapoly = xml_helper.load("./{*}Grid/{*}Row/{*}DeltaKCOAPoly")
    assert not np.count_nonzero(output_row_deltakcoapoly)
    assert not np.allclose(pixels_in, pixels_out)
