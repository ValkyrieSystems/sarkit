import copy
import pathlib

import numpy as np
import pytest
from lxml import etree

import sarkit._processing as skproc
import sarkit.sicd as sksicd

good_sicd_xml_path = (
    pathlib.Path(__file__).absolute().parent / "../../data" / "example-sicd-1.2.1.xml"
)


@pytest.fixture
def good_xml():
    with open(good_sicd_xml_path, "rb") as infile:
        root = etree.parse(infile)
    return root


@pytest.fixture
def amp_phs(good_xml):
    xml_helper = sksicd.XmlHelper(good_xml)
    num_rows = xml_helper.load("./{*}ImageData/{*}NumRows")
    num_cols = xml_helper.load("./{*}ImageData/{*}NumCols")
    shape = (num_rows, num_cols)

    xml_helper.set("./{*}ImageData/{*}PixelType", "AMP8I_PHS8I")
    pixel_type_elem = xml_helper.element_tree.find("./{*}ImageData/{*}PixelType")
    elem_ns = etree.QName(pixel_type_elem).namespace
    ns = f"{{{elem_ns}}}" if elem_ns else ""

    xml_helper.element_tree.find("./{*}ImageData/{*}PixelType").addnext(
        etree.Element(ns + "AmpTable")
    )
    xml_helper.set("./{*}ImageData/{*}AmpTable", np.linspace(0, 1e3, 256))

    arr = np.random.default_rng().integers(256, size=shape + (2,), dtype=np.uint8)
    arr = arr.view(dtype=sksicd.PIXEL_TYPES["AMP8I_PHS8I"]["dtype"]).reshape(shape)
    return arr, good_xml


def power(a):
    try:
        return a.real**2 + a.imag**2
    except Exception:
        return a["real"].astype(np.float32) ** 2 + a["imag"].astype(np.float32) ** 2


def _check_pixel_scaling(in1, xmlhelp1, in2, xmlhelp2):
    "Check that the pixels refer to the same RCS, tol is sqrt(2) in units of in2"

    rcs_poly1 = xmlhelp1.load("./{*}Radiometric/{*}RCSSFPoly")
    rcs_poly2 = xmlhelp2.load("./{*}Radiometric/{*}RCSSFPoly")

    scale_factor = rcs_poly1[0, 0] / rcs_poly2[0, 0]
    # The polys should be scaled copies of each other
    np.testing.assert_allclose(rcs_poly1, rcs_poly2 * scale_factor)

    # converting to RCS-ish and scaling to the abs of in2
    # because the tolerance is easier to quantify
    np.testing.assert_allclose(
        (power(in1) * scale_factor) ** 0.5, power(in2) ** 0.5, atol=np.sqrt(2)
    )


def test_bad_pixel_type(amp_phs):
    complex_array, good_xml = amp_phs
    with pytest.raises(TypeError):
        skproc.sicd_as_re32f_im32f(
            np.zeros(complex_array.shape, dtype=np.complex64), good_xml
        )
    with pytest.raises(TypeError):
        skproc.sicd_as_re16i_im16i(
            np.zeros(complex_array.shape, dtype=np.complex64), good_xml
        )
    with pytest.raises(TypeError):
        skproc.sicd_as_amp8i_phs8i(
            np.zeros(complex_array.shape, dtype=np.complex64), good_xml, np.arange(256)
        )


def test_pixel_type(amp_phs):
    complex_array, good_xml = amp_phs
    xmlhelp_in = sksicd.XmlHelper(good_xml)
    amp_array_in = xmlhelp_in.load("./{*}ImageData/{*}AmpTable")[complex_array["amp"]]

    # AMP8I_PHS8I -> RE32F_IM32F
    f32_arr, f32_xml = skproc.sicd_as_re32f_im32f(complex_array, good_xml)
    xmlhelp_f32 = sksicd.XmlHelper(f32_xml)
    assert xmlhelp_f32.load("./{*}ImageData/{*}AmpTable") is None
    _check_pixel_scaling(f32_arr, xmlhelp_f32, amp_array_in, xmlhelp_in)
    assert xmlhelp_f32.load("./{*}ImageData/{*}PixelType") == "RE32F_IM32F"

    # AMP8I_PHS8I -> RE32F_IM32F NO LUT
    no_lut_xml = copy.deepcopy(good_xml)
    lut_node = no_lut_xml.find("./{*}ImageData/{*}AmpTable")
    lut_node.getparent().remove(lut_node)
    f32_arr_0, f32_xml_0 = skproc.sicd_as_re32f_im32f(complex_array, no_lut_xml)
    xmlhelp_f32_0 = sksicd.XmlHelper(f32_xml_0)
    assert xmlhelp_f32_0.load("./{*}ImageData/{*}AmpTable") is None
    _check_pixel_scaling(
        f32_arr_0, xmlhelp_f32_0, complex_array["amp"].astype(np.float32), xmlhelp_in
    )
    assert xmlhelp_f32_0.load("./{*}ImageData/{*}PixelType") == "RE32F_IM32F"

    # AMP8I_PHS8I -> RE16I_IM16I
    i16_arr, i16_xml = skproc.sicd_as_re16i_im16i(complex_array, good_xml)
    xmlhelp_i16 = sksicd.XmlHelper(i16_xml)
    assert xmlhelp_i16.load("./{*}ImageData/{*}AmpTable") is None
    _check_pixel_scaling(f32_arr, xmlhelp_f32, i16_arr, xmlhelp_i16)
    assert xmlhelp_i16.load("./{*}ImageData/{*}PixelType") == "RE16I_IM16I"

    # RE32F_IM32F -> RE16I_IM16I
    i16_arr_2, i16_xml_2 = skproc.sicd_as_re16i_im16i(f32_arr, f32_xml)
    xmlhelp_i16_2 = sksicd.XmlHelper(i16_xml_2)
    assert xmlhelp_i16_2.load("./{*}ImageData/{*}PixelType") == "RE16I_IM16I"

    _check_pixel_scaling(f32_arr, xmlhelp_f32, i16_arr_2, xmlhelp_i16_2)

    # RE32F_IM32F -> AMP8I_PHS8I
    i8_arr, i8_xml = skproc.sicd_as_amp8i_phs8i(
        f32_arr, f32_xml, xmlhelp_in.load("./{*}ImageData/{*}AmpTable")
    )
    xmlhelp_i8 = sksicd.XmlHelper(i8_xml)
    assert xmlhelp_i8.load("./{*}ImageData/{*}PixelType") == "AMP8I_PHS8I"

    amp_array = xmlhelp_i8.load("./{*}ImageData/{*}AmpTable")[i8_arr["amp"]]
    _check_pixel_scaling(f32_arr, xmlhelp_f32, amp_array, xmlhelp_i8)

    # RE16I_IM16I -> RE32F_IM32F
    f32_arr2, f32_xml2 = skproc.sicd_as_re32f_im32f(i16_arr, i16_xml)
    xmlhelp_f32_2 = sksicd.XmlHelper(f32_xml2)
    assert xmlhelp_f32_2.load("./{*}ImageData/{*}PixelType") == "RE32F_IM32F"

    _check_pixel_scaling(f32_arr, xmlhelp_f32, f32_arr2, xmlhelp_f32_2)

    # RE16I_IM16I -> AMP8I_PHS8I
    amp_table = np.arange(256) / 255 * (2**15 - 1)
    i8_arr_2, i8_xml_2 = skproc.sicd_as_amp8i_phs8i(i16_arr, i16_xml, amp_table)
    xmlhelp_i8_2 = sksicd.XmlHelper(i8_xml_2)
    assert xmlhelp_i8_2.load("./{*}ImageData/{*}PixelType") == "AMP8I_PHS8I"

    amp_array_2 = xmlhelp_i8_2.load("./{*}ImageData/{*}AmpTable")[i8_arr_2["amp"]]
    _check_pixel_scaling(i16_arr, xmlhelp_i16, amp_array_2, xmlhelp_i8_2)
