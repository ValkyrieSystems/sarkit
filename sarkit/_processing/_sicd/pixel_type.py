"""
Change the pixel type of SICDs.
"""

import copy

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.sicd as sksicd


def _max_abs(array):
    return np.abs(array).max()


def _amp8i_phs8i_to_re32f_im32f(array, xml_helper):
    amp_table = xml_helper.load("./{*}ImageData/{*}AmpTable")
    if amp_table is None:
        amp_table = np.arange(256, dtype=np.float64)
    out_array = (
        amp_table[array["amp"]]
        * np.exp(np.complex64(1j * np.pi * 2) * array["phase"] / 256)
    ).astype(np.complex64)

    amp_table_elem = xml_helper.element_tree.find("{*}ImageData/{*}AmpTable")
    if amp_table_elem is not None:
        amp_table_elem.getparent().remove(amp_table_elem)
    xml_helper.set("./{*}ImageData/{*}PixelType", "RE32F_IM32F")
    return out_array


def _re16i_im16i_to_re32f_im32f(array, xml_helper):
    out_array = (array["real"] + np.complex64(1j) * array["imag"]).astype(np.complex64)
    xml_helper.set("./{*}ImageData/{*}PixelType", "RE32F_IM32F")
    return out_array


def sicd_as_re32f_im32f(
    array: npt.NDArray, sicd_xmltree: lxml.etree.ElementTree
) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
    """Converts a SICD to RE32F_IM32F pixel type.

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree

    Returns
    -------
    array_out : ndarray
        2D array of complex pixels. If input matches the requested pixel type,
        the input is returned instead of a copy.
    sicd_xmltree_out : lxml.etree.ElementTree
        Updated SICD XML ElementTree. If input matches the requested pixel type,
        the input is returned instead of a copy.
    """
    input_type = sicd_xmltree.findtext("./{*}ImageData/{*}PixelType")

    if sksicd.PIXEL_TYPES[input_type]["dtype"] != array.dtype.newbyteorder("="):
        raise TypeError(
            f"{array.dtype=} does not match ImageData/PixelType={input_type}"
        )

    if input_type == "RE32F_IM32F":
        return array, sicd_xmltree

    sicd_xmltree_out = copy.deepcopy(sicd_xmltree)
    xml_helper = sksicd.XmlHelper(sicd_xmltree_out)
    if sicd_xmltree.findtext("./{*}ImageData/{*}PixelType") == "RE16I_IM16I":
        out_array = _re16i_im16i_to_re32f_im32f(array, xml_helper)
    elif sicd_xmltree.findtext("./{*}ImageData/{*}PixelType") == "AMP8I_PHS8I":
        out_array = _amp8i_phs8i_to_re32f_im32f(array, xml_helper)
    return out_array, sicd_xmltree_out


def sicd_as_re16i_im16i(
    array: npt.NDArray, sicd_xmltree: lxml.etree.ElementTree
) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
    """Converts a SICD to RE16I_IM16I pixel type.

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree

    Returns
    -------
    array_out : ndarray
        2D array of complex pixels. If input matches the requested pixel type,
        the input is returned instead of a copy.
    sicd_xmltree_out : lxml.etree.ElementTree
        Updated SICD XML ElementTree. If input matches the requested pixel type,
        the input is returned instead of a copy.
    """
    input_type = sicd_xmltree.findtext("./{*}ImageData/{*}PixelType")

    if sksicd.PIXEL_TYPES[input_type]["dtype"] != array.dtype.newbyteorder("="):
        raise TypeError(
            f"{array.dtype=} does not match ImageData/PixelType={input_type}"
        )

    if input_type == "RE16I_IM16I":
        return array, sicd_xmltree

    sicd_xmltree_out = copy.deepcopy(sicd_xmltree)
    xml_helper = sksicd.XmlHelper(sicd_xmltree_out)
    if xml_helper.load("./{*}ImageData/{*}PixelType") == "AMP8I_PHS8I":
        array = _amp8i_phs8i_to_re32f_im32f(array, xml_helper)
    array_f32 = array.reshape(array.shape + (1,)).view(array.real.dtype)
    mabs = _max_abs(array_f32)
    scale = (2**15 - 1) / mabs
    scale_sq = scale * scale
    out_array = (
        np.round(array_f32 * scale)
        .astype(np.int16)
        .view(sksicd.PIXEL_TYPES["RE16I_IM16I"]["dtype"])
        .reshape(array.shape)
    )
    for sf in ["RCS", "SigmaZero", "BetaZero", "GammaZero"]:
        name = f"./{{*}}Radiometric/{{*}}{sf}SFPoly"
        poly = xml_helper.load(name)
        if poly is not None:
            xml_helper.set(name, poly / scale_sq)
    if (
        xml_helper.load("./{*}Radiometric/{*}NoiseLevel/{*}NoiseLevelType")
        == "ABSOLUTE"
    ):
        noise_poly = xml_helper.load("./{*}Radiometric/{*}NoiseLevel/{*}NoisePoly")
        noise_poly[0, 0] += 20 * np.log10(scale)
        xml_helper.set("./{*}Radiometric/{*}NoiseLevel/{*}NoisePoly", noise_poly)
    xml_helper.set("./{*}ImageData/{*}PixelType", "RE16I_IM16I")
    return out_array, sicd_xmltree_out


def sicd_as_amp8i_phs8i(
    array: npt.NDArray, sicd_xmltree: lxml.etree.ElementTree, lut: npt.NDArray
) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
    """Converts a SICD to AMP8I_PHS8I pixel type.

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree
    lut : ndarray
        Amplitude lookup table

    Returns
    -------
    array_out : ndarray
        2D array of complex pixels. If input matches the requested pixel type,
        the input is returned instead of a copy.
    sicd_xmltree_out : lxml.etree.ElementTree
        Updated SICD XML ElementTree. If input matches the requested pixel type,
        the input is returned instead of a copy.
    """
    input_type = sicd_xmltree.findtext("./{*}ImageData/{*}PixelType")

    if sksicd.PIXEL_TYPES[input_type]["dtype"] != array.dtype.newbyteorder("="):
        raise TypeError(
            f"{array.dtype=} does not match ImageData/PixelType={input_type}"
        )

    if lut.size != 256:
        raise ValueError("lut must be size 256")
    xml_helper_in = sksicd.XmlHelper(sicd_xmltree)
    if input_type == "AMP8I_PHS8I" and np.array_equal(
        lut, xml_helper_in.load("./{*}ImageData/{*}AmpTable")
    ):
        return array, sicd_xmltree
    array, sicd_xmltree_out = sicd_as_re32f_im32f(array, sicd_xmltree)
    xml_helper = sksicd.XmlHelper(sicd_xmltree_out)
    out_array = np.empty(array.shape, sksicd.PIXEL_TYPES["AMP8I_PHS8I"]["dtype"])
    lut2 = ((lut[1:] + lut[:-1]) / 2) ** 2
    out_array["amp"] = np.digitize(array.real**2 + array.imag**2, lut2)
    out_array["phase"] = np.round(np.angle(array) / (2 * np.pi) * 256) % 256

    pixel_type_elem = xml_helper.element_tree.find("./{*}ImageData/{*}PixelType")
    elem_ns = lxml.etree.QName(pixel_type_elem).namespace
    ns = f"{{{elem_ns}}}" if elem_ns else ""
    pixel_type_elem.addnext(lxml.etree.Element(ns + "AmpTable"))
    xml_helper.set("./{*}ImageData/{*}AmpTable", lut)
    xml_helper.set("./{*}ImageData/{*}PixelType", "AMP8I_PHS8I")
    return out_array, sicd_xmltree_out
