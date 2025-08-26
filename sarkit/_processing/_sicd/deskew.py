"""
Deskew (apply phase polynomial) to SICDs.
"""

import copy

import lxml.etree
import numba
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt

import sarkit.sicd as sksicd


@numba.njit(parallel=True)
def _apply_phase_poly(array, phase_poly, row_0, row_ss, col_0, col_ss):
    """numba parallelized phase poly application"""
    out = np.empty_like(array)
    for rowidx in numba.prange(out.shape[0]):
        row_val = row_0 + rowidx * row_ss
        col_poly = phase_poly[-1, :]
        for ndx in range(phase_poly.shape[0] - 1, 0, -1):
            col_poly = col_poly * row_val + phase_poly[ndx - 1, :]
        for colidx in range(out.shape[1]):
            col_val = col_0 + colidx * col_ss
            phase_val = col_poly[-1]
            for ndx in range(col_poly.shape[0] - 1, 0, -1):
                phase_val = phase_val * col_val + col_poly[ndx - 1]

            out[rowidx, colidx] = array[rowidx, colidx] * np.exp(
                1j * 2 * np.pi * phase_val
            )

    return out


def _update_grid_metadata(phase_poly, xml_helper):
    """Update the metadata following a deskew operation"""
    for dim in ["Row", "Col"]:
        axis_index = {"Row": 0, "Col": 1}[dim]
        delta_k_coa_poly = _get_delta_kcoa_poly(xml_helper, dim)
        phase_poly_der = npp.polyder(-phase_poly, axis=axis_index) * xml_helper.load(
            f"./{{*}}Grid/{{*}}{dim}/{{*}}Sgn"
        )

        max_dims = np.amax([delta_k_coa_poly.shape, phase_poly_der.shape], axis=0)
        pad = max_dims - phase_poly_der.shape
        phase_poly_der = np.pad(
            phase_poly_der, ((0, pad[0]), (0, pad[1])), mode="constant"
        )
        pad = max_dims - delta_k_coa_poly.shape
        delta_k_coa_poly = np.pad(
            delta_k_coa_poly, ((0, pad[0]), (0, pad[1])), mode="constant"
        )

        updated_poly = delta_k_coa_poly + phase_poly_der
        dkcoa_elem = xml_helper.element_tree.find(
            f"./{{*}}Grid/{{*}}{dim}/{{*}}DeltaKCOAPoly"
        )
        if dkcoa_elem is None:
            dk2_elem = xml_helper.element_tree.find(
                f"./{{*}}Grid/{{*}}{dim}/{{*}}DeltaK2"
            )
            dkcoa_elem = lxml.etree.Element(dk2_elem.tag[:-1] + "COAPoly")
            dk2_elem.addnext(dkcoa_elem)
        xml_helper.set_elem(dkcoa_elem, updated_poly)


def _get_delta_kcoa_poly(xml_helper, axis):
    assert axis in {"Row", "Col"}
    delta_k_coa_poly = xml_helper.load(f"./{{*}}Grid/{{*}}{axis}/{{*}}DeltaKCOAPoly")
    if delta_k_coa_poly is None or not np.count_nonzero(delta_k_coa_poly):
        return np.array([[0.0]])
    return delta_k_coa_poly


def sicd_get_deskew_phase_poly(
    sicd_xmltree: lxml.etree.ElementTree, axis: str
) -> npt.NDArray:
    """Return phase polynomial for deskew

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree
    axis : {'Row', 'Col'}
        Which axis to deskew

    Returns
    -------
    phase_poly : ndarray
        Array of phase polynomial coefficients
    """
    xml_helper = sksicd.XmlHelper(sicd_xmltree)
    axis_index = {"Row": 0, "Col": 1}[axis]
    delta_k_coa_poly = _get_delta_kcoa_poly(xml_helper, axis)
    sign = xml_helper.load(f"./{{*}}Grid/{{*}}{axis}/{{*}}Sgn")
    return npp.polyint(delta_k_coa_poly, axis=axis_index) * sign


def sicd_apply_phase_poly(
    array: npt.NDArray, phase_poly: npt.NDArray, sicd_xmltree: lxml.etree.ElementTree
) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
    """Metadata aware phase poly application

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    phase_poly : ndarray
        Array of phase polynomial coefficients
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree

    Returns
    -------
    array_out : ndarray
        2D array of adjusted complex pixels
    sicd_xmltree_out : lxml.etree.ElementTree
        Updated SICD XML ElementTree
    """
    sicd_xmltree_out = copy.deepcopy(sicd_xmltree)
    xml_helper = sksicd.XmlHelper(sicd_xmltree_out)
    row_ss = xml_helper.load("./{*}Grid/{*}Row/{*}SS")
    row_0 = (
        xml_helper.load("./{*}ImageData/{*}FirstRow")
        - xml_helper.load("./{*}ImageData/{*}SCPPixel/{*}Row")
    ) * row_ss
    col_ss = xml_helper.load("./{*}Grid/{*}Col/{*}SS")
    col_0 = (
        xml_helper.load("./{*}ImageData/{*}FirstCol")
        - xml_helper.load("./{*}ImageData/{*}SCPPixel/{*}Col")
    ) * col_ss

    array_out = _apply_phase_poly(array, phase_poly, row_0, row_ss, col_0, col_ss)
    _update_grid_metadata(phase_poly, xml_helper)

    return array_out, sicd_xmltree_out


def sicd_deskew(
    array: npt.NDArray, sicd_xmltree: lxml.etree.ElementTree, axis: str
) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
    """Deskew complex data array

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree
    axis : {'Row', 'Col'}
        Which axis to deskew

    Returns
    -------
    array_deskew : ndarray
        2D array of deskewed complex pixels
    sicd_xmltree_deskew : lxml.etree.ElementTree
        Updated SICD XML ElementTree
    """
    phase_poly = sicd_get_deskew_phase_poly(sicd_xmltree, axis)
    array_deskew, sicd_xmltree_deskew = sicd_apply_phase_poly(
        array, phase_poly, sicd_xmltree
    )

    return array_deskew, sicd_xmltree_deskew
