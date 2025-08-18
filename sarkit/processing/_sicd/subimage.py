"""
Extract a subimage (chip) from a SICD
"""

import lxml.etree
import numpy.typing as npt

import sarkit.sicd._io


def sicd_subimage(
    array: npt.NDArray,
    sicd_xmltree: lxml.etree.ElementTree,
    first_row: int,
    first_col: int,
    num_rows: int,
    num_cols: int,
) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
    """Extract a subimage

    Updates the ImageData fields as expected and the GeoData/ImageCorners
    using a straight-line projection approximation to a plane.

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree
    first_row : int
        first row to extract, relative to ImageData/FirstRow
    first_col : int
        first column to extract, relative to ImageData/FirstCol
    num_rows : int
        number of rows to extract
    num_cols : int
        number of columns to extract

    Returns
    -------
    array_out : ndarray
        2D array of extracted complex pixels
    sicd_xmltree_out : lxml.etree.ElementTree
        Updated SICD XML ElementTree
    """

    assert first_row >= 0
    assert first_col >= 0
    end_row = first_row + num_rows
    end_col = first_col + num_cols
    assert end_row <= array.shape[0]
    assert end_col <= array.shape[1]
    array_out = array[first_row:end_row, first_col:end_col].copy()

    sicd_xmltree_out = sarkit.sicd._io._update_sicd_subimage_xml(
        sicd_xmltree, first_row, first_col, num_rows, num_cols
    )
    return array_out, sicd_xmltree_out
