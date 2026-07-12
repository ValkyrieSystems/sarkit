import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.sicd as sksicd


def _scppixel(xmltree: lxml.etree.ElementTree) -> npt.NDArray:
    ew = sksicd.ElementWrapper(xmltree.getroot())
    return np.asarray(ew["ImageData"]["SCPPixel"])


def _ss(xmltree: lxml.etree.ElementTree) -> npt.NDArray:
    ew = sksicd.ElementWrapper(xmltree.getroot())
    return np.asarray([ew["Grid"]["Row"]["SS"], ew["Grid"]["Col"]["SS"]])


def rowcol_to_irowicol(
    xmltree: lxml.etree.ElementTree, rowcol: npt.ArrayLike
) -> npt.NDArray:
    """Global Row, Column Indices (row, col) to SCP Pixel-Centered Image Indices (irow, icol)

    Parameters
    ----------
    xmltree : lxml.etree.ElementTree
        SICD metadata
    rowcol : (..., 2) array_like
        N-D array of indices with (row, col) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of indices with (irow, icol) in the last dimension

    Notes
    -----
    row, col are defined in SICD as only having integer values.  This function allows them to have continuous values.

    """
    irowicol = rowcol - _scppixel(xmltree)
    return irowicol


def irowicol_to_xrowycol(
    xmltree: lxml.etree.ElementTree, irowicol: npt.ArrayLike
) -> npt.NDArray:
    """SCP Pixel-Centered Image Indices (irow, icol) to SCP Centered Image Coordinates (xrow, ycol)

    Parameters
    ----------
    xmltree : lxml.etree.ElementTree
        SICD metadata
    irowicol : (..., 2) array_like
        N-D array of indices with (irow, icol) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of indices with (xrow, ycol) in the last dimension

    """
    xrowycol = _ss(xmltree) * irowicol
    return xrowycol


def rowcol_to_xrowycol(
    xmltree: lxml.etree.ElementTree, rowcol: npt.ArrayLike
) -> npt.NDArray:
    """Global Row, Column Indices (row, col) to SCP Centered Image Coordinates (xrow, ycol)

    Parameters
    ----------
    xmltree : lxml.etree.ElementTree
        SICD metadata
    rowcol : (..., 2) array_like
        N-D array of indices with (row, col) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of indices with (xrow, ycol) in the last dimension

    Notes
    -----
    row, col are defined in SICD as only having integer values.  This function allows them to have continuous values.

    """
    irowicol = rowcol_to_irowicol(xmltree, rowcol)
    return irowicol_to_xrowycol(xmltree, irowicol)


def irowicol_to_rowcol(
    xmltree: lxml.etree.ElementTree, irowicol: npt.ArrayLike
) -> npt.NDArray:
    """SCP Pixel-Centered Image Indices (irow, icol) to Global Row, Column Indices (row, col)

    Parameters
    ----------
    xmltree : lxml.etree.ElementTree
        SICD metadata
    irowicol : (..., 2) array_like
        N-D array of indices with (irow, icol) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of indices with (row, col) in the last dimension

    Notes
    -----
    row, col are defined in SICD as only having integer values.  This function allows them to have continuous values.

    """
    rowcol = np.asarray(irowicol) + _scppixel(xmltree)
    return rowcol


def xrowycol_to_irowicol(
    xmltree: lxml.etree.ElementTree, xrowycol: npt.ArrayLike
) -> npt.NDArray:
    """SCP Centered Image Coordinates (xrow, ycol) to SCP Pixel-Centered Image Indices (irow, icol)

    Parameters
    ----------
    xmltree : lxml.etree.ElementTree
        SICD metadata
    xrowycol : (..., 2) array_like
        N-D array of indices with (xrow, ycol) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of indices with (irow, icol) in the last dimension

    """
    irowicol = xrowycol / _ss(xmltree)
    return irowicol


def xrowycol_to_rowcol(
    xmltree: lxml.etree.ElementTree, xrowycol: npt.ArrayLike
) -> npt.NDArray:
    """SCP Centered Image Coordinates (xrow, ycol) to Global Row, Column Indices (row, col)

    Parameters
    ----------
    xmltree : lxml.etree.ElementTree
        SICD metadata
    xrowycol : (..., 2) array_like
        N-D array of indices with (xrow, yol) in the last dimension

    Returns
    -------
    (..., 2) ndarray
        N-D array of indices with (row, col) in the last dimension

    Notes
    -----
    row, col are defined in SICD as only having integer values.  This function allows them to have continuous values.

    """
    irowicol = xrowycol_to_irowicol(xmltree, xrowycol)
    return irowicol_to_rowcol(xmltree, irowicol)
