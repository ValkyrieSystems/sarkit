"""
Functions for interacting with SIDD XML
"""

import numbers
from collections.abc import Sequence
from typing import Any

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit._xmlhelp as skxml
import sarkit._xmlhelp2 as skxml2

from . import _xmlhelp2

NSMAP = {
    "sicommon": "urn:SICommon:1.0",
}


# The following transcoders happen to share common implementation across several standards
@skxml.inheritdocstring
class BoolType(skxml.BoolType):
    pass


@skxml.inheritdocstring
class DblType(skxml.DblType):
    pass


@skxml.inheritdocstring
class EnuType(skxml.EnuType):
    pass


@skxml.inheritdocstring
class IntType(skxml.IntType):
    pass


@skxml.inheritdocstring
class TxtType(skxml.TxtType):
    pass


@skxml.inheritdocstring
class XdtType(skxml.XdtType):
    pass


class XyzType(skxml.XyzType):
    """Transcoder for XML parameter types containing scalar X, Y, and Z components.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class AngleMagnitudeType(skxml.ArrayType):
    """Transcoder for double-precision floating point angle magnitude XML parameter type.

    Children are in the SICommon namespace.
    """

    def __init__(self) -> None:
        super().__init__(
            subelements={c: skxml.DblType() for c in ("Angle", "Magnitude")},
            child_ns=NSMAP["sicommon"],
        )


class LatLonType(skxml.LatLonType):
    """Transcoder for XML parameter types containing scalar Lat and Lon components.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


@skxml.inheritdocstring
class ParameterType(skxml.ParameterType):
    pass


class PolyCoef1dType(skxml.PolyType):
    """Transcoder for one-dimensional polynomial (PolyCoef1D) XML parameter types.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class PolyCoef2dType(skxml.Poly2dType):
    """Transcoder for two-dimensional polynomial (PolyCoef2D) XML parameter types.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class RowColIntType(skxml.RowColType):
    """Transcoder for XML parameter types containing scalar, integer Row and Col components (RC_INT).

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class XyzPolyType(skxml.XyzPolyType):
    """Transcoder for XYZ_POLY XML parameter types containing triplets of 1D polynomials.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class FilterCoefficientType(skxml.Type):
    """
    Transcoder for FilterCoefficients.
    Attributes may either be (row, col) or (phasing, point)

    Parameters
    ----------
    attrib_type : str
        Attribute names, either "rowcol" or "phasingpoint"
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, attrib_type: str, child_ns: str = "") -> None:
        if attrib_type == "rowcol":
            self.size_x_name = "numRows"
            self.size_y_name = "numCols"
            self.coef_x_name = "row"
            self.coef_y_name = "col"
        elif attrib_type == "phasingpoint":
            self.size_x_name = "numPhasings"
            self.size_y_name = "numPoints"
            self.coef_x_name = "phasing"
            self.coef_y_name = "point"
        else:
            raise ValueError(f"Unknown attrib_type of {attrib_type}")
        self.child_ns = child_ns

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns an array of filter coefficients encoded in ``elem``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to parse

        Returns
        -------
        coefs : ndarray
            2-dimensional array of coefficients ordered so that the coefficient of x=m and y=n is contained in ``val[m, n]``

        """
        shape = (int(elem.get(self.size_x_name)), int(elem.get(self.size_y_name)))
        coefs = np.zeros(shape, np.float64)
        coef_by_indices = {
            (int(coef.get(self.coef_x_name)), int(coef.get(self.coef_y_name))): float(
                coef.text
            )
            for coef in elem
        }
        for indices, coef in coef_by_indices.items():
            coefs[*indices] = coef
        return coefs

    def set_elem(self, elem: lxml.etree.Element, val: npt.ArrayLike) -> None:
        """Set ``elem`` node using the filter coefficients from ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : array_like
            2-dimensional array of coefficients ordered so that the coefficient of x=m and y=n is contained in ``val[m, n]``

        """
        coefs = np.asarray(val)
        if coefs.ndim != 2:
            raise ValueError("Filter coefficient array must be 2-dimensional")
        elem[:] = []
        elem_ns = self.child_ns if self.child_ns else lxml.etree.QName(elem).namespace
        ns = f"{{{elem_ns}}}" if elem_ns else ""
        elem.set(self.size_x_name, str(coefs.shape[0]))
        elem.set(self.size_y_name, str(coefs.shape[1]))
        for coord, coef in np.ndenumerate(coefs):
            attribs = {
                self.coef_x_name: str(coord[0]),
                self.coef_y_name: str(coord[1]),
            }
            lxml.etree.SubElement(elem, ns + "Coef", attrib=attribs).text = str(coef)


class IntListType(skxml.Type):
    """
    Transcoder for ints in a list XML parameter types.

    """

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns space-separated ints as ndarray of ints"""
        val = "" if elem.text is None else elem.text
        return np.array([int(tok) for tok in val.split(" ")], dtype=int)

    def set_elem(
        self, elem: lxml.etree.Element, val: Sequence[numbers.Integral]
    ) -> None:
        """Sets ``elem`` node using the list of integers in ``val``."""
        elem.text = " ".join([str(entry) for entry in val])


class ImageCornersType(skxml.ListType):
    """
    Transcoder for GeoData/ImageCorners XML parameter types.

    Lat/Lon children are in SICommon namespace.
    """

    def __init__(self) -> None:
        super().__init__("ICP", LatLonType())

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns the array of ImageCorners encoded in ``elem``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to parse

        Returns
        -------
        coefs : (4, 2) ndarray
            Array of [latitude (deg), longitude (deg)] image corners.

        """
        return np.asarray(
            [
                self.sub_type.parse_elem(x)
                for x in sorted(elem, key=lambda x: x.get("index"))
            ]
        )

    def set_elem(
        self, elem: lxml.etree.Element, val: Sequence[Sequence[float]]
    ) -> None:
        """Set the ICP children of ``elem`` using the ordered vertices from ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : (4, 2) array_like
            Array of [latitude (deg), longitude (deg)] image corners.

        """
        elem[:] = []
        labels = ("1:FRFC", "2:FRLC", "3:LRLC", "4:LRFC")
        icp_ns = lxml.etree.QName(elem).namespace
        icp_ns = f"{{{icp_ns}}}" if icp_ns else ""
        for label, coord in zip(labels, val):
            icp = lxml.etree.SubElement(
                elem, icp_ns + self.sub_tag, attrib={"index": label}
            )
            self.sub_type.set_elem(icp, coord)


class RangeAzimuthType(skxml.ArrayType):
    """
    Transcoder for double-precision floating point range and azimuth XML parameter types.

    Children are in the SICommon namespace.

    """

    def __init__(self) -> None:
        super().__init__(
            subelements={c: skxml.DblType() for c in ("Range", "Azimuth")},
            child_ns=NSMAP["sicommon"],
        )


class RowColDblType(skxml.ArrayType):
    """
    Transcoder for double-precision floating point row and column XML parameter types.

    Children are in the SICommon namespace.

    """

    def __init__(self) -> None:
        super().__init__(
            subelements={c: skxml.DblType() for c in ("Row", "Col")},
            child_ns=NSMAP["sicommon"],
        )


class SfaPointType(skxml.ArrayType):
    """
    Transcoder for double-precision floating point Simple Feature Access 2D or 3D Points.

    """

    def __init__(self) -> None:
        self._subelem_superset: dict[str, skxml.Type] = {
            c: skxml.DblType() for c in ("X", "Y", "Z")
        }
        super().__init__(subelements=self._subelem_superset, child_ns="urn:SFA:1.2.0")

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns an array containing the sub-elements encoded in ``elem``."""
        if len(elem) not in (2, 3):
            raise ValueError("Unexpected number of subelements (requires 2 or 3)")
        self.subelements = {
            k: v
            for idx, (k, v) in enumerate(self._subelem_superset.items())
            if idx < len(elem)
        }
        return super().parse_elem(elem)

    def set_elem(self, elem: lxml.etree.Element, val: Sequence[Any]) -> None:
        """Set ``elem`` node using ``val``."""
        if len(val) not in (2, 3):
            raise ValueError("Unexpected number of values (requires 2 or 3)")
        self.subelements = {
            k: v
            for idx, (k, v) in enumerate(self._subelem_superset.items())
            if idx < len(val)
        }
        super().set_elem(elem, val)


class LUTInfoType(skxml.Type):
    """
    Transcoder for LUTInfo nodes under LookupTableType's Custom child.

    """

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns an array containing the LUTs encoded in ``elem``."""
        return np.array(
            [
                IntListType().parse_elem(x)
                for x in sorted(elem, key=lambda x: int(x.get("lut")))
            ]
        )

    def set_elem(self, elem: lxml.etree.Element, val: Sequence[Any]) -> None:
        """Set ``elem`` node using ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : array_like
            (numLuts, size)-shaped array of LUTs to set

        """
        elem[:] = []
        elem_ns = lxml.etree.QName(elem).namespace
        ns = f"{{{elem_ns}}}" if elem_ns else ""
        luts = np.asarray(val)
        elem.set("numLuts", str(luts.shape[0]))
        elem.set("size", str(luts.shape[1]))
        for index, sub_val in enumerate(luts):
            subelem = lxml.etree.SubElement(elem, ns + "LUTValues")
            IntListType().set_elem(subelem, sub_val)
            subelem.set("lut", str(index + 1))


class XmlHelper(skxml2.XmlHelper):
    """
    XmlHelper for Sensor Independent Derived Data (SIDD).

    """

    def __init__(self, element_tree):
        root_ns = lxml.etree.QName(element_tree.getroot()).namespace
        super().__init__(element_tree, _xmlhelp2.XsdHelper(root_ns))
