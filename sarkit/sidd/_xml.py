"""
Functions for interacting with SIDD XML
"""

import importlib.resources
import numbers
import pathlib
from collections.abc import Sequence
from typing import Any

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.xmlhelp as skxml
import sarkit.xmlhelp._transcoders as skxt

from . import _constants as siddconst

NSMAP = {
    "sicommon": "urn:SICommon:1.0",
}


# The following transcoders happen to share common implementation across several standards
@skxt.inheritdocstring
class BoolType(skxt.BoolType):
    pass


@skxt.inheritdocstring
class DblType(skxt.DblType):
    pass


@skxt.inheritdocstring
class EnuType(skxt.EnuType):
    pass


@skxt.inheritdocstring
class IntType(skxt.IntType):
    pass


@skxt.inheritdocstring
class TxtType(skxt.TxtType):
    pass


@skxt.inheritdocstring
class XdtType(skxt.XdtType):
    pass


class XyzType(skxt.XyzType):
    """Transcoder for XML parameter types containing scalar X, Y, and Z components.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class AngleMagnitudeType(skxt.ArrayType):
    """Transcoder for double-precision floating point angle magnitude XML parameter type.

    Children are in the SICommon namespace.
    """

    def __init__(self) -> None:
        super().__init__(
            subelements={c: skxt.DblType() for c in ("Angle", "Magnitude")},
            child_ns=NSMAP["sicommon"],
        )


class LatLonType(skxt.LatLonType):
    """Transcoder for XML parameter types containing scalar Lat and Lon components.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


@skxt.inheritdocstring
class ParameterType(skxt.ParameterType):
    pass


class PolyCoef1dType(skxt.PolyType):
    """Transcoder for one-dimensional polynomial (PolyCoef1D) XML parameter types.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class PolyCoef2dType(skxt.Poly2dType):
    """Transcoder for two-dimensional polynomial (PolyCoef2D) XML parameter types.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class RowColIntType(skxt.RowColType):
    """Transcoder for XML parameter types containing scalar, integer Row and Col components (RC_INT).

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class XyzPolyType(skxt.XyzPolyType):
    """Transcoder for XYZ_POLY XML parameter types containing triplets of 1D polynomials.

    Children are in the SICommon namespace.
    """

    def __init__(self):
        super().__init__(child_ns=NSMAP["sicommon"])


class FilterCoefficientType(skxt.Type):
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


class IntListType(skxt.Type):
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


class ImageCornersType(skxt.NdArrayType):
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


class RangeAzimuthType(skxt.ArrayType):
    """
    Transcoder for double-precision floating point range and azimuth XML parameter types.

    Children are in the SICommon namespace.

    """

    def __init__(self) -> None:
        super().__init__(
            subelements={c: skxt.DblType() for c in ("Range", "Azimuth")},
            child_ns=NSMAP["sicommon"],
        )


class RowColDblType(skxt.ArrayType):
    """
    Transcoder for double-precision floating point row and column XML parameter types.

    Children are in the SICommon namespace.

    """

    def __init__(self) -> None:
        super().__init__(
            subelements={c: skxt.DblType() for c in ("Row", "Col")},
            child_ns=NSMAP["sicommon"],
        )


class SfaPointType(skxt.ArrayType):
    """
    Transcoder for double-precision floating point Simple Feature Access 2D or 3D Points.

    """

    def __init__(self) -> None:
        self._subelem_superset: dict[str, skxt.Type] = {
            c: skxt.DblType() for c in ("X", "Y", "Z")
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


class LUTInfoType(skxt.Type):
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


class XmlHelper(skxml.XmlHelper):
    """
    :py:class:`~sarkit.xmlhelp.XmlHelper` for SIDD

    """

    def __init__(self, element_tree):
        root_ns = lxml.etree.QName(element_tree.getroot()).namespace
        super().__init__(element_tree, XsdHelper(root_ns))


class XsdHelper(skxml.XsdHelper):
    """
    :py:class:`~sarkit.xmlhelp.XsdHelper` for SIDD

    """

    def _read_xsdtypes_json(self, root_ns: str) -> str:
        """Return the text contents of the appropriate xsdtypes JSON"""
        schema_name = siddconst.VERSION_INFO[root_ns]["schema"].name
        return importlib.resources.read_text(
            "sarkit.sidd.xsdtypes",
            pathlib.PurePath(schema_name).with_suffix(".json").name,
        )

    def get_transcoder(self, typename, tag=None):
        """Return the appropriate transcoder given the typename (and optionally tag)."""
        known_builtins = {
            "{http://www.w3.org/2001/XMLSchema}string": TxtType(),
            "{http://www.w3.org/2001/XMLSchema}dateTime": XdtType(),
            "{http://www.w3.org/2001/XMLSchema}int": IntType(),
            "{http://www.w3.org/2001/XMLSchema}double": DblType(),
        }
        typedef = self.xsdtypes[typename]
        easy = {
            "{urn:SFA:1.2.0}PointType": SfaPointType(),
            "{urn:SICommon:1.0}AngleZeroToExclusive360MagnitudeType": AngleMagnitudeType(),
            "{urn:SICommon:1.0}LatLonRestrictionType": LatLonType(),
            "{urn:SICommon:1.0}LatLonType": LatLonType(),
            "{urn:SICommon:1.0}LatLonVertexType": LatLonType(),
            "{urn:SICommon:1.0}LineType": skxt.NdArrayType("Endpoint", LatLonType()),
            "{urn:SICommon:1.0}ParameterType": ParameterType(),
            "{urn:SICommon:1.0}Poly1DType": PolyCoef1dType(),
            "{urn:SICommon:1.0}Poly2DType": PolyCoef2dType(),
            "{urn:SICommon:1.0}PolygonType": skxt.NdArrayType("Vertex", LatLonType()),
            "{urn:SICommon:1.0}RangeAzimuthType": RangeAzimuthType(),
            "{urn:SICommon:1.0}RowColDoubleType": RowColDblType(),
            "{urn:SICommon:1.0}RowColIntType": RowColIntType(),
            "{urn:SICommon:1.0}RowColVertexType": RowColIntType(),
            "{urn:SICommon:1.0}XYZPolyType": XyzPolyType(),
            "{urn:SICommon:1.0}XYZType": XyzType(),
            "<UNNAMED>-{urn:SICommon:1.0}LineType/{urn:SICommon:1.0}Endpoint": LatLonType(),
            "<UNNAMED>-{urn:SICommon:1.0}PolygonType/{urn:SICommon:1.0}Vertex": LatLonType(),
            "{urn:SIDD:3.0.0}FilterBankCoefType": FilterCoefficientType("phasingpoint"),
            "{urn:SIDD:3.0.0}FilterKernelCoefType": FilterCoefficientType("rowcol"),
            "{urn:SIDD:3.0.0}ImageCornersType": ImageCornersType(),
            "{urn:SIDD:3.0.0}LookupTableType": IntListType(),
            "{urn:SIDD:3.0.0}LUTInfoType": LUTInfoType(),
            "{urn:SIDD:3.0.0}PolygonType": skxt.NdArrayType("Vertex", LatLonType()),
            "{urn:SIDD:3.0.0}ValidDataType": skxt.NdArrayType(
                "Vertex", RowColIntType()
            ),
            "<UNNAMED>-{urn:SIDD:3.0.0}ImageCornersType/{urn:SIDD:3.0.0}ICP": LatLonType(),
        }
        easy["{urn:SIDD:2.0.0}FilterBankCoefType"] = easy[
            "{urn:SIDD:3.0.0}FilterBankCoefType"
        ]
        easy["{urn:SIDD:2.0.0}FilterKernelCoefType"] = easy[
            "{urn:SIDD:3.0.0}FilterKernelCoefType"
        ]
        easy["{urn:SIDD:2.0.0}ImageCornersType"] = easy[
            "{urn:SIDD:3.0.0}ImageCornersType"
        ]
        easy["{urn:SIDD:2.0.0}LookupTableType"] = easy[
            "{urn:SIDD:3.0.0}LookupTableType"
        ]
        easy["{urn:SIDD:2.0.0}LUTInfoType"] = easy["{urn:SIDD:3.0.0}LUTInfoType"]
        easy["{urn:SIDD:2.0.0}PolygonType"] = easy["{urn:SIDD:3.0.0}PolygonType"]
        easy["{urn:SIDD:2.0.0}ValidDataType"] = easy["{urn:SIDD:3.0.0}ValidDataType"]
        easy["<UNNAMED>-{urn:SIDD:2.0.0}ImageCornersType/{urn:SIDD:2.0.0}ICP"] = easy[
            "<UNNAMED>-{urn:SIDD:3.0.0}ImageCornersType/{urn:SIDD:3.0.0}ICP"
        ]

        if tag in ("{urn:SIDD:2.0.0}LocalDateTime", "{urn:SIDD:3.0.0}LocalDateTime"):
            return skxt.XdtType(force_utc=False)
        if typename.startswith("{http://www.w3.org/2001/XMLSchema}"):
            return known_builtins[typename]
        if typename in easy:
            return easy[typename]
        if not typedef.children:
            return known_builtins.get(typedef.text_typename, TxtType())
        return None


class ElementWrapper(skxml.ElementWrapper):
    """:py:class:`~sarkit.xmlhelp.ElementWrapper` for SIDD that can set ``xsdhelper`` automatically.

    Refer to :py:class:`sarkit.xmlhelp.ElementWrapper` for full documentation.
    """

    def __init__(
        self,
        elem,
        xsdhelper=None,
        wrapped_parent=None,
        typename=None,
        elementpath=None,
        roottag=None,
    ):
        if xsdhelper is None:
            root_ns = lxml.etree.QName(roottag or elem).namespace
            xsdhelper = XsdHelper(root_ns)
        super().__init__(
            elem, xsdhelper, wrapped_parent, typename, elementpath, roottag
        )
