"""
Functions for interacting with CPHD XML
"""

import copy
import importlib.resources
import pathlib
from collections.abc import Sequence

import lxml.etree

import sarkit._xmlhelp as skxml
import sarkit._xmlhelp2 as skxml2
import sarkit.cphd._io as cphd_io

from . import _constants as cphdconst


# The following transcoders happen to share common implementation across several standards
@skxml.inheritdocstring
class TxtType(skxml.TxtType):
    pass


@skxml.inheritdocstring
class EnuType(skxml.EnuType):
    pass


@skxml.inheritdocstring
class BoolType(skxml.BoolType):
    pass


@skxml.inheritdocstring
class XdtType(skxml.XdtType):
    pass


@skxml.inheritdocstring
class IntType(skxml.IntType):
    pass


@skxml.inheritdocstring
class DblType(skxml.DblType):
    pass


@skxml.inheritdocstring
class HexType(skxml.HexType):
    pass


@skxml.inheritdocstring
class LineSampType(skxml.LineSampType):
    pass


@skxml.inheritdocstring
class XyType(skxml.XyType):
    pass


@skxml.inheritdocstring
class XyzType(skxml.XyzType):
    pass


@skxml.inheritdocstring
class LatLonType(skxml.LatLonType):
    pass


@skxml.inheritdocstring
class LatLonHaeType(skxml.LatLonHaeType):
    pass


@skxml.inheritdocstring
class PolyType(skxml.PolyType):
    pass


@skxml.inheritdocstring
class Poly2dType(skxml.Poly2dType):
    pass


@skxml.inheritdocstring
class XyzPolyType(skxml.XyzPolyType):
    pass


@skxml.inheritdocstring
class ParameterType(skxml.ParameterType):
    pass


class ImageAreaCornerPointsType(skxml.NdArrayType):
    """
    Transcoder for CPHD-like SceneCoordinates/ImageAreaCornerPoints XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__("IACP", skxml.LatLonType(), include_size_attr=False)

    def set_elem(
        self, elem: lxml.etree.Element, val: Sequence[Sequence[float]]
    ) -> None:
        """Set the IACP children of ``elem`` using the ordered vertices from ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : (4, 2) array_like
            Array of [latitude (deg), longitude (deg)] image corners.

        """
        if len(val) != 4:
            raise ValueError(f"Must have 4 corner points (given {len(val)})")
        super().set_elem(elem, val)


class PvpType(skxml.SequenceType):
    """
    Transcoder for per-vector parameter (PVP) XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__(
            {
                "Offset": skxml.IntType(),
                "Size": skxml.IntType(),
                "Format": skxml.TxtType(),
            }
        )

    def parse_elem(self, elem: lxml.etree.Element) -> dict:
        """Returns a dict containing the sequence of subelements encoded in ``elem``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to parse

        Returns
        -------
        elem_dict : dict
            Subelement values by name:

            * "Name" : `str` (`AddedPvpType` only)
            * "Offset" : `int`
            * "Size" : `int`
            * "dtype" : `numpy.dtype`
        """
        elem_dict = super().parse_subelements(elem)
        elem_dict["dtype"] = cphd_io.binary_format_string_to_dtype(elem_dict["Format"])
        del elem_dict["Format"]
        return elem_dict

    def set_elem(self, elem: lxml.etree.Element, val: dict) -> None:
        """Sets ``elem`` node using the sequence of subelements in the dict ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : dict
            Subelement values by name:

            * "Name" : `str` (`AddedPvpType` only)
            * "Offset" : `int`
            * "Size" : `int`
            * "dtype" : `numpy.dtype`
        """
        local_val = copy.deepcopy(val)
        local_val["Format"] = cphd_io.dtype_to_binary_format_string(local_val["dtype"])
        del local_val["dtype"]
        super().set_subelements(elem, local_val)


class AddedPvpType(PvpType):
    """
    Transcoder for added per-vector parameter (APVP) XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__()
        self.subelements = {"Name": skxml.TxtType(), **self.subelements}


class XmlHelper(skxml2.XmlHelper):
    """
    XmlHelper for Compensated Phase History Data (CPHD).

    """

    def __init__(self, element_tree):
        root_ns = lxml.etree.QName(element_tree.getroot()).namespace
        super().__init__(element_tree, XsdHelper(root_ns))


class XsdHelper(skxml2.XsdHelper):
    def _read_xsdtypes_json(self, root_ns: str) -> str:
        """Return the text contents of the appropriate xsdtypes JSON"""
        schema_name = cphdconst.VERSION_INFO[root_ns]["schema"].name
        return importlib.resources.read_text(
            "sarkit.cphd.xsdtypes",
            pathlib.PurePath(schema_name).with_suffix(".json").name,
        )

    def get_transcoder(self, typename, tag=None):
        """Return the appropriate transcoder given the typename (and optionally tag)."""
        known_builtins = {
            "{http://www.w3.org/2001/XMLSchema}boolean": BoolType(),
            "{http://www.w3.org/2001/XMLSchema}double": DblType(),
            "{http://www.w3.org/2001/XMLSchema}dateTime": XdtType(),
            "{http://www.w3.org/2001/XMLSchema}hexBinary": HexType(),
            "{http://www.w3.org/2001/XMLSchema}integer": IntType(),
            "{http://www.w3.org/2001/XMLSchema}nonNegativeInteger": IntType(),
            "{http://www.w3.org/2001/XMLSchema}positiveInteger": IntType(),
            "{http://www.w3.org/2001/XMLSchema}string": TxtType(),
        }
        typedef = self.xsdtypes[typename]
        cphd_101 = {
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonPolygonType/{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}Vertex": LatLonType(),
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LineType/{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}Endpoint": LatLonType(),
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}SceneCoordinatesType/{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}ImageAreaCornerPoints": ImageAreaCornerPointsType(),
            (
                "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}SceneCoordinatesType"
                "/{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}ImageGrid"
                "/{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}SegmentList"
                "/{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}Segment"
                "/{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}SegmentPolygon"
            ): skxml.NdArrayType("SV", LineSampType()),
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}XYPolygonType/{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}Vertex": XyType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LSType": LineSampType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LSVertexType": LineSampType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonCornerRestrictionType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonHAERestrictionType": LatLonHaeType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonPolygonType": skxml.NdArrayType(
                "Vertex", LatLonType()
            ),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonRestrictionType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LineType": skxml.NdArrayType(
                "Endpoint", LatLonType()
            ),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}ParameterType": ParameterType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}PerVectorParameterF8": PvpType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}PerVectorParameterI8": PvpType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}PerVectorParameterXYZ": PvpType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}Poly1DType": PolyType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}Poly2DType": Poly2dType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}UserDefinedPVPType": AddedPvpType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}XYPolygonType": skxml.NdArrayType(
                "Vertex", XyType()
            ),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}XYType": XyType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}XYZType": XyzType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}XYZPolyType": XyzPolyType(),
        }
        cphd_110 = {
            k.replace(
                "http://api.nsgreg.nga.mil/schema/cphd/1.0.1",
                "http://api.nsgreg.nga.mil/schema/cphd/1.1.0",
            ): v
            for k, v in cphd_101.items()
        }
        cphd_110 |= {
            "{http://api.nsgreg.nga.mil/schema/cphd/1.1.0}PerVectorParameterEB": PvpType(),
        }
        easy = cphd_101 | cphd_110
        if typename.startswith("{http://www.w3.org/2001/XMLSchema}"):
            return known_builtins[typename]
        if typename in easy:
            return easy[typename]
        if not typedef.children:
            return known_builtins.get(typedef.text_typename, TxtType())
        return None
