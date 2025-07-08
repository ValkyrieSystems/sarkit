"""
Functions for interacting with CPHD XML
"""

import copy
import importlib.resources
import pathlib
from collections.abc import Sequence

import lxml.etree

import sarkit.cphd._io as cphd_io
import sarkit.xmlhelp as skxml
import sarkit.xmlhelp._transcoders as skxt

from . import _constants as cphdconst


# The following transcoders happen to share common implementation across several standards
@skxt.inheritdocstring
class TxtType(skxt.TxtType):
    pass


@skxt.inheritdocstring
class EnuType(skxt.EnuType):
    pass


@skxt.inheritdocstring
class BoolType(skxt.BoolType):
    pass


@skxt.inheritdocstring
class XdtType(skxt.XdtType):
    pass


@skxt.inheritdocstring
class IntType(skxt.IntType):
    pass


@skxt.inheritdocstring
class DblType(skxt.DblType):
    pass


@skxt.inheritdocstring
class HexType(skxt.HexType):
    pass


@skxt.inheritdocstring
class LineSampType(skxt.LineSampType):
    pass


@skxt.inheritdocstring
class XyType(skxt.XyType):
    pass


@skxt.inheritdocstring
class XyzType(skxt.XyzType):
    pass


@skxt.inheritdocstring
class LatLonType(skxt.LatLonType):
    pass


@skxt.inheritdocstring
class LatLonHaeType(skxt.LatLonHaeType):
    pass


@skxt.inheritdocstring
class PolyType(skxt.PolyType):
    pass


@skxt.inheritdocstring
class Poly2dType(skxt.Poly2dType):
    pass


@skxt.inheritdocstring
class XyzPolyType(skxt.XyzPolyType):
    pass


@skxt.inheritdocstring
class ParameterType(skxt.ParameterType):
    pass


class ImageAreaCornerPointsType(skxt.NdArrayType):
    """
    Transcoder for CPHD-like SceneCoordinates/ImageAreaCornerPoints XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__("IACP", skxt.LatLonType(), include_size_attr=False)

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


class PvpType(skxt.SequenceType):
    """
    Transcoder for per-vector parameter (PVP) XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__(
            {
                "Offset": skxt.IntType(),
                "Size": skxt.IntType(),
                "Format": skxt.TxtType(),
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
        self.subelements = {"Name": skxt.TxtType(), **self.subelements}


class XmlHelper(skxml.XmlHelper):
    """
    :py:class:`~sarkit.xmlhelp.XmlHelper` for CPHD

    """

    def __init__(self, element_tree):
        root_ns = lxml.etree.QName(element_tree.getroot()).namespace
        super().__init__(element_tree, XsdHelper(root_ns))


class XsdHelper(skxml.XsdHelper):
    """
    :py:class:`~sarkit.xmlhelp.XsdHelper` for CPHD

    """

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
            ): skxt.NdArrayType("SV", LineSampType()),
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}XYPolygonType/{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}Vertex": XyType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LSType": LineSampType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LSVertexType": LineSampType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonCornerRestrictionType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonHAERestrictionType": LatLonHaeType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonPolygonType": skxt.NdArrayType(
                "Vertex", LatLonType()
            ),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonRestrictionType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LatLonType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}LineType": skxt.NdArrayType(
                "Endpoint", LatLonType()
            ),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}ParameterType": ParameterType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}PerVectorParameterF8": PvpType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}PerVectorParameterI8": PvpType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}PerVectorParameterXYZ": PvpType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}Poly1DType": PolyType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}Poly2DType": Poly2dType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}UserDefinedPVPType": AddedPvpType(),
            "{http://api.nsgreg.nga.mil/schema/cphd/1.0.1}XYPolygonType": skxt.NdArrayType(
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


class ElementWrapper(skxml.ElementWrapper):
    """:py:class:`~sarkit.xmlhelp.ElementWrapper` for CPHD that can set ``xsdhelper`` automatically.

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
