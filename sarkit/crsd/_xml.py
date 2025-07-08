"""
Functions for interacting with CRSD XML
"""

import importlib.resources
import pathlib

import lxml.etree

import sarkit.cphd as skcphd
import sarkit.xmlhelp as skxml
import sarkit.xmlhelp._transcoders as skxt

from . import _constants as crsdconst


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


# PxP/APxP are below


@skxt.inheritdocstring
class MtxType(skxt.MtxType):
    pass


@skxt.inheritdocstring
class ParameterType(skxt.ParameterType):
    pass


# The following transcoders happen to share common implementations with CPHD
class PxpType(skcphd.PvpType):
    """Transcoder for Per-x-Parameter (PxP) XML parameter types."""


class AddedPxpType(skcphd.AddedPvpType):
    """Transcoder for Added Per-x-Parameter (APxP) XML parameter types."""


@skxt.inheritdocstring
class ImageAreaCornerPointsType(skcphd.ImageAreaCornerPointsType):
    pass


class EdfType(skxt.SequenceType):
    """
    Transcoder for Error Decorrelation Function (EDF) XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__(
            subelements={c: skxt.DblType() for c in ("CorrCoefZero", "DecorrRate")}
        )

    def parse_elem(self, elem) -> tuple[float, float]:
        """Returns (CorrCoefZero, DecorrRate) values encoded in ``elem``."""
        return tuple(super().parse_subelements(elem).values())

    def set_elem(self, elem, val: tuple[float, float]) -> None:
        """Set children of ``elem`` from tuple: (``CorrCoefZero``, ``DecorrRate``)."""
        super().set_subelements(elem, {"CorrCoefZero": val[0], "DecorrRate": val[1]})


class XmlHelper(skxml.XmlHelper):
    """
    :py:class:`~sarkit.xmlhelp.XmlHelper` for CRSD

    """

    def __init__(self, element_tree):
        root_ns = lxml.etree.QName(element_tree.getroot()).namespace
        super().__init__(element_tree, XsdHelper(root_ns))


class XsdHelper(skxml.XsdHelper):
    """
    :py:class:`~sarkit.xmlhelp.XsdHelper` for CRSD

    """

    def _read_xsdtypes_json(self, root_ns: str) -> str:
        """Return the text contents of the appropriate xsdtypes JSON"""
        schema_name = crsdconst.VERSION_INFO[root_ns]["schema"].name
        return importlib.resources.read_text(
            "sarkit.crsd.xsdtypes",
            pathlib.PurePath(schema_name).with_suffix(".json").name,
        )

    def get_transcoder(self, typename, tag=None):
        """Return the appropriate transcoder given the typename (and optionally tag)."""
        known_builtins = {
            "{http://www.w3.org/2001/XMLSchema}boolean": BoolType(),
            "{http://www.w3.org/2001/XMLSchema}double": DblType(),
            "{http://www.w3.org/2001/XMLSchema}hexBinary": HexType(),
            "{http://www.w3.org/2001/XMLSchema}integer": IntType(),
            "{http://www.w3.org/2001/XMLSchema}nonNegativeInteger": IntType(),
            "{http://www.w3.org/2001/XMLSchema}positiveInteger": IntType(),
            "{http://www.w3.org/2001/XMLSchema}string": TxtType(),
        }
        typedef = self.xsdtypes[typename]
        easy = {
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonPolygonType/{http://api.nsgreg.nga.mil/schema/crsd/1.0}Vertex": LatLonType(),
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/crsd/1.0}LineType/{http://api.nsgreg.nga.mil/schema/crsd/1.0}Endpoint": LatLonType(),
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/crsd/1.0}SceneCoordinatesBaseType/{http://api.nsgreg.nga.mil/schema/crsd/1.0}ImageAreaCornerPoints": ImageAreaCornerPointsType(),
            (
                "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/crsd/1.0}SceneCoordinatesSARType"
                "/{http://api.nsgreg.nga.mil/schema/crsd/1.0}ImageGrid"
                "/{http://api.nsgreg.nga.mil/schema/crsd/1.0}SegmentList"
                "/{http://api.nsgreg.nga.mil/schema/crsd/1.0}Segment"
                "/{http://api.nsgreg.nga.mil/schema/crsd/1.0}SegmentPolygon"
            ): skxt.NdArrayType("SV", LineSampType()),
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/crsd/1.0}XYPolygonType/{http://api.nsgreg.nga.mil/schema/crsd/1.0}Vertex": XyType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}ErrorDecorrFuncType": EdfType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LSType": LineSampType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LSVertexType": LineSampType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonCornerRestrictionType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonHAERestrictionType": LatLonHaeType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonPolygonType": skxt.NdArrayType(
                "Vertex", LatLonType()
            ),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonRestrictionType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LineType": skxt.NdArrayType(
                "Endpoint", LatLonType()
            ),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}Matrix2x2Type": MtxType((2, 2)),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}Matrix3x3Type": MtxType((3, 3)),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}Matrix4x4Type": MtxType((4, 4)),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}Matrix6x6Type": MtxType((6, 6)),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}ParameterType": ParameterType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}PerParameterEB": PxpType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}PerParameterF8": PxpType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}PerParameterI8": PxpType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}PerParameterIntFrac": PxpType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}PerParameterXYZ": PxpType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}Poly1DType": PolyType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}Poly2DType": Poly2dType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}UserDefinedPxPType": AddedPxpType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}XDTType": XdtType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}XYPolygonType": skxt.NdArrayType(
                "Vertex", XyType()
            ),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}XYType": XyType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}XYZType": XyzType(),
        }
        if typename.startswith("{http://www.w3.org/2001/XMLSchema}"):
            return known_builtins[typename]
        if typename in easy:
            return easy[typename]
        if not typedef.children:
            return known_builtins.get(typedef.text_typename, TxtType())
        return None


class ElementWrapper(skxml.ElementWrapper):
    """:py:class:`~sarkit.xmlhelp.ElementWrapper` for CRSD that can set ``xsdhelper`` automatically.

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
