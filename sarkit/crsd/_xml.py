"""
Functions for interacting with CRSD XML
"""

import importlib.resources
import pathlib

import lxml.etree

import sarkit._xmlhelp as skxml
import sarkit._xmlhelp2 as skxml2
import sarkit.cphd as skcphd

from . import _constants as crsdconst


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


# PxP/APxP are below


@skxml.inheritdocstring
class MtxType(skxml.MtxType):
    pass


@skxml.inheritdocstring
class ParameterType(skxml.ParameterType):
    pass


# The following transcoders happen to share common implementations with CPHD
class PxpType(skcphd.PvpType):
    """Transcoder for Per-x-Parameter (PxP) XML parameter types."""


class AddedPxpType(skcphd.AddedPvpType):
    """Transcoder for Added Per-x-Parameter (APxP) XML parameter types."""


@skxml.inheritdocstring
class ImageAreaCornerPointsType(skcphd.ImageAreaCornerPointsType):
    pass


class EdfType(skxml.SequenceType):
    """
    Transcoder for Error Decorrelation Function (EDF) XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__(
            subelements={c: skxml.DblType() for c in ("CorrCoefZero", "DecorrRate")}
        )

    def parse_elem(self, elem) -> tuple[float, float]:
        """Returns (CorrCoefZero, DecorrRate) values encoded in ``elem``."""
        return tuple(super().parse_subelements(elem).values())

    def set_elem(self, elem, val: tuple[float, float]) -> None:
        """Set children of ``elem`` from tuple: (``CorrCoefZero``, ``DecorrRate``)."""
        super().set_subelements(elem, {"CorrCoefZero": val[0], "DecorrRate": val[1]})


class XmlHelper(skxml2.XmlHelper):
    """
    XmlHelper for Compensated Radar Signal Data (CRSD).

    """

    def __init__(self, element_tree):
        root_ns = lxml.etree.QName(element_tree.getroot()).namespace
        super().__init__(element_tree, XsdHelper(root_ns))


class XsdHelper(skxml2.XsdHelper):
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
            ): skxml.NdArrayType("SV", LineSampType()),
            "<UNNAMED>-{http://api.nsgreg.nga.mil/schema/crsd/1.0}XYPolygonType/{http://api.nsgreg.nga.mil/schema/crsd/1.0}Vertex": XyType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}ErrorDecorrFuncType": EdfType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LSType": LineSampType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LSVertexType": LineSampType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonCornerRestrictionType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonHAERestrictionType": LatLonHaeType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonPolygonType": skxml.NdArrayType(
                "Vertex", LatLonType()
            ),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonRestrictionType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LatLonType": LatLonType(),
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}LineType": skxml.NdArrayType(
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
            "{http://api.nsgreg.nga.mil/schema/crsd/1.0}XYPolygonType": skxml.NdArrayType(
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
