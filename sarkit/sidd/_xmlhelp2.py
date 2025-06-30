import importlib.resources
import pathlib

import sarkit._xmlhelp
import sarkit._xmlhelp2
import sarkit.sidd._xml as sksiddxml

from . import _constants as siddconst


class XsdHelper(sarkit._xmlhelp2.XsdHelper):
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
            "{http://www.w3.org/2001/XMLSchema}string": sksiddxml.TxtType(),
            "{http://www.w3.org/2001/XMLSchema}dateTime": sksiddxml.XdtType(),
            "{http://www.w3.org/2001/XMLSchema}int": sksiddxml.IntType(),
            "{http://www.w3.org/2001/XMLSchema}double": sksiddxml.DblType(),
        }
        typedef = self.xsdtypes[typename]
        easy = {
            "{urn:SFA:1.2.0}PointType": sksiddxml.SfaPointType(),
            "{urn:SICommon:1.0}AngleZeroToExclusive360MagnitudeType": sksiddxml.AngleMagnitudeType(),
            "{urn:SICommon:1.0}LatLonRestrictionType": sksiddxml.LatLonType(),
            "{urn:SICommon:1.0}LatLonType": sksiddxml.LatLonType(),
            "{urn:SICommon:1.0}LatLonVertexType": sksiddxml.LatLonType(),
            "{urn:SICommon:1.0}LineType": sarkit._xmlhelp.NdArrayType(
                "Endpoint", sksiddxml.LatLonType()
            ),
            "{urn:SICommon:1.0}ParameterType": sksiddxml.ParameterType(),
            "{urn:SICommon:1.0}Poly1DType": sksiddxml.PolyCoef1dType(),
            "{urn:SICommon:1.0}Poly2DType": sksiddxml.PolyCoef2dType(),
            "{urn:SICommon:1.0}PolygonType": sarkit._xmlhelp.NdArrayType(
                "Vertex", sksiddxml.LatLonType()
            ),
            "{urn:SICommon:1.0}RangeAzimuthType": sksiddxml.RangeAzimuthType(),
            "{urn:SICommon:1.0}RowColDoubleType": sksiddxml.RowColDblType(),
            "{urn:SICommon:1.0}RowColIntType": sksiddxml.RowColIntType(),
            "{urn:SICommon:1.0}RowColVertexType": sksiddxml.RowColIntType(),
            "{urn:SICommon:1.0}XYZPolyType": sksiddxml.XyzPolyType(),
            "{urn:SICommon:1.0}XYZType": sksiddxml.XyzType(),
            "<UNNAMED>-{urn:SICommon:1.0}LineType/{urn:SICommon:1.0}Endpoint": sksiddxml.LatLonType(),
            "<UNNAMED>-{urn:SICommon:1.0}PolygonType/{urn:SICommon:1.0}Vertex": sksiddxml.LatLonType(),
            "{urn:SIDD:3.0.0}FilterBankCoefType": sksiddxml.FilterCoefficientType(
                "phasingpoint"
            ),
            "{urn:SIDD:3.0.0}FilterKernelCoefType": sksiddxml.FilterCoefficientType(
                "rowcol"
            ),
            "{urn:SIDD:3.0.0}ImageCornersType": sksiddxml.ImageCornersType(),
            "{urn:SIDD:3.0.0}LookupTableType": sksiddxml.IntListType(),
            "{urn:SIDD:3.0.0}LUTInfoType": sksiddxml.LUTInfoType(),
            "{urn:SIDD:3.0.0}PolygonType": sarkit._xmlhelp.NdArrayType(
                "Vertex", sksiddxml.LatLonType()
            ),
            "{urn:SIDD:3.0.0}ValidDataType": sarkit._xmlhelp.NdArrayType(
                "Vertex", sksiddxml.RowColIntType()
            ),
            "<UNNAMED>-{urn:SIDD:3.0.0}ImageCornersType/{urn:SIDD:3.0.0}ICP": sksiddxml.LatLonType(),
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

        if typename.startswith("{http://www.w3.org/2001/XMLSchema}"):
            return known_builtins[typename]
        if typename in easy:
            return easy[typename]
        if not typedef.children:
            return known_builtins.get(typedef.text_typename, sksiddxml.TxtType())
        return None
