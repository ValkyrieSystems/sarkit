import importlib.resources
import pathlib

import sarkit._xmlhelp
import sarkit._xmlhelp2
import sarkit.sidd._xml as sksiddxml

from . import _constants as siddconst


class XmlHelper2(sarkit._xmlhelp2.XmlHelper2):
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
            "{urn:SICommon:1.0}XYZType": sksiddxml.XyzType(),
            "{urn:SICommon:1.0}Poly1DType": sksiddxml.PolyCoef1dType(),
            "{urn:SICommon:1.0}Poly2DType": sksiddxml.PolyCoef2dType(),
            "{urn:SICommon:1.0}XYZPolyType": sksiddxml.XyzPolyType(),
            "{urn:SICommon:1.0}AngleZeroToExclusive360MagnitudeType": sksiddxml.AngleMagnitudeType(),
            "{urn:SICommon:1.0}LatLonType": sksiddxml.LatLonType(),
            "{urn:SIDD:3.0.0}PolygonType": sarkit._xmlhelp.ListType(
                "Vertex", sksiddxml.LatLonType()
            ),
            "{urn:SICommon:1.0}ParameterType": sksiddxml.ParameterType(),
        }
        if typename.startswith("{http://www.w3.org/2001/XMLSchema}"):
            return known_builtins[typename]
        if typename in easy:
            return easy[typename]
        if not typedef.children and not typedef.attributes:
            return known_builtins.get(typedef.text_typename, sksiddxml.TxtType())
        return None
