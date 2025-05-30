import importlib.resources
import pathlib

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
        if typename.startswith("{http://www.w3.org/2001/XMLSchema}"):
            return known_builtins[typename]
        if typename == "{urn:SICommon:1.0}Poly1DType":
            return sksiddxml.PolyCoef1dType()
        if typename == "{urn:SICommon:1.0}Poly2DType":
            return sksiddxml.PolyCoef2dType()
        if typename == "{urn:SICommon:1.0}XYZPolyType":
            return sksiddxml.XyzPolyType()
        if not typedef.children and not typedef.attributes:
            return known_builtins.get(typedef.text_typename, sksiddxml.TxtType())
        if typedef.children or typedef.attributes:
            return sarkit._xmlhelp2.DictType(typedef, self)
        return None
