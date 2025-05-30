import abc
import dataclasses
import json
from typing import Any

import lxml.etree

import sarkit._xmlhelp


@dataclasses.dataclass
class ChildDef:
    tag: str
    typename: str
    repeat: bool = False


@dataclasses.dataclass
class XsdTypeDef:
    attributes: list[str] = dataclasses.field(default_factory=list)
    children: list[ChildDef] = dataclasses.field(default_factory=list)
    text_typename: str | None = None

    def get_childdef(self, tag):
        """Return the first ChildDef in children whose tag matches tag."""
        return next((cdef for cdef in self.children if cdef.tag == tag), None)


def dumps_xsdtypes(xsdtypes):
    return json.dumps(
        {k: dataclasses.asdict(v) for k, v in xsdtypes.items()},
        sort_keys=True,
        indent=2,
    )


def loads_xsdtypes(s: str):
    def as_dataclass(dct):
        for cls in (XsdTypeDef, ChildDef):
            try:
                return cls(**dct)
            except TypeError:
                continue
        return dct

    return json.loads(s, object_hook=as_dataclass)


class XmlHelper2(abc.ABC):
    def __init__(self, root_ns: str):
        xsdtypes_json_str = self._read_xsdtypes_json(root_ns)
        self.xsdtypes = loads_xsdtypes(xsdtypes_json_str)

    @abc.abstractmethod
    def _read_xsdtypes_json(self, root_ns: str) -> str:
        """Return the text contents of the appropriate xsdtypes JSON"""

    def get_typeinfo(self, elem: lxml.etree.Element):
        """Return the typename and typedef for a subelement"""
        elempath = elem.getroottree().getelementpath(elem)
        current_typedef = self.xsdtypes["/"]
        if elempath == ".":
            # special handling for root
            return ("/", current_typedef)
        for component in elempath.split("/"):
            comp_type = current_typedef.get_childdef(component.split("[")[0]).typename
            current_typedef = self.xsdtypes.get(comp_type)
        return comp_type, current_typedef

    @abc.abstractmethod
    def get_transcoder(self, typename, tag=None):
        """Return the appropriate transcoder given the typename (and optionally tag)."""

    def get_elem_transcoder(self, elem: lxml.etree.Element):
        return self.get_transcoder(self.get_typeinfo(elem)[0], tag=elem.tag)

    def load_elem(self, elem):
        """Decode ``elem`` (an XML element) to a Python object."""
        return self.get_elem_transcoder(elem).parse_elem(elem)

    def set_elem(self, elem, val):
        """Encode ``val`` (a Python object) into the XML element ``elem``."""
        self.get_elem_transcoder(elem).set_elem(elem, val)


class DictType(sarkit._xmlhelp.Type):
    """Transcoder for XML types that should be parsed to a dict.

    Known children and/or attributes not present during transcoding are ignored.
    Unknown children and/or attributes are ignored during parsing, but result in an exception during setting.
    The order of the dict keys during the setting is unimportant.

    This class, like other parts of the module, assumes that "repeatable" children are contiguous and share a type
    definition.
    """

    def __init__(
        self,
        typedef: XsdTypeDef,
        xmlhelper: XmlHelper2,
    ):
        self.typedef = typedef
        self._xmlhelper = xmlhelper

    def parse_elem(self, elem) -> dict[str, Any]:
        val = {
            "@" + k: elem.get(k) for k in self.typedef.attributes if k in elem.keys()
        }

        if not self.typedef.children:
            if len(elem) > 0:
                raise ValueError(f"{elem} not expected to have children")

            if self.typedef.text_typename:
                transcoder = self._xmlhelper.get_transcoder(
                    self.typedef.text_typename, tag=elem.tag
                )
                val["#text"] = transcoder.parse_elem(elem)

        for subelem in elem:
            subelem_localname = lxml.etree.QName(subelem).localname
            subelem_childdef = self.typedef.get_childdef(subelem.tag)
            subelem_transcoder = self._xmlhelper.get_transcoder(
                subelem_childdef.typename, subelem.tag
            )

            if subelem_childdef.repeat:
                val.setdefault(subelem_localname, []).append(
                    subelem_transcoder.parse_elem(subelem)
                )
            else:
                if subelem_localname in val:
                    raise ValueError(f"{subelem_localname} not expected to repeat")
                val[subelem_localname] = subelem_transcoder.parse_elem(subelem)

        return val

    def set_elem(self, elem, val):
        elem.clear()
        valcopy = val.copy()

        for possible_attr in self.typedef.attributes:
            attr_value = valcopy.pop(f"@{possible_attr}", None)
            if attr_value is not None:
                elem.set(possible_attr, attr_value)

        if not self.typedef.children:
            textval = valcopy.pop("#text")
            self._xmlhelper.get_transcoder(self.typedef.text_typename).set_elem(
                elem, textval
            )
        else:
            for childdef in self.typedef.children:
                child_value = valcopy.pop(
                    lxml.etree.QName(childdef.tag).localname, None
                )
                if child_value is not None:
                    subelem_transcoder = self._xmlhelper.get_transcoder(
                        childdef.typename, childdef.tag
                    )
                    if childdef.repeat:
                        for item in child_value:
                            new_child = elem.makeelement(childdef.tag)
                            elem.append(new_child)
                            subelem_transcoder.set_elem(new_child, item)
                    else:
                        new_child = elem.makeelement(childdef.tag)
                        elem.append(new_child)
                        subelem_transcoder.set_elem(new_child, child_value)

        unrecognized_keys = list(valcopy.keys())
        if unrecognized_keys:
            raise ValueError(f"{unrecognized_keys=}in val for {self.typedef}")
