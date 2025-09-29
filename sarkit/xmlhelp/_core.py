import abc
import collections.abc
import dataclasses
import json
import re

import lxml.etree


@dataclasses.dataclass
class ChildDef:
    """XSD child element definition"""

    tag: str
    typename: str
    repeat: bool = False


@dataclasses.dataclass
class XsdTypeDef:
    """XSD type definition"""

    attributes: list[str] = dataclasses.field(default_factory=list)
    children: list[ChildDef] = dataclasses.field(default_factory=list)
    text_typename: str | None = None

    def get_childdef(self, tag) -> ChildDef | None:
        """Return the first ChildDef in children whose tag matches tag."""
        return next((cdef for cdef in self.children if cdef.tag == tag), None)

    def get_childdef_from_localname(self, localname: str) -> ChildDef | None:
        """Return the `ChildDef` in ``children`` by localname (e.g. no namespace) or ``None``."""
        return next(
            (
                cdef
                for cdef in self.children
                if lxml.etree.QName(cdef.tag).localname == localname
            ),
            None,
        )

    def get_attribute_from_localname(self, localname: str) -> str | None:
        """Return attribute by localname (e.g. no namespace) or ``None``."""
        for attrib in self.attributes:
            if lxml.etree.QName(attrib).localname == localname:
                return attrib
        return None


def dumps_xsdtypes(xsdtypes):
    def _asdict(x):
        if dataclasses.is_dataclass(x):
            return dataclasses.asdict(x)
        return x

    return json.dumps(
        {k: _asdict(v) for k, v in xsdtypes.items()},
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


def split_elempath(elempath: str) -> list[str]:
    """Return an ordered list of an ElementPath's various elements (discarding positional predicates)."""
    return [
        m.group("elem")
        for m in re.finditer(r"(?P<elem>(\{.*?\})?[^/[]+)(\[\d+\])?/?", elempath)
    ]


class XsdHelper(abc.ABC):
    """Abstract base class that retrieves transcoders and type info for elements of a given XML schema.

    Parameters
    ----------
    root_ns : str
        Target namespace of the schema document
    """

    def __init__(self, root_ns: str):
        xsdtypes_json_str = self._read_xsdtypes_json(root_ns)
        self.xsdtypes = loads_xsdtypes(xsdtypes_json_str)

    @abc.abstractmethod
    def _read_xsdtypes_json(self, root_ns: str) -> str:
        """Return the text contents of the appropriate xsdtypes JSON"""

    def get_typeinfo(self, elempath: str, roottag: str):
        """Return the typename and typedef for a elementpath"""
        current_typename = self.xsdtypes["/"][roottag]
        current_typedef = self.xsdtypes[current_typename]
        if elempath == ".":
            # special handling for root
            return current_typename, current_typedef
        for component in split_elempath(elempath):
            current_typename = current_typedef.get_childdef(component).typename
            current_typedef = self.xsdtypes.get(current_typename)
        return current_typename, current_typedef

    def get_elem_typeinfo(self, elem: lxml.etree.Element):
        """Return the typename and typedef for a subelement"""
        return self.get_typeinfo(
            elem.getroottree().getelementpath(elem), elem.getroottree().getroot().tag
        )

    @abc.abstractmethod
    def get_transcoder(self, typename, tag=None):
        """Return the appropriate transcoder given the typename (and optionally tag)."""

    def get_elem_transcoder(self, elem: lxml.etree.Element):
        """Return the appropriate transcoder given an element."""
        return self.get_transcoder(self.get_elem_typeinfo(elem)[0], tag=elem.tag)


class _UNSET:
    """Sentinel class indicating a value is not set.  Allows for `None` as a valid value."""


class ElementWrapper(collections.abc.MutableMapping):
    """Wrapper for lxml.etree.Element that provides dictionary-ish interface

    Getting/setting of schema-valid subelements does not inherently raise an exception if they don't exist.
    Setter will create non-existent ancestors as necessary.
    Repeatable elements are treated as tuples.
    Transcoded values are copies, not references. Some effort has been made to make them immutable.
    If you manage to change them, the changes are not reflected in the underlying XML element.
    Attributes are accessed using BadgerFish notation (e.g. @attr).
    Keys are local names. Namespaces are handled automatically.

    Parameters
    ----------
    elem : lxml.etree.Element or None
        Element to wrap or ``None`` for a placeholder element.
        If ``elem`` is ``None``, ``elementpath`` must be set.
    xsdhelper : XsdHelper
        XML helper for the root namespace of the document that contains ``elem``
    wrapped_parent : ElementWrapper or None
        Wrapper for the parent of ``elem`` or ``None``. Only used when trying to set an item of a wrapped placeholder.
    typename : str or None
        XSD typename of the element being wrapped. If ``None``, the type information is retrieved from the XML Helper.
    elementpath : str or None
        ElementPath expression that identifies the location of ``elem`` in its tree.
        If ``None``, the path is retrieved from the element.
        Required if ``elem`` is ``None``.
    roottag : str or None
        Tag of the root element of ``elem``'s tree.
        If ``None``, the tag is retrieved from the element.
        Required if ``elem`` is ``None``.

    Notes
    -----
    The wrapped element must be located in a tree with a root element expected by ``xsdhelper``.
    """

    def __init__(
        self,
        elem: "lxml.etree.Element | None",
        xsdhelper: XsdHelper,
        wrapped_parent: "ElementWrapper | None" = None,
        typename: str | None = None,
        elementpath: str | None = None,
        roottag: str | None = None,
    ):
        self.elem = elem
        self.wrapped_parent = self if wrapped_parent is None else wrapped_parent
        if elementpath is None:
            assert elem is not None
            elementpath = elem.getroottree().getelementpath(elem)
        if roottag is None:
            assert elem is not None
            roottag = elem.getroottree().getroot().tag
        if typename is None:
            typename = xsdhelper.get_typeinfo(elementpath, roottag)[0]
        self.elementpath = elementpath
        self.roottag = roottag
        self.xsdhelper = xsdhelper
        self.typename = typename
        self.typedef = xsdhelper.xsdtypes[typename]

    def _getchilddef(self, localname):
        childdef = self.typedef.get_childdef_from_localname(localname)
        if childdef is None:
            raise KeyError(localname)
        return childdef

    def _getattribname(self, localname):
        attribname = self.typedef.get_attribute_from_localname(
            localname.removeprefix("@")
        )
        if attribname is None or self.elem is None:
            raise KeyError(localname)
        return attribname

    def _getelem(
        self, localname: str
    ) -> "None | lxml.etree.Element | list[lxml.etree.Element]":
        """Return an element or tuple of elements given a localname."""
        childdef = self._getchilddef(localname)
        if childdef.repeat:
            if self.elem is None:
                return tuple()
            return tuple(self.elem.findall("{*}" + localname))
        if self.elem is None:
            return None
        return self.elem.find("{*}" + localname)

    def __repr__(self):
        return f"ElementWrapper({str(dict(self))})"

    def __getitem__(self, localname: str):
        if localname.startswith("@"):
            attribname = self._getattribname(localname)
            assert self.elem is not None  # for mypy.  Case handled by _getattribname()
            return self.elem.get(attribname)

        elem = self._getelem(localname)
        if isinstance(elem, tuple):
            return tuple(self._handle_subelem(x, localname) for x in elem)
        return self._handle_subelem(elem, localname)

    def get(self, localname: str, default=_UNSET):
        """Return value from an ElementWrapper.

        If the localname is not schema-valid a KeyError is raised.
        Otherwise, return the value for localname if localname is in the ElementWrapper, else default.
        If default is not given it defaults to the behavior of __getitem__.

        """
        # ElementWrapper.__getitem__ returns an empty ElementWrapper for valid, missing keys
        if default is _UNSET or localname in self:
            return self[localname]

        return default

    def _handle_subelem(
        self, subelem: "lxml.etree.Element | None", subelem_localname: str
    ):
        """Retrieve a transcoded value (leaf) or wrapped element (branch) from a subelement."""
        childdef = self.typedef.get_childdef_from_localname(subelem_localname)
        elempath = self.elementpath + f"/{childdef.tag}"
        transcoder = self.xsdhelper.get_transcoder(childdef.typename, childdef.tag)
        if transcoder is None or subelem is None:
            return ElementWrapper(
                subelem,
                elementpath=elempath,
                roottag=self.roottag,
                wrapped_parent=self,
                typename=childdef.typename,
                xsdhelper=self.xsdhelper,
            )
        transcoded_val = transcoder.parse_elem(subelem)
        if isinstance(transcoded_val, list):
            return tuple(transcoded_val)
        if hasattr(transcoded_val, "setflags"):
            transcoded_val.setflags(write=False)
        return transcoded_val

    def _get_inserter(self, childtag):
        """Return a function that inserts the child element in the appropriate location."""
        successor = None
        for child in reversed(self.typedef.children):
            if child.tag == childtag:
                break
            this_child = self.elem.find(child.tag)
            if this_child is not None:
                successor = this_child

        appendfunc = self.elem.append if successor is None else successor.addprevious
        return appendfunc

    def __setitem__(self, localname: str, value):
        if self.elem is None:
            elemtag = split_elempath(self.elementpath)[-1]
            self.elem = lxml.etree.Element(elemtag)
            self.wrapped_parent[lxml.etree.QName(elemtag).localname] = self.elem

        if localname.startswith("@"):
            attribname = self._getattribname(localname)
            self.elem.set(attribname, str(value))
            return

        childdef = self._getchilddef(localname)

        transcoder = self.xsdhelper.get_transcoder(childdef.typename, childdef.tag)

        def _val_to_elem(val):
            if isinstance(val, lxml.etree._Element):
                return val
            if isinstance(val, ElementWrapper):
                return val.elem
            if transcoder is None:
                return lxml.etree.Element(childdef.tag)
            return transcoder.make_elem(childdef.tag, val)

        for subelem in self.elem.findall("{*}" + localname):
            self.elem.remove(subelem)

        appendfunc = self._get_inserter(childdef.tag)
        if childdef.repeat:
            for idx, val in enumerate(value):
                if isinstance(val, dict) and transcoder is None:
                    appendfunc(lxml.etree.Element(childdef.tag))
                    for k, v in val.items():
                        self[localname][idx][k] = v
                else:
                    appendfunc(_val_to_elem(val))
        else:
            if isinstance(value, dict) and transcoder is None:
                for k, v in value.items():
                    self[localname][k] = v
            else:
                appendfunc(_val_to_elem(value))

    def add(self, localname, val=None):
        """Add a new subelement and optionally set its value.

        Useful for adding a repeatable subelement to work around the tuple from __getitem__.

        Parameters
        ----------
        localname : str
            Local name of element to add
        val
            Value to set new subelement to.

        Returns
        -------
        Any
            the new transcoded or wrapped subelement
        """
        childdef = self._getchilddef(localname)

        setval = val if val is not None else {}

        if childdef.repeat:
            self[localname] += (setval,)
            return self[localname][-1]
        if localname in self:
            raise ValueError(f"{localname} already exists")
        self[localname] = setval
        return self[localname]

    def __delitem__(self, localname):
        if localname.startswith("@"):
            attribname = self._getattribname(localname)
            del self.elem.attrib[attribname]
        else:
            childdef = self._getchilddef(localname)
            for subelem in self.elem.findall(childdef.tag):
                self.elem.remove(subelem)

    def _keys(self):
        keys = []
        if self.elem is not None:
            for attribname in self.elem.keys():
                localname = lxml.etree.QName(attribname).localname
                if self.typedef.get_attribute_from_localname(localname) is not None:
                    keys.append("@" + localname)

            keys.sort()
            for subelem in self.elem:
                localname = lxml.etree.QName(subelem).localname
                if localname not in keys and (
                    self.typedef.get_childdef_from_localname(localname) is not None
                ):
                    keys.append(localname)
        return keys

    def __iter__(self):
        return iter(self._keys())

    def __len__(self):
        return len(self._keys())

    def __contains__(self, localname):
        # Make sure localname is valid
        if localname.startswith("@"):
            _ = self._getattribname(localname)
        else:
            _ = self._getchilddef(localname)

        return localname in self._keys()

    def to_dict(self) -> dict:
        """Recursively convert the ElementWrapper to a dictionary."""

        def convert(v):
            if isinstance(v, ElementWrapper):
                return v.to_dict()
            if isinstance(v, tuple):
                return tuple(map(convert, v))
            return v

        return {k: convert(v) for k, v in self.items()}

    def from_dict(self, val):
        """Populate the ElementWrapper with the contents of a dictionary.

        Similar to ``dict.update``
        """
        for k, v in val.items():
            self[k] = v


class XmlHelper:
    """
    Base Class for generic XmlHelpers, which provide methods for transcoding data
    between XML and more convenient Python objects.

    Parameters
    ----------
    element_tree : lxml.etree.ElementTree
        An XML element tree containing the data being operated on.
    xsdhelper : XsdHelper
        XsdHelper object corresponding to ``element_tree``'s schema
    """

    def __init__(self, element_tree, xsdhelper):
        self.element_tree = element_tree
        self.xsdhelper = xsdhelper

    def _get_transcoder(self, elem):
        t = self.xsdhelper.get_elem_transcoder(elem)
        if t is None:
            raise LookupError(
                f"{self.element_tree.getelementpath(elem)} is not transcodable"
            )
        return t

    def load_elem(self, elem):
        """Decode ``elem`` (an XML element) to a Python object."""
        return self._get_transcoder(elem).parse_elem(elem)

    def load(self, pattern):
        """
        Find and load the first subelement matching ``pattern`` in ``element_tree``.

        Returns the decoded Python object or `None`.

        """
        elem = self.element_tree.find(pattern)
        if elem is None:
            return
        return self.load_elem(elem)

    def set_elem(self, elem, val):
        """Encode ``val`` (a Python object) into the XML element ``elem``."""
        self._get_transcoder(elem).set_elem(elem, val)

    def set(self, pattern, val):
        """
        Find and set the first subelement matching ``pattern`` in ``element_tree`` using
        ``val``.

        """
        elem = self.element_tree.find(pattern)
        if elem is None:
            raise ValueError(f"{pattern=} did not match any elements")
        self.set_elem(elem, val)
