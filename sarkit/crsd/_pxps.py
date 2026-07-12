import lxml.etree
import numpy as np

from . import _io as crsdio
from . import _xml as crsdxml


def _get_pxp_dtype(pxp_node: lxml.etree.Element, num_bytes: int) -> np.dtype:
    """Get PXP dtype.

    Parameters
    ----------
    pxp_node: lxml.etree.Element
        The root element of the PXP data descriptor in the CRSD XML
    num_bytes: int
        Number of bytes in a single PXP set

    Returns
    -------
    numpy.dtype
    """

    bytes_per_word = 8
    names = []
    formats = []
    offsets = []

    def handle_field(field_node):
        node_name = lxml.etree.QName(field_node).localname
        if node_name in ("AddedPVP", "AddedPPP"):
            names.append(field_node.find("./{*}Name").text)
        else:
            names.append(node_name)

        formats.append(
            crsdio.binary_format_string_to_dtype(field_node.find("./{*}Format").text)
        )
        offsets.append(int(field_node.find("./{*}Offset").text) * bytes_per_word)

    for pnode in pxp_node:
        handle_field(pnode)

    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": num_bytes}
    )
    return dtype


def get_ppp_dtype(crsd_xmltree: lxml.etree.ElementTree) -> np.dtype:
    """Get a structured datatype for the Per Pulse Parameter set defined in ``crsd_xmltree``.

    Parameters
    ----------
    crsd_xmltree : lxml.etree.ElementTree
        CRSD XML ElementTree

    Returns
    -------
    numpy.dtype
        Structured datatype for the Per Pulse Parameter set defined in ``crsd_xmltree``
    """
    return _get_pxp_dtype(
        crsd_xmltree.find("./{*}PPP"),
        int(crsd_xmltree.findtext("./{*}Data/{*}Transmit/{*}NumBytesPPP")),
    )


def get_pvp_dtype(crsd_xmltree: lxml.etree.ElementTree) -> np.dtype:
    """Get a structured datatype for the Per Vector Parameter set defined in ``crsd_xmltree``.

    Parameters
    ----------
    crsd_xmltree : lxml.etree.ElementTree
        CRSD XML ElementTree

    Returns
    -------
    numpy.dtype
        Structured datatype for the Per Vector Parameter set defined in ``crsd_xmltree``
    """
    return _get_pxp_dtype(
        crsd_xmltree.find("./{*}PVP"),
        int(crsd_xmltree.findtext("./{*}Data/{*}Receive/{*}NumBytesPVP")),
    )


def _dtype_to_pxp_element(ns: str, tag: str, dtype: np.dtype) -> lxml.etree.Element:
    """Return a CRSD/PxP XML element describing structured datatype ``dtype``.

    Parameters
    ----------
    ns : str
        Namespace of PxP XML element to create
    tag : {"PVP", "PPP"}
        Tag of PxP XML element to create
    dtype : numpy.dtype
        Structured datatype definining the Per-x-Parameter set

    Returns
    -------
    lxml.etree.Element
        CRSD/PxP XML element
    """
    # use CRSDsar because it has both
    root = lxml.etree.Element(lxml.etree.QName(ns, "CRSDsar"))
    ew = crsdxml.ElementWrapper(root)
    ew[tag] = lxml.etree.Element(lxml.etree.QName(ns, tag))
    _, typedef = ew.xsdhelper.get_elem_typeinfo(ew[tag].elem)

    # find nested elements
    nested_parents_by_child = {}
    for child in typedef.children:
        transcoder = ew.xsdhelper.get_transcoder(child.typename, root.tag)
        if not isinstance(transcoder, crsdxml.PxpType):
            child_typedef = ew.xsdhelper.xsdtypes[child.typename]
            parent_localname = lxml.etree.QName(child.tag).localname
            for inner_child in child_typedef.children:
                child_localname = lxml.etree.QName(inner_child.tag).localname
                nested_parents_by_child[child_localname] = parent_localname

    assert dtype.fields is not None  # placate mypy
    for name, field in dtype.fields.items():
        this_info = {
            "dtype": field[0],
            "Offset": field[1] // 8,
            "Size": field[0].itemsize // 8,
        }
        if name in nested_parents_by_child:
            ew[tag][nested_parents_by_child[name]][name] = this_info
        else:
            try:
                ew[tag][name] = this_info
            except KeyError:
                ew[tag].add(f"Added{tag}", this_info | {"Name": name})
    return ew[tag].elem


def dtype_to_pvp_element(ns: str, dtype: np.dtype) -> lxml.etree.Element:
    """Return a CRSD/PVP XML element describing structured datatype ``dtype``.

    Parameters
    ----------
    ns : str
        Namespace of PVP XML element to create
    dtype : numpy.dtype
        Structured datatype definining the Per Vector Parameter set

    Returns
    -------
    lxml.etree.Element
        CRSD/PVP XML element
    """
    return _dtype_to_pxp_element(ns, "PVP", dtype)


def dtype_to_ppp_element(ns: str, dtype: np.dtype) -> lxml.etree.Element:
    """Return a CRSD/PPP XML element describing structured datatype ``dtype``.

    Parameters
    ----------
    ns : str
        Namespace of PPP XML element to create
    dtype : numpy.dtype
        Structured datatype definining the Per Pulse Parameter set

    Returns
    -------
    lxml.etree.Element
        CRSD/PPP XML element
    """
    return _dtype_to_pxp_element(ns, "PPP", dtype)


def _get_defined_pxp_dtype(ns: str, tag: str) -> np.dtype:
    """Get a structured datatype containing all defined PxPs from the requested namespace.

    Parameters
    ----------
    ns : str
        Namespace of PxP XML element used to determine parameter set
    tag : {"PVP", "PPP"}
        Tag of PxP XML element used to determine parameter set

    Returns
    -------
    numpy.dtype
        Structured datatype containing all defined PxPs from the requested namespace
    """
    # use CRSDsar because it has both
    root = lxml.etree.Element(lxml.etree.QName(ns, "CRSDsar"))
    ew = crsdxml.ElementWrapper(root)
    ew[tag] = lxml.etree.Element(lxml.etree.QName(ns, tag))
    _, typedef = ew.xsdhelper.get_elem_typeinfo(ew[tag].elem)

    def get_defined_pxps(tdef) -> list[tuple[str, np.dtype]]:
        pxp_set = []
        for child in tdef.children:
            localname = lxml.etree.QName(child.tag).localname
            transcoder = ew.xsdhelper.get_transcoder(child.typename, root.tag)
            if isinstance(transcoder, crsdxml.DefinedPxpType):
                pxp_set.append((localname, transcoder.dtype))
            else:
                child_typedef = ew.xsdhelper.xsdtypes[child.typename]
                pxp_set.extend(get_defined_pxps(child_typedef))
        return pxp_set

    pxp_set = get_defined_pxps(typedef)
    return np.dtype(pxp_set)


def get_defined_ppp_dtype(ns: str) -> np.dtype:
    """Get a structured datatype containing all defined PPPs from the requested namespace.

    Parameters
    ----------
    ns : str
        Namespace of PPP XML element used to determine parameter set

    Returns
    -------
    numpy.dtype
        Structured datatype containing all defined PPPs from the requested namespace
    """
    return _get_defined_pxp_dtype(ns, "PPP")


def get_defined_pvp_dtype(ns: str) -> np.dtype:
    """Get a structured datatype containing all defined PVPs from the requested namespace.

    Parameters
    ----------
    ns : str
        Namespace of PVP XML element used to determine parameter set

    Returns
    -------
    numpy.dtype
        Structured datatype containing all defined PVPs from the requested namespace
    """
    return _get_defined_pxp_dtype(ns, "PVP")
