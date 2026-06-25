import lxml.etree
import numpy as np

from . import _io as cphdio
from . import _xml as cphdxml


def get_pvp_dtype(cphd_xmltree: lxml.etree.ElementTree) -> np.dtype:
    """Get a structured datatype for the Per Vector Parameter set defined in ``cphd_xmltree``.

    Parameters
    ----------
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML ElementTree

    Returns
    -------
    numpy.dtype
        Structured datatype for the Per Vector Parameter set defined in ``cphd_xmltree``
    """

    pvp_node = cphd_xmltree.find("./{*}PVP")
    num_bytes_pvp = int(cphd_xmltree.findtext("./{*}Data/{*}NumBytesPVP"))

    bytes_per_word = 8
    names = []
    formats = []
    offsets = []

    def handle_field(field_node):
        node_name = lxml.etree.QName(field_node).localname
        if node_name == "AddedPVP":
            names.append(field_node.find("./{*}Name").text)
        else:
            names.append(node_name)

        formats.append(
            cphdio.binary_format_string_to_dtype(field_node.find("./{*}Format").text)
        )
        offsets.append(int(field_node.find("./{*}Offset").text) * bytes_per_word)

    for pnode in pvp_node:
        if lxml.etree.QName(pnode).localname in ("TxAntenna", "RcvAntenna"):
            for subnode in pnode:
                handle_field(subnode)
        else:
            handle_field(pnode)

    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": num_bytes_pvp,
        }
    )
    return dtype


def dtype_to_pvp_element(ns: str, dtype: np.dtype) -> lxml.etree.Element:
    """Return a CPHD/PVP XML element describing structured datatype ``dtype``.

    Parameters
    ----------
    ns : str
        Namespace of PVP XML element to create
    dtype : numpy.dtype
        Structured datatype definining the Per Vector Parameter set

    Returns
    -------
    lxml.etree.Element
        CPHD/PVP XML element
    """
    root = lxml.etree.Element(lxml.etree.QName(ns, "CPHD"))
    ew = cphdxml.ElementWrapper(root)
    ew["PVP"] = lxml.etree.Element(lxml.etree.QName(ns, "PVP"))
    _, typedef = ew.xsdhelper.get_elem_typeinfo(ew["PVP"].elem)

    # find nested elements
    nested_parents_by_child = {}
    for child in typedef.children:
        transcoder = ew.xsdhelper.get_transcoder(child.typename, root.tag)
        if not isinstance(transcoder, cphdxml.PvpType):
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
            ew["PVP"][nested_parents_by_child[name]][name] = this_info
        else:
            try:
                ew["PVP"][name] = this_info
            except KeyError:
                ew["PVP"].add("AddedPVP", this_info | {"Name": name})
    return ew["PVP"].elem


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
    root = lxml.etree.Element(lxml.etree.QName(ns, "CPHD"))
    ew = cphdxml.ElementWrapper(root)
    ew["PVP"] = lxml.etree.Element(lxml.etree.QName(ns, "PVP"))
    _, typedef = ew.xsdhelper.get_elem_typeinfo(ew["PVP"].elem)

    def get_defined_pvps(tdef) -> list[tuple[str, np.dtype]]:
        pvp_set = []
        for child in tdef.children:
            localname = lxml.etree.QName(child.tag).localname
            transcoder = ew.xsdhelper.get_transcoder(child.typename, root.tag)
            if isinstance(transcoder, cphdxml.DefinedPvpType):
                pvp_set.append((localname, transcoder.dtype))
            else:
                child_typedef = ew.xsdhelper.xsdtypes[child.typename]
                pvp_set.extend(get_defined_pvps(child_typedef))
        return pvp_set

    pvp_set = get_defined_pvps(typedef)
    return np.dtype(pvp_set)
