"""
Functions to read and write CPHD files.
"""

import copy
import dataclasses
import logging
import os

import lxml.etree
import numpy as np
import numpy.typing as npt

from sarkit import _iohelp

from . import _constants as cphdconst


def _to_binary_format_string_recursive(dtype):
    dtype = np.dtype(dtype)
    if dtype.subdtype is not None:
        dt, shape = dtype.subdtype
        f = _to_binary_format_string_recursive(dt)
        if shape == (3,):
            return "".join(["%s=%s;" % (xyz, f) for xyz in "XYZ"])
        elif shape == (2,):
            return "".join(["DC%s=%s;" % (xy, f) for xy in "XY"])
        else:
            raise ValueError(
                "only dtype arrays of length 2 or 3 supported: %s" % repr(dtype)
            )

    if dtype.kind == "V":
        offset_sorted = sorted(dtype.fields.items(), key=lambda x: x[-1][-1])
        return "".join(
            [
                "%s=%s;" % (name, _to_binary_format_string_recursive(dt))
                for name, (dt, _) in offset_sorted
            ]
        )

    types = {"u": "U", "i": "I", "f": "F", "c": "CF", "S": "S"}
    return "%s%s" % (types[dtype.kind], dtype.itemsize)


def dtype_to_binary_format_string(dtype: np.dtype) -> str:
    """Return the binary format string corresponding to a `numpy.dtype`.

    See the "Allowed Binary Formats" table in the Design & Implementation Description Document

    Parameters
    ----------
    dtype : `numpy.dtype`
        Data-type about which to get binary format string.
        Endianness is ignored.

    Returns
    -------
    str
        Binary format string for the specified `numpy.dtype`

    Examples
    --------

    .. doctest::

        >>> import numpy as np
        >>> import sarkit.cphd as skcphd

        >>> skcphd.dtype_to_binary_format_string(np.uint8)
        'U1'

        >>> skcphd.dtype_to_binary_format_string(np.dtype([('a', np.int16), ('b', 'S30')]))
        'a=I2;b=S30;'
    """
    result = _to_binary_format_string_recursive(dtype)

    if ";;" in result:  # pragma: nocover
        raise ValueError("dtype not supported: %s" % repr(dtype))

    return result


def _single_binary_format_string_to_dtype(form):
    if form.startswith("S"):
        dtype = np.dtype(form)
    else:
        lookup = {
            "U1": np.dtype("u1"),
            "U2": np.dtype("u2"),
            "U4": np.dtype("u4"),
            "U8": np.dtype("u8"),
            "I1": np.dtype("i1"),
            "I2": np.dtype("i2"),
            "I4": np.dtype("i4"),
            "I8": np.dtype("i8"),
            "F4": np.dtype("f4"),
            "F8": np.dtype("f8"),
            "CI2": np.dtype([("real", np.int8), ("imag", np.int8)]),
            "CI4": np.dtype([("real", np.int16), ("imag", np.int16)]),
            "CI8": np.dtype([("real", np.int32), ("imag", np.int32)]),
            "CI16": np.dtype([("real", np.int64), ("imag", np.int64)]),
            "CF8": np.dtype("c8"),
            "CF16": np.dtype("c16"),
        }
        dtype = lookup[form]

    return dtype


def binary_format_string_to_dtype(format_string: str) -> np.dtype:
    """Return the `numpy.dtype` corresponding to a binary format string.

    See the "Allowed Binary Formats" table in the Design & Implementation Description Document

    Parameters
    ----------
    format_string : str
        Binary format string about which to get the dtype.

    Returns
    -------
    `numpy.dtype`
        `numpy.dtype` corresponding to the binary format string (in native byte order)

    Examples
    --------

    .. doctest::

        >>> import numpy as np
        >>> import sarkit.cphd as skcphd

        >>> skcphd.binary_format_string_to_dtype('U1')
        dtype('uint8')

        >>> skcphd.binary_format_string_to_dtype('a=I2;b=S30;')
        dtype([('a', '<i2'), ('b', 'S30')])
    """
    components = format_string.split(";")

    if "=" in components[0]:
        comptypes = []
        for comp in components[:-1]:
            kvp = comp.split("=")
            comptypes.append((kvp[0], _single_binary_format_string_to_dtype(kvp[1])))

        # special handling of XYZ and EB types
        keys, types = list(zip(*comptypes))
        if keys == ("X", "Y", "Z") and len(set(types)) == 1:
            dtype = np.dtype("3" + comptypes[0][1].name)
        elif keys == ("DCX", "DCY") and len(set(types)) == 1:
            dtype = np.dtype("2" + comptypes[0][1].name)
        else:
            dtype = np.dtype(comptypes)
    else:
        dtype = _single_binary_format_string_to_dtype(components[0])

    return dtype


def mask_support_array(
    array: npt.NDArray, nodata_hex: str | None = None
) -> np.ma.MaskedArray:
    """Apply a NODATA hex string to a support array to mask the array.

    Parameters
    ----------
    array : np.ndarray
        Support array to compare to NODATA and create a masked array from
    nodata_hex : str, optional
        If None, all array elements are valid. Otherwise, use the hex string to
        compare to the values in ``array`` to make the mask

    Returns
    -------
    masked_array : :py:class:`~numpy.ma.MaskedArray`
        ``array`` with NODATA elements masked
    """
    if nodata_hex is None:
        return np.ma.array(array)
    nodata_v = np.void(bytes.fromhex(nodata_hex))
    return np.ma.array(
        array,
        mask=array.view(nodata_v.dtype) == nodata_v,
        fill_value=nodata_v.view(array.dtype),
    )


def _describe_signal(
    cphd_xmltree: lxml.etree.ElementTree,
    channel_identifier: str,
    *,
    standard_format=False,
) -> tuple[tuple[int, ...], np.dtype]:
    """Return the shape and dtype of the signal array identified by ``channel_identifier``"""
    channel_info = cphd_xmltree.find(
        f"{{*}}Data/{{*}}Channel[{{*}}Identifier='{channel_identifier}']"
    )
    is_compressed = (
        compressed_signal_size := channel_info.findtext("{*}CompressedSignalSize")
    ) is not None
    if is_compressed and not standard_format:
        dtype = np.dtype(np.uint8)
        shape: tuple[int, ...] = (int(compressed_signal_size),)
    else:
        dtype = binary_format_string_to_dtype(
            cphd_xmltree.find("./{*}Data/{*}SignalArrayFormat").text
        )
        shape = (
            int(channel_info.find("{*}NumVectors").text),
            int(channel_info.find("{*}NumSamples").text),
        )
    return shape, dtype


@dataclasses.dataclass(kw_only=True)
class FileHeaderPart:
    """CPHD header fields which are set per program specific Product Design Document

    Attributes
    ----------
    additional_kvps : dict of {str : str}
        Additional key-value pairs
    """

    additional_kvps: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class Metadata:
    """Settable CPHD metadata

    Attributes
    ----------
    file_header_part : FileHeaderPart
        CPHD File Header fields which can be set
    xmltree : lxml.etree.ElementTree
        CPHD XML
    """

    file_header_part: FileHeaderPart = dataclasses.field(default_factory=FileHeaderPart)
    xmltree: lxml.etree.ElementTree


def read_file_header(file):
    """Read a file header.

    The file object's position is assumed to be at the start of the file.

    Parameters
    ----------
    file : `file object`
        The open file object, which will be progressively read.

    Returns
    -------
    file_type_header : str
        File type header from the first line of the file
    kvp_list : dict of {str : str}
        Key-Value pair list of header fields
    """
    file_type_header = file.readline().decode()

    kvp_list = {}
    while (line := file.readline()) != cphdconst.SECTION_TERMINATOR:
        field, value = line.decode().strip("\n").split(" := ")
        kvp_list[field] = value
    return file_type_header, kvp_list


def get_pvp_dtype(cphd_xmltree):
    """Get PVP dtype.

    Parameters
    ----------
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML ElementTree

    Returns
    -------
    numpy.dtype
    """

    pvp_node = cphd_xmltree.find("./{*}PVP")

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
            binary_format_string_to_dtype(field_node.find("./{*}Format").text)
        )
        offsets.append(int(field_node.find("./{*}Offset").text) * bytes_per_word)

    for pnode in pvp_node:
        if lxml.etree.QName(pnode).localname in ("TxAntenna", "RcvAntenna"):
            for subnode in pnode:
                handle_field(subnode)
        else:
            handle_field(pnode)

    dtype = np.dtype(({"names": names, "formats": formats, "offsets": offsets}))
    return dtype


class Reader:
    """Read a CPHD file

    A Reader object can be used as a context manager in a ``with`` statement.
    Attributes, but not methods, can be safely accessed outside of the context manager's context.

    Parameters
    ----------
    file : `file object`
        CPHD file to read

    Attributes
    ----------
    metadata : Metadata
       CPHD metadata

    See Also
    --------
    Writer

    Examples
    --------

    .. testsetup:: cphd_io

        import sarkit.cphd as skcphd
        import lxml.etree
        meta = skcphd.Metadata(xmltree=lxml.etree.parse("data/example-cphd-1.0.1.xml"))

        file = pathlib.Path(tmpdir.name) / "foo"
        with file.open("wb") as f, skcphd.Writer(f, meta) as w:
            f.seek(
                w._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"]
                + w._file_header_kvp["SIGNAL_BLOCK_SIZE"]
                - 1
            )
            f.write(b"0")

    .. doctest:: cphd_io

        >>> import sarkit.cphd as skcphd
        >>> with file.open("rb") as f, skcphd.Reader(f) as r:
        ...     ch_id = r.metadata.xmltree.findtext("{*}Data/{*}Channel/{*}Identifier")
        ...     sig, pvp = r.read_channel(ch_id)
        ...     sa_id = r.metadata.xmltree.findtext("{*}Data/{*}SupportArray/{*}Identifier")
        ...     sa = r.read_support_array(sa_id)
    """

    def __init__(self, file):
        self._file_object = file

        # skip the version line and read header
        _, self._kvp_list = read_file_header(self._file_object)

        extra_header_keys = set(self._kvp_list.keys()) - cphdconst.DEFINED_HEADER_KEYS
        additional_kvps = {key: self._kvp_list[key] for key in extra_header_keys}

        self._file_object.seek(self._xml_block_byte_offset)
        xml_bytes = self._file_object.read(int(self._kvp_list["XML_BLOCK_SIZE"]))

        self.metadata = Metadata(
            xmltree=lxml.etree.fromstring(xml_bytes).getroottree(),
            file_header_part=FileHeaderPart(additional_kvps=additional_kvps),
        )

    @property
    def _xml_block_byte_offset(self) -> int:
        """Offset to the XML block"""
        return int(self._kvp_list["XML_BLOCK_BYTE_OFFSET"])

    @property
    def _xml_block_size(self) -> int:
        """Size of the XML block"""
        return int(self._kvp_list["XML_BLOCK_SIZE"])

    @property
    def _pvp_block_byte_offset(self) -> int:
        """Offset to the PVP block"""
        return int(self._kvp_list["PVP_BLOCK_BYTE_OFFSET"])

    @property
    def _pvp_block_size(self) -> int:
        """Size of the PVP block"""
        return int(self._kvp_list["PVP_BLOCK_SIZE"])

    @property
    def _signal_block_byte_offset(self) -> int:
        """Offset to the Signal block"""
        return int(self._kvp_list["SIGNAL_BLOCK_BYTE_OFFSET"])

    @property
    def _signal_block_size(self) -> int:
        """Size of the Signal block"""
        return int(self._kvp_list["SIGNAL_BLOCK_SIZE"])

    @property
    def _support_block_byte_offset(self) -> int | None:
        """Offset to the Support block"""
        if "SUPPORT_BLOCK_BYTE_OFFSET" in self._kvp_list:
            return int(self._kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"])
        return None

    @property
    def _support_block_size(self) -> int | None:
        """Size of the Support block"""
        if "SUPPORT_BLOCK_SIZE" in self._kvp_list:
            return int(self._kvp_list["SUPPORT_BLOCK_SIZE"])
        return None

    def read_signal(
        self,
        channel_identifier: str,
        *,
        start_vector: int | None = None,
        stop_vector: int | None = None,
    ) -> npt.NDArray:
        """Read signal data from a CPHD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier
        start_vector : int or None, optional
            Lowest vector index to retrieve (inclusive). If None, defaults to first vector.
        stop_vector : int or None, optional
            Highest vector index to retrieve (exclusive). If None, defaults to one after last vector.

        Returns
        -------
        ndarray
            Signal array identified by ``channel_identifier``

            When standard, shape=(``stop_vector`` - ``start_vector``, NumSamples), dtype determined by SignalArrayFormat.

            When compressed, shape=(CompressedSignalSize,), dtype= `numpy.uint8`

        Notes
        -----
        ``start_vector`` and ``stop_vector`` are not supported when signal data is compressed
        """
        signal_shape, dtype = _describe_signal(
            self.metadata.xmltree, channel_identifier
        )
        dtype = dtype.newbyteorder(">")
        out_shape: tuple[int, ...]
        if len(signal_shape) == 1:  # compressed
            if start_vector is not None or stop_vector is not None:
                raise ValueError(
                    "start_vector and stop_vector not supported for compressed signals"
                )
            out_shape = signal_shape
            slice_offset = 0
        else:
            # Convert None and negative values to absolute indices
            start_vector, stop_vector, _ = slice(start_vector, stop_vector).indices(
                signal_shape[0]
            )
            out_shape = (max(stop_vector - start_vector, 0), signal_shape[1])
            slice_offset = dtype.itemsize * start_vector * signal_shape[1]

        signal_offset = int(
            self.metadata.xmltree.findtext(
                f"{{*}}Data/{{*}}Channel[{{*}}Identifier='{channel_identifier}']/{{*}}SignalArrayByteOffset"
            )
        )
        self._file_object.seek(
            slice_offset + signal_offset + self._signal_block_byte_offset
        )
        out = _iohelp.fromfile(
            self._file_object, dtype, count=np.prod(out_shape)
        ).reshape(out_shape)
        return out

    def read_pvps(
        self,
        channel_identifier: str,
        *,
        start_vector: int | None = None,
        stop_vector: int | None = None,
    ) -> npt.NDArray:
        """Read pvp data from a CPHD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier
        start_vector : int or None, optional
            Lowest vector index to retrieve (inclusive). If None, defaults to first vector.
        stop_vector : int or None, optional
            Highest vector index to retrieve (exclusive). If None, defaults to one after last vector.

        Returns
        -------
        ndarray
            CPHD PVP array

        """
        channel_info = self.metadata.xmltree.find(
            f"{{*}}Data/{{*}}Channel[{{*}}Identifier='{channel_identifier}']"
        )
        num_vect = int(channel_info.find("./{*}NumVectors").text)

        # Convert None and negative values to absolute indices
        start_vector, stop_vector, _ = slice(start_vector, stop_vector).indices(
            num_vect
        )
        count = max(stop_vector - start_vector, 0)

        pvp_dtype = get_pvp_dtype(self.metadata.xmltree).newbyteorder("B")
        slice_offset = pvp_dtype.itemsize * start_vector
        pvp_offset = int(channel_info.find("./{*}PVPArrayByteOffset").text)
        self._file_object.seek(slice_offset + pvp_offset + self._pvp_block_byte_offset)
        out = _iohelp.fromfile(self._file_object, dtype=pvp_dtype, count=count)
        return out

    def read_channel(
        self,
        channel_identifier: str,
        *,
        start_vector: int | None = None,
        stop_vector: int | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Read signal and pvp data from a CPHD file channel

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier
        start_vector : int or None, optional
            Lowest vector index to retrieve (inclusive). If None, defaults to first vector.
        stop_vector : int or None, optional
            Highest vector index to retrieve (exclusive). If None, defaults to one after last vector.

        Returns
        -------
        signal_array : ndarray
            Signal array for channel = channel_identifier
        pvp_array : ndarray
            PVP array for channel = channel_identifier

        """
        signal = self.read_signal(
            channel_identifier, start_vector=start_vector, stop_vector=stop_vector
        )
        pvp = self.read_pvps(
            channel_identifier, start_vector=start_vector, stop_vector=stop_vector
        )

        return signal, pvp

    def _read_support_array(self, sa_identifier):
        elem_format = self.metadata.xmltree.find(
            f"{{*}}SupportArray/*[{{*}}Identifier='{sa_identifier}']/{{*}}ElementFormat"
        )
        dtype = binary_format_string_to_dtype(elem_format.text).newbyteorder("B")

        sa_info = self.metadata.xmltree.find(
            f"{{*}}Data/{{*}}SupportArray[{{*}}Identifier='{sa_identifier}']"
        )
        num_rows = int(sa_info.find("./{*}NumRows").text)
        num_cols = int(sa_info.find("./{*}NumCols").text)
        shape = (num_rows, num_cols)

        sa_offset = int(sa_info.find("./{*}ArrayByteOffset").text)
        self._file_object.seek(sa_offset + self._support_block_byte_offset)
        assert dtype.itemsize == int(sa_info.find("./{*}BytesPerElement").text)
        return _iohelp.fromfile(self._file_object, dtype, np.prod(shape)).reshape(shape)

    def read_support_array(self, sa_identifier, masked=True):
        """Read SupportArray"""
        array = self._read_support_array(sa_identifier)
        if not masked:
            return array
        nodata = self.metadata.xmltree.findtext(
            f"{{*}}SupportArray/*[{{*}}Identifier='{sa_identifier}']/{{*}}NODATA"
        )
        return mask_support_array(array, nodata)

    def done(self):
        "Indicates to the reader that the user is done with it"
        self._file_object = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.done()


class Writer:
    """Write a CPHD file

    A Writer object can be used as a context manager in a ``with`` statement.

    Parameters
    ----------
    file : `file object`
        CPHD file to write
    metadata : Metadata
        CPHD metadata to write (copied on construction)

    See Also
    --------
    Reader

    Examples
    --------
    Generate some metadata and data

    .. doctest:: cphd_io

        >>> import lxml.etree

        >>> xmltree = lxml.etree.parse("data/example-cphd-1.0.1.xml")
        >>> first_channel = xmltree.find("{*}Data/{*}Channel")
        >>> ch_id = first_channel.findtext("{*}Identifier")
        >>> num_v = int(first_channel.findtext("{*}NumVectors"))
        >>> num_s = int(first_channel.findtext("{*}NumSamples"))
        >>> sig_format = xmltree.findtext("{*}Data/{*}SignalArrayFormat")

        >>> import sarkit.cphd as skcphd

        >>> meta = skcphd.Metadata(
        ...     xmltree=xmltree,
        ...     file_header_part=skcphd.FileHeaderPart(additional_kvps={"K": "V"}),
        ... )

        >>> import numpy as np

        >>> sig = np.zeros((num_v, num_s), dtype=skcphd.binary_format_string_to_dtype(sig_format))
        >>> pvps = np.zeros(num_v, dtype=skcphd.get_pvp_dtype(xmltree))

    Write a channel's signal and PVP arrays to a file.

    .. doctest:: cphd_io

        >>> with (tmppath / "written.cphd").open("wb") as f, skcphd.Writer(f, meta) as w:
        ...     w.write_signal(ch_id, sig)
        ...     w.write_pvp(ch_id, pvps)
    """

    def __init__(self, file, metadata: Metadata):
        align_to = 64
        self._file_object = file

        self._metadata = copy.deepcopy(metadata)
        cphd_xmltree = self._metadata.xmltree

        xml_block_body = lxml.etree.tostring(cphd_xmltree, encoding="utf-8")

        pvp_itemsize = int(cphd_xmltree.find("./{*}Data/{*}NumBytesPVP").text)
        self._channel_size_offsets = {}
        for chan_node in cphd_xmltree.findall("./{*}Data/{*}Channel"):
            channel_identifier = chan_node.find("./{*}Identifier").text
            channel_signal_offset = int(
                chan_node.find("./{*}SignalArrayByteOffset").text
            )
            shape, dtype = _describe_signal(cphd_xmltree, channel_identifier)
            channel_signal_size = int(np.prod(shape)) * dtype.itemsize
            channel_pvp_offset = int(chan_node.find("./{*}PVPArrayByteOffset").text)
            channel_pvp_size = (
                int(chan_node.find("./{*}NumVectors").text) * pvp_itemsize
            )

            self._channel_size_offsets[channel_identifier] = {
                "signal_offset": channel_signal_offset,
                "signal_size": channel_signal_size,
                "pvp_offset": channel_pvp_offset,
                "pvp_size": channel_pvp_size,
            }

        signal_block_size = max(
            chan["signal_size"] + chan["signal_offset"]
            for chan in self._channel_size_offsets.values()
        )
        pvp_block_size = max(
            chan["pvp_size"] + chan["pvp_offset"]
            for chan in self._channel_size_offsets.values()
        )

        self._sa_size_offsets = {}
        for sa_node in cphd_xmltree.findall("./{*}Data/{*}SupportArray"):
            sa_identifier = sa_node.find("./{*}Identifier").text
            sa_offset = int(sa_node.find("./{*}ArrayByteOffset").text)
            sa_size = (
                int(sa_node.find("./{*}NumRows").text)
                * int(sa_node.find("./{*}NumCols").text)
                * int(sa_node.find("./{*}BytesPerElement").text)
            )

            self._sa_size_offsets[sa_identifier] = {
                "offset": sa_offset,
                "size": sa_size,
            }

        def _align(val):
            return int(np.ceil(float(val) / align_to) * align_to)

        self._file_header_kvp = {
            "XML_BLOCK_SIZE": len(xml_block_body),
            "XML_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
            "PVP_BLOCK_SIZE": pvp_block_size,
            "PVP_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
            "SIGNAL_BLOCK_SIZE": signal_block_size,
            "SIGNAL_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
            "CLASSIFICATION": cphd_xmltree.findtext(
                "{*}CollectionID/{*}Classification"
            ),
            "RELEASE_INFO": cphd_xmltree.findtext("{*}CollectionID/{*}ReleaseInfo"),
        }
        if self._sa_size_offsets:
            self._file_header_kvp["SUPPORT_BLOCK_SIZE"] = max(
                sa["size"] + sa["offset"] for sa in self._sa_size_offsets.values()
            )
            self._file_header_kvp["SUPPORT_BLOCK_BYTE_OFFSET"] = (
                np.iinfo(np.uint64).max,
            )  # placeholder

        self._file_header_kvp.update(self._metadata.file_header_part.additional_kvps)

        def _serialize_header():
            version = cphdconst.VERSION_INFO[
                lxml.etree.QName(cphd_xmltree.getroot()).namespace
            ]["version"]
            header_str = f"CPHD/{version}\n"
            header_str += "".join(
                (f"{key} := {value}\n" for key, value in self._file_header_kvp.items())
            )
            return header_str.encode() + cphdconst.SECTION_TERMINATOR

        next_offset = _align(len(_serialize_header()))
        self._file_header_kvp["XML_BLOCK_BYTE_OFFSET"] = next_offset
        next_offset = _align(
            next_offset
            + self._file_header_kvp["XML_BLOCK_SIZE"]
            + len(cphdconst.SECTION_TERMINATOR)
        )

        if self._sa_size_offsets:
            self._file_header_kvp["SUPPORT_BLOCK_BYTE_OFFSET"] = next_offset
            next_offset = _align(
                next_offset + self._file_header_kvp["SUPPORT_BLOCK_SIZE"]
            )

        self._file_header_kvp["PVP_BLOCK_BYTE_OFFSET"] = next_offset
        next_offset = _align(next_offset + self._file_header_kvp["PVP_BLOCK_SIZE"])

        self._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"] = next_offset
        next_offset = _align(next_offset + self._file_header_kvp["SIGNAL_BLOCK_SIZE"])

        self._file_object.seek(0)
        self._file_object.write(_serialize_header())
        self._file_object.seek(self._file_header_kvp["XML_BLOCK_BYTE_OFFSET"])
        self._file_object.write(xml_block_body + cphdconst.SECTION_TERMINATOR)

        self._signal_arrays_written: set[str] = set()
        self._pvp_arrays_written: set[str] = set()
        self._support_arrays_written: set[str] = set()

    def write_signal(self, channel_identifier: str, signal_array: npt.NDArray):
        """Write signal data to a CPHD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier
        signal_array : ndarray
            Signal data to write.

            When standard, shape=(NumVectors, NumSamples), dtype determined by SignalArrayFormat.

            When compressed, shape=(CompressedSignalSize,), dtype= `numpy.uint8`
        """
        # TODO Add support for partial CPHD writing
        shape, dtype = _describe_signal(self._metadata.xmltree, channel_identifier)
        if dtype != signal_array.dtype.newbyteorder("="):
            raise ValueError(f"{signal_array.dtype=} is not compatible with {dtype=}")
        if shape != signal_array.shape:
            raise ValueError(f"{signal_array.shape=} does not match {shape=}")

        buff_to_write = signal_array.astype(dtype.newbyteorder(">"), copy=False).data
        expected_nbytes = self._channel_size_offsets[channel_identifier]["signal_size"]
        if buff_to_write.nbytes != expected_nbytes:
            raise ValueError(
                f"{buff_to_write.nbytes=} does not match {expected_nbytes=}"
            )

        self._file_object.seek(
            self._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"]
            + self._channel_size_offsets[channel_identifier]["signal_offset"]
        )
        self._file_object.write(buff_to_write)
        self._signal_arrays_written.add(channel_identifier)

    def write_pvp(self, channel_identifier: str, pvp_array: npt.NDArray):
        """Write pvp data to a CPHD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier
        pvp_array : ndarray
            Array of PVPs

        """
        assert (
            pvp_array.nbytes
            == self._channel_size_offsets[channel_identifier]["pvp_size"]
        )

        self._pvp_arrays_written.add(channel_identifier)
        self._file_object.seek(self._file_header_kvp["PVP_BLOCK_BYTE_OFFSET"])
        self._file_object.seek(
            self._channel_size_offsets[channel_identifier]["pvp_offset"], os.SEEK_CUR
        )
        output_dtype = pvp_array.dtype.newbyteorder(">")
        pvp_array.astype(output_dtype, copy=False).tofile(self._file_object)

    def write_support_array(
        self, support_array_identifier: str, support_array: npt.NDArray
    ):
        """Write support array data to a CPHD file

        Parameters
        ----------
        support_array_identifier : str
            Unique support array identifier
        support_array : ndarray
            Array of support data
        """
        data_sa_elem = self._metadata.xmltree.find(
            f"{{*}}Data/{{*}}SupportArray[{{*}}Identifier='{support_array_identifier}']"
        )
        expected_shape = (
            int(data_sa_elem.findtext("{*}NumRows")),
            int(data_sa_elem.findtext("{*}NumCols")),
        )
        sa_elem = self._metadata.xmltree.find(
            f"{{*}}SupportArray/*[{{*}}Identifier='{support_array_identifier}']"
        )
        element_format = sa_elem.findtext("{*}ElementFormat")
        expected_dtype = binary_format_string_to_dtype(element_format)
        expected_nodata = sa_elem.findtext("{*}NODATA")

        if expected_dtype != support_array.dtype.newbyteorder("="):
            raise ValueError(
                f"{support_array.dtype=} is not compatible with {expected_dtype=}"
            )
        if expected_shape != support_array.shape:
            raise ValueError(f"{support_array.shape=} does not match {expected_shape=}")
        if isinstance(support_array, np.ma.MaskedArray):
            actual_nodata = (
                support_array.fill_value.astype(expected_dtype.newbyteorder(">"))
                .tobytes()
                .hex()
            )

            def _is_masked(array):
                # structured arrays don't play nice with np.ma
                if array.dtype.names is None:
                    return np.ma.is_masked(array)
                return any(_is_masked(array[n]) for n in array.dtype.names)

            if _is_masked(support_array) and expected_nodata != actual_nodata:
                raise ValueError(f"{actual_nodata=} does not match {expected_nodata=}")

        self._file_object.seek(self._file_header_kvp["SUPPORT_BLOCK_BYTE_OFFSET"])
        self._file_object.seek(
            self._sa_size_offsets[support_array_identifier]["offset"], os.SEEK_CUR
        )
        output_dtype = support_array.dtype.newbyteorder(">")
        self._file_object.write(support_array.astype(output_dtype, copy=False).data)
        self._support_arrays_written.add(support_array_identifier)

    def done(self):
        """Warn about unwritten arrays declared in the XML"""
        channel_names = set(
            node.text
            for node in self._metadata.xmltree.findall(
                "./{*}Data/{*}Channel/{*}Identifier"
            )
        )
        missing_signal_channels = channel_names - self._signal_arrays_written
        if missing_signal_channels:
            logging.warning(
                f"Not all Signal Arrays written.  Missing {missing_signal_channels}"
            )

        missing_pvp_channels = channel_names - self._pvp_arrays_written
        if missing_pvp_channels:
            logging.warning(
                f"Not all PVP Arrays written.  Missing {missing_pvp_channels}"
            )

        sa_names = set(
            node.text
            for node in self._metadata.xmltree.findall(
                "./{*}Data/{*}SupportArray/{*}Identifier"
            )
        )
        missing_sa = sa_names - self._support_arrays_written
        if missing_sa:
            logging.warning(f"Not all Support Arrays written.  Missing {missing_sa}")

        self._file_object = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.done()
