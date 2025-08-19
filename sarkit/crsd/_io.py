"""
Functions to read and write CRSD files.
"""

import copy
import dataclasses
import logging
import os

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.cphd as skcphd
from sarkit import _iohelp

from . import _constants as crsdconst


# Happens to match CPHD
def dtype_to_binary_format_string(dtype: np.dtype) -> str:
    return skcphd.dtype_to_binary_format_string(dtype)


# Happens to match CPHD
def binary_format_string_to_dtype(format_string: str) -> np.dtype:
    return skcphd.binary_format_string_to_dtype(format_string)


dtype_to_binary_format_string.__doc__ = getattr(
    skcphd.dtype_to_binary_format_string, "__doc__", ""
).replace("cphd", "crsd")
binary_format_string_to_dtype.__doc__ = getattr(
    skcphd.binary_format_string_to_dtype, "__doc__", ""
).replace("cphd", "crsd")

mask_support_array = skcphd.mask_support_array


def _describe_signal(
    xmltree: lxml.etree.ElementTree,
    channel_identifier: str,
) -> tuple[tuple[int, int], np.dtype]:
    """Return the shape and dtype of the signal array in standard format identified by ``channel_identifier``."""
    data_rcv = xmltree.find("{*}Data/{*}Receive")
    channel_info = data_rcv.find(f"{{*}}Channel[{{*}}ChId='{channel_identifier}']")
    dtype = binary_format_string_to_dtype(data_rcv.findtext("{*}SignalArrayFormat"))
    shape = (
        int(channel_info.find("{*}NumVectors").text),
        int(channel_info.find("{*}NumSamples").text),
    )
    return shape, dtype


def describe_support_array(
    xmltree: lxml.etree.ElementTree,
    sa_id: str,
) -> tuple[tuple[int, int], np.dtype, lxml.etree.Element]:
    """Return metadata describing the support array identified by ``sa_id``"""
    data_sa_elem = xmltree.find(
        f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{sa_id}']"
    )
    expected_shape = (
        int(data_sa_elem.findtext("{*}NumRows")),
        int(data_sa_elem.findtext("{*}NumCols")),
    )
    sa_elem = xmltree.find(f"{{*}}SupportArray/*[{{*}}Identifier='{sa_id}']")
    element_format = sa_elem.findtext("{*}ElementFormat")
    expected_dtype = binary_format_string_to_dtype(element_format)
    return expected_shape, expected_dtype, sa_elem


@dataclasses.dataclass(kw_only=True)
class FileHeaderPart:
    """CRSD header fields which are set per program specific Product Design Document

    Attributes
    ----------
    additional_kvps : dict of {str : str}
        Additional key-value pairs
    """

    additional_kvps: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class Metadata:
    """Settable CRSD metadata

    Attributes
    ----------
    file_header_part : FileHeaderPart
        CRSD File Header fields which can be set
    xmltree : lxml.etree.ElementTree
        CRSD XML
    """

    file_header_part: FileHeaderPart = dataclasses.field(default_factory=FileHeaderPart)
    xmltree: lxml.etree.ElementTree


read_file_header = skcphd.read_file_header


def _get_pxp_dtype(pxp_node):
    """Get PXP dtype.

    Parameters
    ----------
    pxp_elem: lxml.etree.Element
        The root element of the PXP data descriptor in the CRSD XML

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
            binary_format_string_to_dtype(field_node.find("./{*}Format").text)
        )
        offsets.append(int(field_node.find("./{*}Offset").text) * bytes_per_word)

    for pnode in pxp_node:
        handle_field(pnode)

    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets})
    return dtype


def get_ppp_dtype(crsd_xmltree):
    """Get PPP dtype.

    Parameters
    ----------
    crsd_xmltree : lxml.etree.ElementTree
        CRSD XML ElementTree

    Returns
    -------
    numpy.dtype
    """
    return _get_pxp_dtype(crsd_xmltree.find("./{*}PPP"))


def get_pvp_dtype(crsd_xmltree):
    """Get PVP dtype.

    Parameters
    ----------
    crsd_xmltree : lxml.etree.ElementTree
        CRSD XML ElementTree

    Returns
    -------
    numpy.dtype
    """
    return _get_pxp_dtype(crsd_xmltree.find("./{*}PVP"))


class Reader:
    """Read a CRSD file

    A Reader object can be used as a context manager in a ``with`` statement.
    Attributes, but not methods, can be safely accessed outside of the context manager's context.

    Parameters
    ----------
    file : `file object`
        CRSD file to read

    Attributes
    ----------
    metadata : Metadata
       CRSD metadata

    See Also
    --------
    Writer

    Examples
    --------

    .. testsetup:: crsd_io

        import sarkit.crsd as skcrsd
        import lxml.etree
        meta = skcrsd.Metadata(
            xmltree=lxml.etree.parse("data/example-crsd-1.0.xml")
        )

        file = pathlib.Path(tmpdir.name) / "foo"
        with file.open("wb") as f, skcrsd.Writer(f, meta) as w:
            f.seek(
                w._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"]
                + w._file_header_kvp["SIGNAL_BLOCK_SIZE"]
                - 1
            )
            f.write(b"0")

    .. doctest:: crsd_io

        >>> import sarkit.crsd as skcrsd
        >>> with file.open("rb") as f, skcrsd.Reader(f) as r:
        ...     sa_id = r.metadata.xmltree.findtext("{*}Data/{*}Support//{*}SAId")
        ...     sa = r.read_support_array(sa_id)
        ...     tx_id = r.metadata.xmltree.findtext("{*}Data/{*}Transmit//{*}TxId")
        ...     txseq = r.read_ppps(tx_id)
        ...     ch_id = r.metadata.xmltree.findtext("{*}Data/{*}Receive/{*}Channel/{*}ChId")
        ...     sig, pvp = r.read_channel(ch_id)
    """

    def __init__(self, file):
        self._file_object = file

        # skip the version line and read header
        _, self._kvp_list = read_file_header(self._file_object)

        extra_header_keys = set(self._kvp_list.keys()) - crsdconst.DEFINED_HEADER_KEYS
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
    def _pvp_block_byte_offset(self) -> int | None:
        """Offset to the PVP block"""
        if (n := self._kvp_list.get("PVP_BLOCK_BYTE_OFFSET")) is not None:
            return int(n)
        return None

    @property
    def _pvp_block_size(self) -> int | None:
        """Size of the PVP block"""
        if (n := self._kvp_list.get("PVP_BLOCK_SIZE")) is not None:
            return int(n)
        return None

    @property
    def _ppp_block_byte_offset(self) -> int | None:
        """Offset to the PPP block"""
        if (n := self._kvp_list.get("PPP_BLOCK_BYTE_OFFSET")) is not None:
            return int(n)
        return None

    @property
    def _ppp_block_size(self) -> int | None:
        """Size of the PPP block"""
        if (n := self._kvp_list.get("PPP_BLOCK_SIZE")) is not None:
            return int(n)
        return None

    @property
    def _signal_block_byte_offset(self) -> int | None:
        """Offset to the Signal block"""
        if (n := self._kvp_list.get("SIGNAL_BLOCK_BYTE_OFFSET")) is not None:
            return int(n)
        return None

    @property
    def _signal_block_size(self) -> int | None:
        """Size of the Signal block"""
        if (n := self._kvp_list.get("SIGNAL_BLOCK_SIZE")) is not None:
            return int(n)
        return None

    @property
    def _support_block_byte_offset(self) -> int:
        """Offset to the Support block"""
        return int(self._kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"])

    @property
    def _support_block_size(self) -> int:
        """Size of the Support block"""
        return int(self._kvp_list["SUPPORT_BLOCK_SIZE"])

    def read_signal(self, channel_identifier: str) -> npt.NDArray:
        """Read signal data from a CRSD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier

        Returns
        -------
        ndarray
            Signal array identified by ``channel_identifier``;
            shape=(NumVectors, NumSamples), dtype determined by SignalArrayFormat.

        Raises
        ------
        RuntimeError
            If the signal block is compressed.

        See Also
        --------
        read_signal_compressed
        """
        data_rcv = self.metadata.xmltree.find("{*}Data/{*}Receive")
        if data_rcv.find("{*}SignalCompression") is not None:
            raise RuntimeError(
                "Signal block is compressed; use read_signal_compressed instead."
            )
        signal_offset = int(
            data_rcv.findtext(
                f"{{*}}Channel[{{*}}ChId='{channel_identifier}']/{{*}}SignalArrayByteOffset"
            )
        )
        assert self._signal_block_byte_offset is not None  # placate mypy
        self._file_object.seek(signal_offset + self._signal_block_byte_offset)
        shape, dtype = _describe_signal(self.metadata.xmltree, channel_identifier)
        dtype = dtype.newbyteorder(">")
        return _iohelp.fromfile(self._file_object, dtype, np.prod(shape)).reshape(shape)

    def read_signal_compressed(self) -> npt.NDArray:
        """Read signal data from a CRSD file with signal arrays stored in compressed format

        Returns
        -------
        ndarray
            Compressed signal byte sequence;
            shape=(CompressedSignalSize,), dtype= `numpy.uint8`

        Raises
        ------
        RuntimeError
            If the metadata indicates the signal block is not compressed

        See Also
        --------
        read_signal
        """
        compressed_size_str = self.metadata.xmltree.findtext(
            "{*}Data/{*}Receive/{*}SignalCompression/{*}CompressedSignalSize"
        )
        if compressed_size_str is None:
            raise RuntimeError(
                "Signal block is not compressed; use read_signal instead."
            )
        assert self._signal_block_byte_offset is not None  # placate mypy
        self._file_object.seek(self._signal_block_byte_offset)
        dtype = np.dtype("uint8")
        nbytes = int(compressed_size_str)
        return _iohelp.fromfile(self._file_object, dtype, nbytes)

    def read_pvps(self, channel_identifier: str) -> npt.NDArray:
        """Read pvp data from a CRSD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier

        Returns
        -------
        ndarray
            CRSD PVP array

        """
        channel_info = self.metadata.xmltree.find(
            f"{{*}}Data/{{*}}Receive/{{*}}Channel[{{*}}ChId='{channel_identifier}']"
        )
        num_vect = int(channel_info.find("./{*}NumVectors").text)

        pvp_offset = int(channel_info.find("./{*}PVPArrayByteOffset").text)
        assert self._pvp_block_byte_offset is not None  # placate mypy
        self._file_object.seek(pvp_offset + self._pvp_block_byte_offset)

        pvp_dtype = get_pvp_dtype(self.metadata.xmltree).newbyteorder("B")
        return _iohelp.fromfile(self._file_object, pvp_dtype, num_vect)

    def read_channel(self, channel_identifier: str) -> tuple[npt.NDArray, npt.NDArray]:
        """Read signal and pvp data from a CRSD file channel

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier

        Returns
        -------
        signal_array : ndarray
            Signal array for channel = channel_identifier
        pvp_array : ndarray
            PVP array for channel = channel_identifier

        """
        return self.read_signal(channel_identifier), self.read_pvps(channel_identifier)

    def read_ppps(self, sequence_identifier: str) -> npt.NDArray:
        """Read ppp data from a CRSD file

        Parameters
        ----------
        sequence_identifier : str
            Transmit sequence unique identifier

        Returns
        -------
        ndarray
            CRSD PPP array

        """
        channel_info = self.metadata.xmltree.find(
            f"{{*}}Data/{{*}}Transmit/{{*}}TxSequence[{{*}}TxId='{sequence_identifier}']"
        )
        num_pulse = int(channel_info.find("./{*}NumPulses").text)

        ppp_offset = int(channel_info.find("./{*}PPPArrayByteOffset").text)
        assert self._ppp_block_byte_offset is not None  # placate mypy
        self._file_object.seek(ppp_offset + self._ppp_block_byte_offset)

        ppp_dtype = get_ppp_dtype(self.metadata.xmltree).newbyteorder("B")
        return _iohelp.fromfile(self._file_object, ppp_dtype, num_pulse)

    def _read_support_array(self, sa_identifier):
        shape, dtype, _ = describe_support_array(self.metadata.xmltree, sa_identifier)
        dtype = dtype.newbyteorder("B")
        sa_info = self.metadata.xmltree.find(
            f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{sa_identifier}']"
        )
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
    """Write a CRSD file

    A Writer object can be used as a context manager in a ``with`` statement.

    Parameters
    ----------
    file : `file object`
        CRSD file to write
    metadata : Metadata
        CRSD metadata to write (copied on construction)

    See Also
    --------
    Reader

    Examples
    --------
    Generate some metadata and data

    .. doctest:: crsd_io

        >>> import lxml.etree

        >>> xmltree = lxml.etree.parse("data/example-crsd-1.0.xml")
        >>> first_sequence = xmltree.find("{*}Data/{*}Transmit/{*}TxSequence")
        >>> tx_id = first_sequence.findtext("{*}TxId")
        >>> num_p = int(first_sequence.findtext("{*}NumPulses"))
        >>> first_channel = xmltree.find("{*}Data/{*}Receive/{*}Channel")
        >>> ch_id = first_channel.findtext("{*}ChId")
        >>> num_v = int(first_channel.findtext("{*}NumVectors"))
        >>> num_s = int(first_channel.findtext("{*}NumSamples"))
        >>> sig_format = xmltree.findtext("{*}Data/{*}Receive/{*}SignalArrayFormat")

        >>> import sarkit.crsd as skcrsd

        >>> meta = skcrsd.Metadata(
        ...     xmltree=xmltree,
        ...     file_header_part=skcrsd.FileHeaderPart(additional_kvps={"K": "V"}),
        ... )

        >>> import numpy as np

        >>> sig = np.zeros((num_v, num_s), dtype=skcrsd.binary_format_string_to_dtype(sig_format))
        >>> pvps = np.zeros(num_v, dtype=skcrsd.get_pvp_dtype(xmltree))
        >>> ppps = np.zeros(num_p, dtype=skcrsd.get_ppp_dtype(xmltree))

    Write a channel's signal array and PVP arrays and a transmit sequence's PPP array to a file.

    .. doctest:: crsd_io

        >>> with (tmppath / "written.crsd").open("wb") as f, skcrsd.Writer(f, meta) as w:
        ...     w.write_signal(ch_id, sig)
        ...     w.write_pvp(ch_id, pvps)
        ...     w.write_ppp(tx_id, ppps)
    """

    def __init__(self, file, metadata: Metadata):
        align_to = 64
        self._file_object = file

        self._metadata = copy.deepcopy(metadata)
        crsd_xmltree = self._metadata.xmltree

        xml_block_body = lxml.etree.tostring(crsd_xmltree, encoding="utf-8")

        self._sequence_size_offsets = {}
        if crsd_xmltree.find("./{*}Data/{*}Transmit") is not None:
            ppp_itemsize = int(
                crsd_xmltree.find("./{*}Data/{*}Transmit/{*}NumBytesPPP").text
            )
            for seq_node in crsd_xmltree.findall("./{*}Data/{*}Transmit/{*}TxSequence"):
                sequence_identifier = seq_node.find("./{*}TxId").text
                sequence_ppp_offset = int(seq_node.find("./{*}PPPArrayByteOffset").text)
                sequence_ppp_size = (
                    int(seq_node.find("./{*}NumPulses").text) * ppp_itemsize
                )
                self._sequence_size_offsets[sequence_identifier] = {
                    "ppp_offset": sequence_ppp_offset,
                    "ppp_size": sequence_ppp_size,
                }

        self._channel_size_offsets = {}
        if crsd_xmltree.find("./{*}Data/{*}Receive") is not None:
            pvp_itemsize = int(
                crsd_xmltree.find("./{*}Data/{*}Receive/{*}NumBytesPVP").text
            )
            for chan_node in crsd_xmltree.findall("./{*}Data/{*}Receive/{*}Channel"):
                channel_identifier = chan_node.find("./{*}ChId").text
                channel_signal_offset = int(
                    chan_node.find("./{*}SignalArrayByteOffset").text
                )
                shape, dtype = _describe_signal(crsd_xmltree, channel_identifier)
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

        self._sa_size_offsets = {}
        for sa_node in crsd_xmltree.findall("./{*}Data/{*}Support/{*}SupportArray"):
            sa_identifier = sa_node.find("./{*}SAId").text
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

        support_block_size = max(
            sa["size"] + sa["offset"] for sa in self._sa_size_offsets.values()
        )

        def _align(val):
            return int(np.ceil(float(val) / align_to) * align_to)

        self._file_header_kvp = {
            "CLASSIFICATION": crsd_xmltree.findtext("{*}ProductInfo/{*}Classification"),
            "RELEASE_INFO": crsd_xmltree.findtext("{*}ProductInfo/{*}ReleaseInfo"),
            "XML_BLOCK_SIZE": len(xml_block_body),
            "XML_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
        }
        if self._sequence_size_offsets:
            ppp_block_size = max(
                seq["ppp_size"] + seq["ppp_offset"]
                for seq in self._sequence_size_offsets.values()
            )
            self._file_header_kvp.update(
                {
                    "PPP_BLOCK_SIZE": ppp_block_size,
                    "PPP_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
                }
            )
        if self._channel_size_offsets:
            compressed_size_str = crsd_xmltree.findtext(
                "{*}Data/{*}Receive/{*}SignalCompression/{*}CompressedSignalSize"
            )
            if compressed_size_str is not None:
                signal_block_size = int(compressed_size_str)
                if not all(
                    chan["signal_offset"] == 0
                    for chan in self._channel_size_offsets.values()
                ):
                    logging.warning(
                        "Signal compression is indicated but some SignalArrayByteOffsets are not 0"
                    )
            else:
                signal_block_size = max(
                    chan["signal_size"] + chan["signal_offset"]
                    for chan in self._channel_size_offsets.values()
                )
            pvp_block_size = max(
                chan["pvp_size"] + chan["pvp_offset"]
                for chan in self._channel_size_offsets.values()
            )

            self._file_header_kvp.update(
                {
                    "PVP_BLOCK_SIZE": pvp_block_size,
                    "PVP_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
                    "SIGNAL_BLOCK_SIZE": signal_block_size,
                    "SIGNAL_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
                }
            )
        self._file_header_kvp["SUPPORT_BLOCK_SIZE"] = support_block_size
        self._file_header_kvp["SUPPORT_BLOCK_BYTE_OFFSET"] = (
            np.iinfo(np.uint64).max,
        )  # placeholder

        self._file_header_kvp.update(self._metadata.file_header_part.additional_kvps)

        def _serialize_header():
            version = crsdconst.VERSION_INFO[
                lxml.etree.QName(crsd_xmltree.getroot()).namespace
            ]["version"]
            if self._sequence_size_offsets and self._channel_size_offsets:
                file_type = "CRSDsar"
            elif self._channel_size_offsets:
                file_type = "CRSDrcv"
            elif self._sequence_size_offsets:
                file_type = "CRSDtx"
            else:
                raise ValueError("Must have transmit sequences and/or receive channels")
            header_str = f"{file_type}/{version}\n"
            header_str += "".join(
                (f"{key} := {value}\n" for key, value in self._file_header_kvp.items())
            )
            return header_str.encode() + crsdconst.SECTION_TERMINATOR

        next_offset = _align(len(_serialize_header()))

        self._file_header_kvp["XML_BLOCK_BYTE_OFFSET"] = next_offset
        next_offset = _align(
            next_offset
            + self._file_header_kvp["XML_BLOCK_SIZE"]
            + len(crsdconst.SECTION_TERMINATOR)
        )

        self._file_header_kvp["SUPPORT_BLOCK_BYTE_OFFSET"] = next_offset
        next_offset = _align(next_offset + self._file_header_kvp["SUPPORT_BLOCK_SIZE"])

        if self._sequence_size_offsets:
            self._file_header_kvp["PPP_BLOCK_BYTE_OFFSET"] = next_offset
            next_offset = _align(next_offset + self._file_header_kvp["PPP_BLOCK_SIZE"])

        if self._channel_size_offsets:
            self._file_header_kvp["PVP_BLOCK_BYTE_OFFSET"] = next_offset
            next_offset = _align(next_offset + self._file_header_kvp["PVP_BLOCK_SIZE"])
            self._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"] = next_offset
            next_offset = _align(
                next_offset + self._file_header_kvp["SIGNAL_BLOCK_SIZE"]
            )

        self._file_object.seek(0)
        self._file_object.write(_serialize_header())
        self._file_object.seek(self._file_header_kvp["XML_BLOCK_BYTE_OFFSET"])
        self._file_object.write(xml_block_body + crsdconst.SECTION_TERMINATOR)

        self._signal_arrays_written: set[str] = set()
        self._pvp_arrays_written: set[str] = set()
        self._ppp_arrays_written: set[str] = set()
        self._support_arrays_written: set[str] = set()

    def write_signal(self, channel_identifier: str, signal_array: npt.NDArray):
        """Write signal data to a CRSD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier
        signal_array : ndarray
            Signal data to write;
            shape=(NumVectors, NumSamples), dtype determined by SignalArrayFormat.

        Raises
        ------
        RuntimeError
            If the signal block is compressed.

        See Also
        --------
        write_signal_compressed
        """
        # TODO Add support for partial CRSD writing
        data_rcv = self._metadata.xmltree.find("{*}Data/{*}Receive")
        if data_rcv.find("{*}SignalCompression") is not None:
            raise RuntimeError(
                "Signal block is compressed; use write_signal_compressed instead."
            )
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

    def write_signal_compressed(self, signal_array: npt.NDArray):
        """Write signal data in compressed format to a CRSD file

        Parameters
        ----------
        signal_array : ndarray
            Compressed signal byte sequence to write;
            shape=(CompressedSignalSize,), dtype= `numpy.uint8`

        Raises
        ------
        RuntimeError
            If the metadata indicates the signal block is not compressed

        See Also
        --------
        write_signal
        """
        compressed_size_str = self._metadata.xmltree.findtext(
            "{*}Data/{*}Receive/{*}SignalCompression/{*}CompressedSignalSize"
        )
        if compressed_size_str is None:
            raise RuntimeError(
                "Signal block is not compressed; use write_signal instead."
            )

        shape = (int(compressed_size_str),)
        dtype = np.dtype("uint8")
        if dtype != signal_array.dtype:
            raise ValueError(f"{signal_array.dtype=} is not compatible with {dtype=}")
        if shape != signal_array.shape:
            raise ValueError(f"{signal_array.shape=} does not match {shape=}")

        buff_to_write = signal_array.data
        expected_nbytes = self._file_header_kvp["SIGNAL_BLOCK_SIZE"]
        if buff_to_write.nbytes != expected_nbytes:
            raise ValueError(
                f"{buff_to_write.nbytes=} does not match {expected_nbytes=}"
            )

        self._file_object.seek(self._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"])
        self._file_object.write(buff_to_write)
        self._signal_arrays_written.update(self._channel_size_offsets.keys())

    def write_pvp(self, channel_identifier: str, pvp_array: npt.NDArray):
        """Write pvp data to a CRSD file

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

    def write_ppp(self, sequence_identifier: str, ppp_array: npt.NDArray):
        """Write ppp data to a CRSD file

        Parameters
        ----------
        sequence_identifier : str
            Sequence unique identifier
        ppp_array : ndarray
            Array of PPPs

        """
        assert (
            ppp_array.nbytes
            == self._sequence_size_offsets[sequence_identifier]["ppp_size"]
        )

        self._ppp_arrays_written.add(sequence_identifier)
        self._file_object.seek(self._file_header_kvp["PPP_BLOCK_BYTE_OFFSET"])
        self._file_object.seek(
            self._sequence_size_offsets[sequence_identifier]["ppp_offset"], os.SEEK_CUR
        )
        output_dtype = ppp_array.dtype.newbyteorder(">")
        ppp_array.astype(output_dtype, copy=False).tofile(self._file_object)

    def write_support_array(
        self, support_array_identifier: str, support_array: npt.NDArray
    ):
        """Write support array data to a CRSD file

        Parameters
        ----------
        support_array_identifier : str
            Unique support array identifier
        support_array : ndarray
            Array of support data

        """
        expected_shape, expected_dtype, sa_elem = describe_support_array(
            self._metadata.xmltree, support_array_identifier
        )
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
                "./{*}Data/{*}Receive/{*}Channel/{*}ChId"
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

        sequence_names = set(
            node.text
            for node in self._metadata.xmltree.findall(
                "./{*}Data/{*}Transmit/{*}TxSequence/{*}TxId"
            )
        )
        missing_ppp_sequences = sequence_names - self._ppp_arrays_written
        if missing_ppp_sequences:
            logging.warning(
                f"Not all PPP Arrays written.  Missing {missing_ppp_sequences}"
            )

        sa_names = set(
            node.text
            for node in self._metadata.xmltree.findall(
                "./{*}Data/{*}SupportArray/{*}SAId"
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
