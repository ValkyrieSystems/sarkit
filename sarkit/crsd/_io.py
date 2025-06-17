"""
Functions to read and write CRSD files.
"""

import copy
import dataclasses
import importlib.resources
import logging
import os
from typing import Final

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.cphd as skcphd

SCHEMA_DIR = importlib.resources.files("sarkit.crsd.schemas")
SECTION_TERMINATOR: Final[bytes] = b"\f\n"
DEFINED_HEADER_KEYS: Final[set] = {
    "XML_BLOCK_SIZE",
    "XML_BLOCK_BYTE_OFFSET",
    "SUPPORT_BLOCK_SIZE",
    "SUPPORT_BLOCK_BYTE_OFFSET",
    "PPP_BLOCK_SIZE",
    "PPP_BLOCK_BYTE_OFFSET",
    "PVP_BLOCK_SIZE",
    "PVP_BLOCK_BYTE_OFFSET",
    "SIGNAL_BLOCK_SIZE",
    "SIGNAL_BLOCK_BYTE_OFFSET",
    "CLASSIFICATION",
    "RELEASE_INFO",
}

VERSION_INFO: Final[dict] = {
    "http://api.nsgreg.nga.mil/schema/crsd/1.0": {
        "version": "1.0",
        "date": "2025-02-25T00:00:00Z",
        "schema": SCHEMA_DIR / "NGA.STND.0080-2_1.0_CRSD_schema_2025_02_25.xsd",
    },
}


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

        extra_header_keys = set(self._kvp_list.keys()) - DEFINED_HEADER_KEYS
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
            2D array of complex samples

        """
        channel_info = self.metadata.xmltree.find(
            f"{{*}}Data/{{*}}Receive/{{*}}Channel[{{*}}ChId='{channel_identifier}']"
        )
        num_vect = int(channel_info.find("./{*}NumVectors").text)
        num_samp = int(channel_info.find("./{*}NumSamples").text)
        shape = (num_vect, num_samp)

        signal_offset = int(channel_info.find("./{*}SignalArrayByteOffset").text)
        assert self._signal_block_byte_offset is not None  # placate mypy
        self._file_object.seek(signal_offset + self._signal_block_byte_offset)

        signal_dtype = binary_format_string_to_dtype(
            self.metadata.xmltree.find("./{*}Data/{*}Receive/{*}SignalArrayFormat").text
        ).newbyteorder("B")

        return np.fromfile(
            self._file_object, signal_dtype, count=np.prod(shape)
        ).reshape(shape)

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
        return np.fromfile(self._file_object, pvp_dtype, count=num_vect)

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
        return np.fromfile(self._file_object, ppp_dtype, count=num_pulse)

    def _read_support_array(self, sa_identifier):
        elem_format = self.metadata.xmltree.find(
            f"{{*}}SupportArray/*[{{*}}Identifier='{sa_identifier}']/{{*}}ElementFormat"
        )
        dtype = binary_format_string_to_dtype(elem_format.text).newbyteorder("B")

        sa_info = self.metadata.xmltree.find(
            f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{sa_identifier}']"
        )
        num_rows = int(sa_info.find("./{*}NumRows").text)
        num_cols = int(sa_info.find("./{*}NumCols").text)
        shape = (num_rows, num_cols)

        sa_offset = int(sa_info.find("./{*}ArrayByteOffset").text)
        self._file_object.seek(sa_offset + self._support_block_byte_offset)
        assert dtype.itemsize == int(sa_info.find("./{*}BytesPerElement").text)
        array = np.fromfile(self._file_object, dtype, count=np.prod(shape)).reshape(
            shape
        )
        return array

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
            signal_itemsize = binary_format_string_to_dtype(
                crsd_xmltree.find("./{*}Data/{*}Receive/{*}SignalArrayFormat").text
            ).itemsize
            pvp_itemsize = int(
                crsd_xmltree.find("./{*}Data/{*}Receive/{*}NumBytesPVP").text
            )
            for chan_node in crsd_xmltree.findall("./{*}Data/{*}Receive/{*}Channel"):
                channel_identifier = chan_node.find("./{*}ChId").text
                channel_signal_offset = int(
                    chan_node.find("./{*}SignalArrayByteOffset").text
                )
                channel_signal_size = (
                    int(chan_node.find("./{*}NumVectors").text)
                    * int(chan_node.find("./{*}NumSamples").text)
                    * signal_itemsize
                )

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
            version = VERSION_INFO[lxml.etree.QName(crsd_xmltree.getroot()).namespace][
                "version"
            ]
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
            return header_str.encode() + SECTION_TERMINATOR

        next_offset = _align(len(_serialize_header()))

        self._file_header_kvp["XML_BLOCK_BYTE_OFFSET"] = next_offset
        next_offset = _align(
            next_offset
            + self._file_header_kvp["XML_BLOCK_SIZE"]
            + len(SECTION_TERMINATOR)
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
        self._file_object.write(xml_block_body + SECTION_TERMINATOR)

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
            2D array of complex samples

        """
        # TODO Add support for partial CRSD writing
        assert (
            signal_array.nbytes
            == self._channel_size_offsets[channel_identifier]["signal_size"]
        )

        self._signal_arrays_written.add(channel_identifier)
        self._file_object.seek(self._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"])
        self._file_object.seek(
            self._channel_size_offsets[channel_identifier]["signal_offset"], os.SEEK_CUR
        )
        output_dtype = signal_array.dtype.newbyteorder(">")
        signal_array.astype(output_dtype, copy=False).tofile(self._file_object)

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
        # TODO: support masked arrays ala CPHD
        assert (
            support_array.nbytes
            == self._sa_size_offsets[support_array_identifier]["size"]
        )

        self._support_arrays_written.add(support_array_identifier)
        self._file_object.seek(self._file_header_kvp["SUPPORT_BLOCK_BYTE_OFFSET"])
        self._file_object.seek(
            self._sa_size_offsets[support_array_identifier]["offset"], os.SEEK_CUR
        )
        output_dtype = support_array.dtype.newbyteorder(">")
        support_array.astype(output_dtype, copy=False).tofile(self._file_object)

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
