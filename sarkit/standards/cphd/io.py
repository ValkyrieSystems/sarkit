"""
========
CPHD I/O
========

Functions to read and write CPHD files.
"""

import dataclasses
import importlib.resources
import logging
import os

import lxml.etree
import numpy as np
import numpy.typing as npt

SCHEMA_DIR = importlib.resources.files("sarkit.standards.cphd.schemas")
CPHD_SECTION_TERMINATOR = b"\f\n"
DEFINED_HEADER_KEYS = {
    "XML_BLOCK_SIZE",
    "XML_BLOCK_BYTE_OFFSET",
    "SUPPORT_BLOCK_SIZE",
    "SUPPORT_BLOCK_BYTE_OFFSET",
    "PVP_BLOCK_SIZE",
    "PVP_BLOCK_BYTE_OFFSET",
    "SIGNAL_BLOCK_SIZE",
    "SIGNAL_BLOCK_BYTE_OFFSET",
    "CLASSIFICATION",
    "RELEASE_INFO",
}

# Keys in ascending order
VERSION_INFO = {
    "http://api.nsgreg.nga.mil/schema/cphd/1.0.1": {
        "version": "1.0.1",
        "date": "2018-05-21T00:00:00Z",
        "schema": SCHEMA_DIR / "CPHD_schema_V1.0.1_2018_05_21.xsd",
    },
    "http://api.nsgreg.nga.mil/schema/cphd/1.1.0": {
        "version": "1.1.0",
        "date": "2021-11-30T00:00:00Z",
        "schema": SCHEMA_DIR / "CPHD_schema_V1.1.0_2021_11_30_FINAL.xsd",
    },
}


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
    """Return the CPHD Binary Format string (table 10-2) description of a numpy.dtype.

    Parameters
    ----------
    dtype : `numpy.dtype`
        (e.g., numpy.int8, numpy.int32, numpy.complex64, etc.)

        The dtype to be converted to PVP type string.

    Returns
    -------
    result : str
        PVP type designator for the specified `numpy.dtype`
        (e.g., ``"I1"``, ``"I4"``, ``"CF8"``, etc.).

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
    """Return the numpy.dtype for CPHD Binary Format string (table 10-2).

    Parameters
    ----------
    format_string : str
        PVP type designator (e.g., ``"I1"``, ``"I4"``, ``"CF8"``, etc.).

    Returns
    -------
    dtype : `numpy.dtype`
        The equivalent `numpy.dtype` of the PVP format string
        (e.g., numpy.int8, numpy.int32, numpy.complex64, etc.).

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


@dataclasses.dataclass(kw_only=True)
class CphdFileHeaderFields:
    """CPHD header fields which are set per program specific Product Design Document

    Attributes
    ----------
    classification : str
        File classification
    release_info : str
        File release info
    additional_kvps : dict of {str : str}, optional
        Dictionary with additional key-value pairs
    """

    classification: str
    release_info: str
    additional_kvps: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.additional_kvps is None:
            self.additional_kvps = {}


@dataclasses.dataclass(kw_only=True)
class CphdPlan:
    """Class describing the plan for creating a CPHD file

    Attributes
    ----------
    file_header : :py:class:`CphdFileHeaderFields`
        CPHD File Header fields which can be set
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML ElementTree

    See Also
    --------
    CphdReader
    CphdWriter
    CphdFileHeaderFields
    """

    file_header: CphdFileHeaderFields
    cphd_xmltree: lxml.etree.ElementTree


def read_file_header(file):
    """Read a CPHD file header.

    The file object's position is assumed to be at the start of the file.

    Parameters
    ----------
    file : file_like
        The open file object, which will be progressively read.

    Returns
    -------
    file_type_header : str
        File type from the first line of a CPHD file
    kvp_list : dict[str, str]
        Key-Value list of header fields
    """
    file_type_header = file.readline().decode().strip("\n")

    kvp_list = {}
    while (line := file.readline()) != CPHD_SECTION_TERMINATOR:
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


class CphdReader:
    """Read a CPHD file

    A CphdReader object can be used as a context manager in a ``with`` statement.

    Parameters
    ----------
    file : `file object`
        CPHD file to read

    Examples
    --------
    >>> with cphd_path.open('rb') as file, CphdReader(file) as reader:
    ...     cphd_xmltree = reader.cphd_xmltree
    ...     signal, pvp = reader.read_channel(<chan_id>)

    Attributes
    ----------
    file_header : :py:class:`CphdFileHeaderFields`
       CPHD header dataclass
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML ElementTree
    plan : :py:class:`CphdPlan`
        A CphdPlan object suitable for use in a CphdWriter
    xml_block_size : int
    xml_block_byte_offset : int
    pvp_block_size : int
    pvp_block_byte_offset : int
    signal_block_size : int
    signal_block_byte_offset : int
    support_block_size : int or None
    support_block_byte_offset : int or None

    See Also
    --------
    CphdPlan
    CphdWriter
    """

    def __init__(self, file):
        self._file_object = file

        # skip the version line and read header
        _, self._kvp_list = read_file_header(self._file_object)

        extra_header_keys = set(self._kvp_list.keys()) - DEFINED_HEADER_KEYS
        additional_kvps = {key: self._kvp_list[key] for key in extra_header_keys}

        self.file_header = CphdFileHeaderFields(
            classification=self._kvp_list["CLASSIFICATION"],
            release_info=self._kvp_list["RELEASE_INFO"],
            additional_kvps=additional_kvps,
        )
        self._file_object.seek(self.xml_block_byte_offset)
        xml_bytes = self._file_object.read(int(self._kvp_list["XML_BLOCK_SIZE"]))
        self.cphd_xmltree = lxml.etree.fromstring(xml_bytes).getroottree()

        self.plan = CphdPlan(
            cphd_xmltree=self.cphd_xmltree,
            file_header=self.file_header,
        )

    @property
    def xml_block_byte_offset(self) -> int:
        """Offset to the XML block"""
        return int(self._kvp_list["XML_BLOCK_BYTE_OFFSET"])

    @property
    def xml_block_size(self) -> int:
        """Size of the XML block"""
        return int(self._kvp_list["XML_BLOCK_SIZE"])

    @property
    def pvp_block_byte_offset(self) -> int:
        """Offset to the PVP block"""
        return int(self._kvp_list["PVP_BLOCK_BYTE_OFFSET"])

    @property
    def pvp_block_size(self) -> int:
        """Size of the PVP block"""
        return int(self._kvp_list["PVP_BLOCK_SIZE"])

    @property
    def signal_block_byte_offset(self) -> int:
        """Offset to the Signal block"""
        return int(self._kvp_list["SIGNAL_BLOCK_BYTE_OFFSET"])

    @property
    def signal_block_size(self) -> int:
        """Size of the Signal block"""
        return int(self._kvp_list["SIGNAL_BLOCK_SIZE"])

    @property
    def support_block_byte_offset(self) -> int | None:
        """Offset to the Support block"""
        if "SUPPORT_BLOCK_BYTE_OFFSET" in self._kvp_list:
            return int(self._kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"])
        else:
            return None

    @property
    def support_block_size(self) -> int | None:
        """Size of the Support block"""
        if "SUPPORT_BLOCK_SIZE" in self._kvp_list:
            return int(self._kvp_list["SUPPORT_BLOCK_SIZE"])
        else:
            return None

    def read_signal(self, channel_identifier: str) -> npt.NDArray:
        """Read signal data from a CPHD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier

        Returns
        -------
        ndarray
            2D array of complex pixels


        """
        channel_info = self.cphd_xmltree.find(
            f"{{*}}Data/{{*}}Channel[{{*}}Identifier='{channel_identifier}']"
        )
        num_vect = int(channel_info.find("./{*}NumVectors").text)
        num_samp = int(channel_info.find("./{*}NumSamples").text)
        shape = (num_vect, num_samp)

        signal_offset = int(channel_info.find("./{*}SignalArrayByteOffset").text)
        self._file_object.seek(signal_offset + self.signal_block_byte_offset)

        signal_dtype = binary_format_string_to_dtype(
            self.cphd_xmltree.find("./{*}Data/{*}SignalArrayFormat").text
        ).newbyteorder("B")

        return np.fromfile(
            self._file_object, signal_dtype, count=np.prod(shape)
        ).reshape(shape)

    def read_pvps(self, channel_identifier: str) -> npt.NDArray:
        """Read pvp data from a CPHD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier

        Returns
        -------
        ndarray
            CPHD PVP array

        """
        channel_info = self.cphd_xmltree.find(
            f"{{*}}Data/{{*}}Channel[{{*}}Identifier='{channel_identifier}']"
        )
        num_vect = int(channel_info.find("./{*}NumVectors").text)

        pvp_offset = int(channel_info.find("./{*}PVPArrayByteOffset").text)
        self._file_object.seek(pvp_offset + self.pvp_block_byte_offset)

        pvp_dtype = get_pvp_dtype(self.cphd_xmltree).newbyteorder("B")
        return np.fromfile(self._file_object, pvp_dtype, count=num_vect)

    def read_channel(self, channel_identifier: str) -> tuple[npt.NDArray, npt.NDArray]:
        """Read signal and pvp data from a CPHD file channel

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

    def read_support_array(self, sa_identifier):
        """Read SupportArray"""
        elem_format = self.cphd_xmltree.find(
            f"{{*}}SupportArray/*[{{*}}Identifier='{sa_identifier}']/{{*}}ElementFormat"
        )
        dtype = binary_format_string_to_dtype(elem_format.text).newbyteorder("B")

        sa_info = self.cphd_xmltree.find(
            f"{{*}}Data/{{*}}SupportArray[{{*}}Identifier='{sa_identifier}']"
        )
        num_rows = int(sa_info.find("./{*}NumRows").text)
        num_cols = int(sa_info.find("./{*}NumCols").text)
        shape = (num_rows, num_cols)

        sa_offset = int(sa_info.find("./{*}ArrayByteOffset").text)
        self._file_object.seek(sa_offset + self.support_block_byte_offset)
        assert dtype.itemsize == int(sa_info.find("./{*}BytesPerElement").text)

        return np.fromfile(self._file_object, dtype, count=np.prod(shape)).reshape(
            shape
        )

    def done(self):
        "Indicates to the reader that the user is done with it"
        self._file_object = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.done()


class CphdWriter:
    """Write a CPHD file

    A CphdWriter object can be used as a context manager in a ``with`` statement.

    Parameters
    ----------
    file : `file object`
        CPHD file to write
    plan : :py:class:`CphdPlan`
        A CphdPlan object

    Notes
    -----
    plan should not be modified after creation of a writer

    Examples
    --------
    >>> with output_path.open('wb') as file, CphdWriter(file, plan) as writer:
    ...     writer.write_signal("1", signal)
    ...     writer.write_pvp("1", pvp)

    See Also
    --------
    CphdPlan
    CphdReader
    """

    def __init__(self, file, plan):
        self._file_object = file

        self._plan = plan

        xml_str = lxml.etree.tostring(plan.cphd_xmltree)

        signal_itemsize = binary_format_string_to_dtype(
            plan.cphd_xmltree.find("./{*}Data/{*}SignalArrayFormat").text
        ).itemsize
        pvp_itemsize = int(plan.cphd_xmltree.find("./{*}Data/{*}NumBytesPVP").text)
        self._channel_size_offsets = {}
        for chan_node in plan.cphd_xmltree.findall("./{*}Data/{*}Channel"):
            channel_identifier = chan_node.find("./{*}Identifier").text
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

        signal_block_size = sum(
            chan["signal_size"] for chan in self._channel_size_offsets.values()
        )
        pvp_block_size = sum(
            chan["pvp_size"] for chan in self._channel_size_offsets.values()
        )

        self._sa_size_offsets = {}
        for sa_node in plan.cphd_xmltree.findall("./{*}Data/{*}SupportArray"):
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

        support_block_size = sum(sa["size"] for sa in self._sa_size_offsets.values())

        def _align(val):
            align_to = 64
            return int(np.ceil(float(val) / align_to) * align_to)

        # TODO Pad out the header?

        self._file_header_kvp = {
            "XML_BLOCK_SIZE": len(xml_str),
            "XML_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
            "PVP_BLOCK_SIZE": pvp_block_size,
            "PVP_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
            "SIGNAL_BLOCK_SIZE": signal_block_size,
            "SIGNAL_BLOCK_BYTE_OFFSET": np.iinfo(np.uint64).max,  # placeholder
            "CLASSIFICATION": plan.file_header.classification,
            "RELEASE_INFO": plan.file_header.release_info,
        }
        if self._sa_size_offsets:
            self._file_header_kvp["SUPPORT_BLOCK_SIZE"] = support_block_size
            self._file_header_kvp["SUPPORT_BLOCK_BYTE_OFFSET"] = (
                np.iinfo(np.uint64).max,
            )  # placeholder

        self._file_header_kvp.update(plan.file_header.additional_kvps)

        def _serialize_header():
            version = VERSION_INFO[
                lxml.etree.QName(plan.cphd_xmltree.getroot()).namespace
            ]["version"]
            header_str = f"CPHD/{version}\n"
            header_str += "".join(
                (f"{key} := {value}\n" for key, value in self._file_header_kvp.items())
            )
            return header_str.encode()

        max_file_header_size = _align(len(_serialize_header()))

        self._file_header_kvp["XML_BLOCK_BYTE_OFFSET"] = max_file_header_size + len(
            CPHD_SECTION_TERMINATOR
        )

        if self._sa_size_offsets:
            self._file_header_kvp["SUPPORT_BLOCK_BYTE_OFFSET"] = (
                self._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"]
                + self._file_header_kvp["SIGNAL_BLOCK_SIZE"]
                + len(CPHD_SECTION_TERMINATOR)
            )

        self._file_header_kvp["PVP_BLOCK_BYTE_OFFSET"] = (
            self._file_header_kvp["XML_BLOCK_BYTE_OFFSET"]
            + self._file_header_kvp["XML_BLOCK_SIZE"]
            + len(CPHD_SECTION_TERMINATOR)
        )
        self._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"] = (
            self._file_header_kvp["PVP_BLOCK_BYTE_OFFSET"]
            + self._file_header_kvp["PVP_BLOCK_SIZE"]
            + len(CPHD_SECTION_TERMINATOR)
        )
        self._file_object.seek(max_file_header_size + len(CPHD_SECTION_TERMINATOR))
        self._file_header_kvp["XML_BLOCK_BYTE_OFFSET"] = self._file_object.tell()
        self._file_object.write(xml_str)
        self._file_object.write(CPHD_SECTION_TERMINATOR)

        if self._sa_size_offsets:
            self._file_header_kvp["SUPPORT_BLOCK_BYTE_OFFSET"] = (
                self._file_object.tell()
            )
            self._file_object.seek(
                self._file_header_kvp["SUPPORT_BLOCK_SIZE"], os.SEEK_CUR
            )

        self._file_header_kvp["PVP_BLOCK_BYTE_OFFSET"] = self._file_object.tell()
        self._file_object.seek(self._file_header_kvp["PVP_BLOCK_SIZE"], os.SEEK_CUR)

        self._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"] = self._file_object.tell()
        self._file_object.seek(self._file_header_kvp["SIGNAL_BLOCK_SIZE"], os.SEEK_CUR)

        self._file_object.seek(0)
        self._file_object.write(_serialize_header())
        self._file_object.write(CPHD_SECTION_TERMINATOR)

        self._signal_arrays_written = set()
        self._pvp_arrays_written = set()
        self._support_arrays_written = set()

    def write_signal(self, channel_identifier: str, signal_array: npt.NDArray):
        """Write signal data to a CPHD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier
        signal_array : ndarray
            2D array of complex pixels

        """
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

        # TODO Add support for partial CPHD writing
        return

    def write_pvp(self, channel_identifier: str, pvp_array: npt.NDArray):
        """Write pvp data to a CPHD file

        Parameters
        ----------
        channel_identifier : str
            Channel unique identifier
        signal_array : ndarray
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
        return

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
            for node in self._plan.cphd_xmltree.findall(
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
            for node in self._plan.cphd_xmltree.findall(
                "./{*}Data/{*}SupportArray/{*}Identifier"
            )
        )
        missing_sa = sa_names - self._support_arrays_written
        if missing_sa:
            logging.warning(f"Not all Support Arrays written.  Missing {missing_sa}")

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.done()
