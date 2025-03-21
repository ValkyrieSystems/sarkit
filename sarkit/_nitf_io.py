"""Utilities for reading and writing NITFs

The Field, Group, SubHeaders, and Nitf classes have a dictionary-esque interface
with key names directly copied from MIL-STD-2500C where possible.

In NITF, the presence of optional fields is controlled by the values of preceding
fields.  This library attempts to mimic this behavior by adding or removing fields
as necessary when a field is updated.  For example adding image segments is accomplished
by setting the NUMI field.

Setting the value of fields is done using the `value` property.  `value` uses common python
types (int, str, etc...) and serializes to the NITF format behind the scenes.
"""

import argparse
import datetime
import logging
import os
import pathlib
import re

import numpy as np

logger = logging.getLogger(__name__)


class PythonConverter:
    """Class for converting between NITF field bytes and python types"""

    def __init__(self, name, size):
        self.name = name
        self.size = size

    def to_bytes(self, decoded_value):
        encoded = self.to_bytes_impl(decoded_value)
        truncated = encoded[: self.size]
        if len(truncated) < len(encoded):
            logger.warning(
                f"NITF header field {self.name} will be truncated to {self.size} characters.\n"
                f"    old: {encoded}"
                f"    new: {truncated}"
            )
        return truncated

    def to_bytes_impl(self, decoded_value):
        raise NotImplementedError()

    def from_bytes(self, encoded_value):
        return self.from_bytes_impl(encoded_value)

    def from_bytes_impl(self, encoded_value):
        raise NotImplementedError()


class StringUtf8(PythonConverter):
    """Convert to/from str"""

    def to_bytes_impl(self, decoded_value):
        return str.ljust(decoded_value, self.size).encode()

    def from_bytes_impl(self, encoded_value):
        return encoded_value.decode().rstrip(" ")


class StringAscii(PythonConverter):
    """Convert to/from str"""

    def to_bytes_impl(self, decoded_value):
        return str.ljust(decoded_value, self.size).encode("ascii")

    def from_bytes_impl(self, encoded_value):
        return encoded_value.decode("ascii").rstrip(" ")


class StringISO8859_1(PythonConverter):  # noqa: N801
    """Convert to/from an ISO 8859-1 str

    Note
    ----
    JBP-2021.2 Table D-1 specifies the full ECS-A character set, which
    happens to match ISO 8859 part 1.
    """

    def to_bytes_impl(self, decoded_value):
        return str.ljust(decoded_value, self.size).encode("iso8859_1")

    def from_bytes_impl(self, encoded_value):
        return encoded_value.decode("iso8859_1").rstrip(" ")


def int_pair(length):
    """returns Python Converter class for handling 2 concatenated ints to/from a tuple"""

    class IntPair(PythonConverter):
        def to_bytes_impl(self, decoded_value):
            return (
                f"{decoded_value[0]:0{length}d}{decoded_value[1]:0{length}d}".encode()
            )

        def from_bytes_impl(self, encoded_value):
            return (int(encoded_value[0:length]), int(encoded_value[length:]))

    return IntPair


class Bytes(PythonConverter):
    """Convert to/from bytes"""

    def to_bytes_impl(self, decoded_value):
        return decoded_value

    def from_bytes_impl(self, encoded_value):
        return encoded_value


class Integer(PythonConverter):
    """convert to/from int"""

    def to_bytes_impl(self, decoded_value):
        decoded_value = int(decoded_value)
        return f"{decoded_value:0{self.size}}".encode()

    def from_bytes_impl(self, encoded_value):
        return int(encoded_value)


class RGB(PythonConverter):
    """convert to/from three int tuple"""

    def to_bytes_impl(self, decoded_value):
        assert self.size == 3
        return (
            decoded_value[0].to_bytes(1, "big")
            + decoded_value[1].to_bytes(1, "big")
            + decoded_value[2].to_bytes(1, "big")
        )

    def from_bytes_impl(self, encoded_value):
        return tuple(encoded_value)


# Character sets
# Extended Character Set (ECS) (see 5.1.7.a.2)
ECS = "\x20-\x7e\xa0-\xff\x0a\x0c\x0d"
# Extended Character Set - Alphanumeric (ECS-A) (see 5.1.7.a.3)
ECSA = "\x20-\x7e\xa0-\xff"
# Basic Character Set (BCS) (see 5.1.7.a.4)
BCS = "\x20-\x7e\x0a\x0c\x0d"
# Basic Character Set - Alphanumeric (BCS-A) (see 5.1.7.a.5)
BCSA = "\x20-\x7e"
# Basic Character Set - Numeric (BCS-N) (see 5.1.7.a.6)
BCSN = "\x30-\x39\x2b\x2d\x2e\x2f"
# Basic Character Set - Numeric Integer (BCS-N integer) (see 5.1.7.a.7)
BCSN_I = "\x30-\x39\x2b\x2d"
# Basic Character Set - Numeric Positive Integer (BCS-N positive integer) (see 5.1.7.a.8)
BCSN_PI = "\x30-\x39"
# UTF-8
U8 = "\x00-\xff"


class RangeCheck:
    """Base Class for checking the range of a NITF field"""

    def isvalid(self, decoded_value):
        raise NotImplementedError()


class AnyRange(RangeCheck):
    """Field has no range restrictions"""

    def isvalid(self, decoded_value):
        return True


class MinMax(RangeCheck):
    """Field has a minimum and/or maximum value

    Args
    ----
    minimum:
        Minimum value.  A value of 'None' indicates no minimum.
    maximum:
        Maximum value.  A value of 'None' indicates no maximum.

    """

    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def isvalid(self, decoded_value):
        valid = True
        if self.minimum is not None:
            valid &= decoded_value >= self.minimum
        if self.maximum is not None:
            valid &= decoded_value <= self.maximum
        return valid


class Regex(RangeCheck):
    """Field value is restricted by a regex"""

    def __init__(self, pattern):
        self.pattern = pattern

    def isvalid(self, decoded_value):
        return bool(re.fullmatch(self.pattern, decoded_value))


class Constant(RangeCheck):
    """Field value must be a constant"""

    def __init__(self, const):
        self.const = const

    def isvalid(self, decoded_value):
        return decoded_value == self.const


class Enum(RangeCheck):
    """Field value must match one value of an Enumeration"""

    def __init__(self, enumeration):
        self.enumeration = set(enumeration)

    def isvalid(self, decoded_value):
        return decoded_value in self.enumeration


class AnyOf(RangeCheck):
    """Field value must match one of many different RangeChecks

    Args
    ----
    *ranges: RangeCheck
        RangeCheck objects to check against

    """

    def __init__(self, *ranges):
        self.ranges = ranges

    def isvalid(self, decoded_value):
        checks = [check.isvalid(decoded_value) for check in self.ranges]
        return any(checks)


class Not(RangeCheck):
    def __init__(self, range_check):
        self.range_check = range_check

    def isvalid(self, decoded_value):
        return not self.range_check.isvalid(decoded_value)


# Common Regex patterns
PATTERN_CC = "[0-9]{2}"
PATTERN_YY = "[0-9]{2}"
PATTERN_MM = "(0[1-9]|1[0-2])"  # MM
PATTERN_DD = "(0[1-9]|[12][0-9]|3[0-1])"  # DD
PATTERN_HH = "([0-1][0-9]|2[0-3])"  # hh
PATTERN_MM = "([0-5][0-9])"  # mm
PATTERN_SS = "([0-5][0-9]|60)"  # ss
DATETIME_REGEX = Regex(
    PATTERN_CC
    + PATTERN_YY
    + PATTERN_MM
    + PATTERN_DD
    + PATTERN_HH
    + PATTERN_MM
    + PATTERN_SS
)
DATE_REGEX = Regex(PATTERN_CC + PATTERN_YY + PATTERN_MM + PATTERN_DD)


class NitfIOComponent:
    """Base Class for read/writable NITF components"""

    def __init__(self, name):
        self.name = name
        self.parent = None

    def load(self, fd):
        """Read from a file descriptor
        Args
        ----
        fd: file-like
            File-like object to read from

        """
        try:
            self.load_impl(fd)
            return self
        except Exception:
            logger.error(f"Failed to read {self.name}")
            raise

    def dump(self, fd, seek_first=False):
        """Write to a file descriptor
        Args
        ----
        fd: file-like
            File-like object to write to
        seek_first: bool
            Seek to the components offset before writing

        """
        if seek_first:
            fd.seek(self.get_offset(), os.SEEK_SET)

        try:
            return self.dump_impl(fd)
        except Exception:
            logger.error(f"Failed to wite {self.name}")
            raise

    def load_impl(self, fd):
        raise NotImplementedError()

    def dump_impl(self, fd):
        raise NotImplementedError()

    def get_offset(self):
        """Return the offset from the start of the file to this component"""
        offset = 0
        if self.parent:
            offset = self.parent.get_offset_of(self)
        return offset

    def length(self):
        raise NotImplementedError()


class Field(NitfIOComponent):
    """NITF Field containing a single value.
    Intended to have 1:1 mapping to rows in MIL-STD-2500C header tables.

    Args
    ----
    name: str
        Name of this field
    description: str
        Text description of the field
    size: int
        Size in bytes of the field
    charset: str
        regex expression matching a single character
    range: `RangeCheck` object
        `RangeCheck` object to check acceptable values
    converter_class: `PythonConverter` class
        PythonConverter class to use to convert to/from python data types
    type: str
    default: any
        Initial python value of the field
    setter_callback: callable
        function to call if the field's value changes

    """

    def __init__(
        self,
        name,
        description,
        size,
        charset,
        range,
        converter_class,
        *,
        default=None,
        setter_callback=None,
    ):
        super().__init__(name)
        self.description = description
        self._size = size
        self.charset = charset
        self.range = range
        self._converter_class = converter_class
        self._converter = converter_class(name, size)
        self.parent = None
        self.setter_callback = setter_callback

        self.encoded_value = None
        if default is not None:
            self.encoded_value = self._converter.to_bytes(default)

            if not self.isvalid():
                logger.warning(
                    f"{self.name}: Invalid field value: {self.encoded_value}"
                )

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.description == other.description
            and self.charset == other.charset
            and self.encoded_value == other.encoded_value
        )

    def isvalid(self):
        valid_charset = (
            True
            if self.charset is None
            else bool(re.fullmatch(f"[{self.charset}]*", self.encoded_value.decode()))
        )

        valid_range = self.range.isvalid(self.value)
        return valid_charset and valid_range

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        old_value = self._size
        self._size = value
        self._converter = self._converter_class(self.name, self._size)

        if (old_value != self._size) and self.setter_callback:
            self.setter_callback(self)

    @property
    def value(self):
        return self._converter.from_bytes(self.encoded_value)

    @value.setter
    def value(self, val):
        self.encoded_value = self._converter.to_bytes(val)

        if not self.isvalid():
            logger.warning(f"{self.name}: Invalid field value: {self.encoded_value}")

        if self.setter_callback:
            self.setter_callback(self)

    def load_impl(self, fd):
        self.encoded_value = fd.read(self.size)
        if not self.isvalid():
            logger.warning(f"{self.name}: Invalid field value: {self.encoded_value}")

        if self.setter_callback:
            self.setter_callback(self)

    def dump_impl(self, fd):
        return fd.write(self.encoded_value)

    def length(self):
        return self.size

    def print(self):
        print(
            f"{self.name:15}{self.size:11} @ {self.get_offset():11} {self.encoded_value}"
        )


class BinaryPlaceholder(NitfIOComponent):
    """Represents a block of large binary data.

    This class does not actually read, write or store data, only seek past it.

    """

    def __init__(self, name, size):
        super().__init__(name)
        self._size = size

    def __eq__(self, other):
        return self.name == other.name and self._size == other._size

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    def load_impl(self, fd):
        fd.seek(self.size, os.SEEK_CUR)

    def dump_impl(self, fd):
        if self.size:
            fd.seek(self.size, os.SEEK_CUR)

    def length(self):
        return self.size

    def print(self):
        print(f"{self.name:15}{self.size:11} @ {self.get_offset():11} <Binary>")


class Group(NitfIOComponent):
    """
    A Collection of NITF fields

    Args
    ----
    name: str
        Name to give the group

    """

    def __init__(self, name):
        super().__init__(name)
        self.children = []
        self.parent = None

    def __eq__(self, other):
        return len(self.children) == len(other.children) and all(
            [left == right for left, right in zip(self.children, other.children)]
        )

    def __len__(self):
        return len(self.children)

    def keys(self):
        return [child.name for child in self.children]

    def load_impl(self, fd):
        for child in self.children:
            child.load(fd)

    def dump_impl(self, fd):
        for child in self.children:
            child.dump(fd)

    def append(self, field):
        field.parent = self
        self.children.append(field)

    def insert_after(self, existing, field):
        insert_pos = self.children.index(existing) + 1
        self.children.insert(insert_pos, field)
        field.parent = self
        return field

    def find_all(self, pattern):
        for child in self.children[:]:
            if re.fullmatch(pattern, child.name):
                yield child

    def remove_all(self, pattern):
        for child in self.find_all(pattern):
            self.children.remove(child)

    def length(self):
        size = 0
        for child in self.children:
            size += child.length()
        return size

    def get_offset_of(self, child_obj):
        offset = 0
        if self.parent:
            offset = self.parent.get_offset_of(self)

        for child in self.children:
            if child is child_obj:
                return offset
            else:
                offset += child.length()
        else:
            raise ValueError(f"Could not find {child_obj.name}")

    def index(self, name):
        return self.keys().index(name)

    def __getitem__(self, key):
        index = self.index(key)
        return self.children[index]

    def get(self, key, default=None):
        try:
            return self[key]
        except ValueError:
            return default

    def __contains__(self, key):
        try:
            self.index(key)
            return True
        except ValueError:
            return False

    def print(self):
        for child in self.children:
            child.print()


class SegmentList(Group):
    def __init__(self, name, field_creator, minimum=0, maximum=1):
        super().__init__(name)
        self.field_creator = field_creator
        self.minimum = minimum
        self.maximum = maximum
        self.children = []
        self.set_size(self.minimum)

    def __getitem__(self, idx):
        return self.children[idx]

    def set_size(self, size):
        if not self.minimum <= size <= self.maximum:
            raise ValueError(f"Invalid {size=}")
        for idx in range(len(self.children), size):
            new_field = self.field_creator(idx + 1)
            self.append(new_field)
        for _ in range(size, len(self.children)):
            self.children.pop()


class FileHeader(Group):
    """
    NITF File Header

    Args
    ----
    name: str
        Name to give the object
    numi_callback: callable
        Function to call when NUMI changes
    lin_callback: callable
        Function to call when LIn changes
    nums_callback: callable
        Function to call when NUMS changes
    lsshn_callback: callable
        Function to call when LSSHn changes
    lsn_callback: callable
        Function to call when LSn changes
    numt_callback: callable
        Function to call when NUMT changes
    ltshn_callback: callable
        Function to call when LTSHn changes
    ltn_callback: callable
        Function to call when LTn changes
    numdes_callback: callable
        Function to call when NUMDES changes
    ldn_callback: callable
        Function to call when LDn changes
    numres_callback: callable
        Function to call when NUMRES changes
    lreshn_callback: callable
        Function to call when LRESHn changes
    lren_callback: callable
        Function to call when LREn changes

    Note
    ----
    See MIL-STD-2500C Table A-1

    """

    def __init__(
        self,
        name,
        numi_callback=None,
        lin_callback=None,
        nums_callback=None,
        lsshn_callback=None,
        lsn_callback=None,
        numt_callback=None,
        ltshn_callback=None,
        ltn_callback=None,
        numdes_callback=None,
        ldn_callback=None,
        numres_callback=None,
        lreshn_callback=None,
        lren_callback=None,
    ):
        super().__init__(name)
        self.numi_callback = numi_callback
        self.lin_callback = lin_callback
        self.nums_callback = nums_callback
        self.lsshn_callback = lsshn_callback
        self.lsn_callback = lsn_callback
        self.numt_callback = numt_callback
        self.ltshn_callback = ltshn_callback
        self.ltn_callback = ltn_callback
        self.numdes_callback = numdes_callback
        self.ldn_callback = ldn_callback
        self.numres_callback = numres_callback
        self.lreshn_callback = lreshn_callback
        self.lren_callback = lren_callback

        # Initialize list with required fields
        self.append(
            Field(
                "FHDR",
                "File Profile Name",
                4,
                BCSA,
                Constant("NITF"),
                StringAscii,
                default="NITF",
            )
        )
        self.append(
            Field(
                "FVER",
                "File Version",
                5,
                BCSA,
                Constant("02.10"),
                StringAscii,
                default="02.10",
            )
        )
        self.append(
            Field("CLEVEL", "Complexity Level", 2, BCSN_PI, MinMax(1, 99), Integer)
        )
        self.append(
            Field(
                "STYPE",
                "Standard Type",
                4,
                BCSA,
                Constant("BF01"),
                StringAscii,
                default="BF01",
            )
        )
        self.append(
            Field(
                "OSTAID",
                "Originating Station ID",
                10,
                BCSA,
                Not(Constant("")),
                StringAscii,
            )
        )
        self.append(
            Field("FDT", "File Date and Time", 14, BCSN_I, DATETIME_REGEX, StringAscii)
        )
        self.append(
            Field(
                "FTITLE",
                "File Title",
                80,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSCLAS",
                "File Security Classification",
                1,
                ECSA,
                Enum(["T", "S", "C", "R", "U"]),
                StringISO8859_1,
            )
        )
        self.append(
            Field(
                "FSCLSY",
                "File Security Classification System",
                2,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSCODE",
                "File Codewords",
                11,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSCTLH",
                "File Control and Handling",
                2,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSREL",
                "File Release Instructions",
                20,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSDCTP",
                "File Declassification Type",
                2,
                ECSA,
                Enum(["", "DD", "DE", "GD", "GE", "O", "X"]),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSDCDT",
                "File Declassification Date",
                8,
                ECSA,
                AnyOf(Constant(""), DATE_REGEX),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSDCXM",
                "File Declassification Exemption",
                4,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSDG",
                "File Downgrade",
                1,
                ECSA,
                Enum(["", "S", "C", "R"]),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSDGDT",
                "File Downgrade Date",
                8,
                ECSA,
                AnyOf(Constant(""), DATE_REGEX),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSCLTX",
                "File Classification Text",
                43,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSCATP",
                "File Classification Authority Type",
                1,
                ECSA,
                Enum(["", "O", "D", "M"]),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSCAUT",
                "File Classification Authority",
                40,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSCRSN",
                "File Classification Reason",
                1,
                ECSA,
                AnyOf(Constant(""), Regex("[A-G]")),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSSRDT",
                "File Security Source Date",
                8,
                ECSA,
                AnyOf(Constant(""), DATE_REGEX),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSCTLN",
                "File Security Control Number",
                15,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FSCOP", "File Copy Number", 5, BCSN_PI, AnyRange(), Integer, default=0
            )
        )
        self.append(
            Field(
                "FSCPYS",
                "File Number of Copies",
                5,
                BCSN_PI,
                AnyRange(),
                Integer,
                default=0,
            )
        )
        self.append(
            Field("ENCRYP", "Encryption", 1, BCSN_PI, AnyRange(), Integer, default=0)
        )
        self.append(
            Field(
                "FBKGC",
                "File Background Color",
                3,
                None,
                AnyRange(),
                RGB,
                default=(0, 0, 0),
            )
        )
        self.append(
            Field(
                "ONAME",
                "Originator's Name",
                24,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "OPHONE",
                "Originator's Phone Number",
                18,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "FL", "File Length", 12, BCSN_PI, MinMax(388, 999_999_999_998), Integer
            )
        )
        self.append(
            Field(
                "HL",
                "NITF File Header Length",
                6,
                BCSN_PI,
                MinMax(388, 999_999),
                Integer,
            )
        )
        self.append(
            Field(
                "NUMI",
                "Number of Image Segments",
                3,
                BCSN_PI,
                AnyRange(),
                Integer,
                default=0,
                setter_callback=self._numi_handler,
            )
        )
        self.append(
            Field(
                "NUMS",
                "Number of Graphic Segments",
                3,
                BCSN_PI,
                AnyRange(),
                Integer,
                default=0,
                setter_callback=self._nums_handler,
            )
        )
        self.append(
            Field(
                "NUMX",
                "Reserved for Future Use",
                3,
                BCSN_PI,
                Constant(0),
                Integer,
                default=0,
            )
        )
        self.append(
            Field(
                "NUMT",
                "Number of Text Segments",
                3,
                BCSN_PI,
                AnyRange(),
                Integer,
                default=0,
                setter_callback=self._numt_handler,
            )
        )
        self.append(
            Field(
                "NUMDES",
                "Number of Data Extension Segments",
                3,
                BCSN_PI,
                AnyRange(),
                Integer,
                default=0,
                setter_callback=self._numdes_handler,
            )
        )
        self.append(
            Field(
                "NUMRES",
                "Number of Reserved Extension Segments",
                3,
                BCSN_PI,
                AnyRange(),
                Integer,
                default=0,
                setter_callback=self._numres_handler,
            )
        )
        self.append(
            Field(
                "UDHDL",
                "User Defined Header Data Length",
                5,
                BCSN_PI,
                AnyOf(Constant(0), MinMax(3, 10**5 - 1)),
                Integer,
                default=0,
                setter_callback=self._udhdl_handler,
            )
        )
        self.append(
            Field(
                "XHDL",
                "Extended Header Data Length",
                5,
                BCSN_PI,
                AnyOf(Constant(0), MinMax(3, 10**5 - 1)),
                Integer,
                default=0,
                setter_callback=self._xhdl_handler,
            )
        )

    def _numi_handler(self, field):
        """Handle NUMI value change"""
        self.remove_all("LISH\\d+")
        self.remove_all("LI\\d+")
        after = field
        for idx in range(1, field.value + 1):
            after = self.insert_after(
                after,
                Field(
                    f"LISH{idx:03}",
                    "Length of nth Image Subheader",
                    6,
                    BCSN_PI,
                    MinMax(439, 999_999),
                    Integer,
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    f"LI{idx:03}",
                    "Length of nth Image Segment",
                    10,
                    BCSN_PI,
                    MinMax(1, 10**10 - 1),
                    Integer,
                    setter_callback=self._lin_handler,
                ),
            )
        if self.numi_callback:
            self.numi_callback(field)

    def _lin_handler(self, field):
        """Handle LIN value change"""
        if self.lin_callback:
            self.lin_callback(field)

    def _nums_handler(self, field):
        self.remove_all("LSSH\\d+")
        self.remove_all("LS\\d+")
        after = field
        for idx in range(1, field.value + 1):
            after = self.insert_after(
                after,
                Field(
                    f"LSSH{idx:03}",
                    "Length of nth Graphic Subheader",
                    4,
                    BCSN_PI,
                    MinMax(258, 999_999),
                    Integer,
                    setter_callback=self._lsshn_handler,
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    f"LS{idx:03}",
                    "Length of nth Graphic Segment",
                    6,
                    BCSN_PI,
                    MinMax(1, 10**10 - 1),
                    Integer,
                    setter_callback=self._lsn_handler,
                ),
            )

        if self.nums_callback:
            self.nums_callback(field)

    def _lsshn_handler(self, field):
        if self.lsshn_callback:
            self.lsshn_callback(field)

    def _lsn_handler(self, field):
        if self.lsn_callback:
            self.lsn_callback(field)

    def _numt_handler(self, field):
        self.remove_all("LTSH\\d+")
        self.remove_all("LT\\d+")
        after = field
        for idx in range(1, field.value + 1):
            after = self.insert_after(
                after,
                Field(
                    f"LTSH{idx:03}",
                    "Length of nth Text Subheader",
                    4,
                    BCSN_PI,
                    MinMax(282, 999_999),
                    Integer,
                    setter_callback=self._ltshn_handler,
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    f"LT{idx:03}",
                    "Length of nth Text Segment",
                    5,
                    BCSN_PI,
                    MinMax(1, 99_999),
                    Integer,
                    setter_callback=self._ltn_handler,
                ),
            )

        if self.numt_callback:
            self.numt_callback(field)

    def _ltshn_handler(self, field):
        if self.ltshn_callback:
            self.ltshn_callback(field)

    def _ltn_handler(self, field):
        if self.ltn_callback:
            self.ltn_callback(field)

    def _numdes_handler(self, field):
        self.remove_all("LDSH\\d+")
        self.remove_all("LD\\d+")
        after = field
        for idx in range(1, field.value + 1):
            after = self.insert_after(
                after,
                Field(
                    f"LDSH{idx:03}",
                    "Length of nth Data Extension Segment Subheader",
                    4,
                    BCSN_PI,
                    MinMax(200, 999_999),
                    Integer,
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    f"LD{idx:03}",
                    "Length of nth Data Extension Segment",
                    9,
                    BCSN_PI,
                    MinMax(1, 10**9 - 1),
                    Integer,
                    setter_callback=self._ldn_handler,
                ),
            )

        if self.numdes_callback:
            self.numdes_callback(field)

    def _ldn_handler(self, field):
        if self.ldn_callback:
            self.ldn_callback(field)

    def _numres_handler(self, field):
        self.remove_all("LRESH\\d+")
        self.remove_all("LRE\\d+")
        after = field
        for idx in range(1, field.value + 1):
            after = self.insert_after(
                after,
                Field(
                    f"LRESH{idx:03}",
                    "Length of nth Reserved Extension Segment Subheader",
                    4,
                    BCSN_PI,
                    MinMax(200, 999_999),
                    Integer,
                    setter_callback=self._lreshn_handler,
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    f"LRE{idx:03}",
                    "Length of nth Reserved Extension Segment",
                    7,
                    BCSN_PI,
                    MinMax(1, 10**7 - 1),
                    Integer,
                    setter_callback=self._lren_handler,
                ),
            )

        if self.numres_callback:
            self.numres_callback(field)

    def _lreshn_handler(self, field):
        if self.lreshn_callback:
            self.lreshn_callback(field)

    def _lren_handler(self, field):
        if self.lren_callback:
            self.lren_callback(field)

    def _udhdl_handler(self, field):
        self.remove_all("UDHOFL")
        self.remove_all("UDHD")
        after = field
        if field.value:
            after = self.insert_after(
                after,
                Field(
                    "UDHOFL",
                    "User Defined Header Overflow",
                    3,
                    BCSN_PI,
                    AnyRange(),
                    Integer,
                    default=0,
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    "UDHD",
                    "User Defined Header Data",
                    field.value - 3,
                    None,
                    AnyRange(),
                    Bytes,
                ),
            )

    def _xhdl_handler(self, field):
        self.remove_all("XHDLOFL")
        self.remove_all("XHD")
        after = field
        if field.value:
            after = self.insert_after(
                after,
                Field(
                    "XHDLOFL",
                    "Extended Header Data Overflow",
                    3,
                    BCSN_PI,
                    AnyRange(),
                    Integer,
                    default=0,
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    "XHD",
                    "Extended Header Data",
                    field.value - 3,
                    None,
                    AnyRange(),
                    Bytes,
                ),
            )


# Table A-2
IMAGE_CATEGORIES = [
    "BARO",
    "BP",
    "CAT",
    "CP",
    "CURRENT",
    "DEPTH",
    "DTEM",
    "EO",
    "FL",
    "FP",
    "HR",
    "HS",
    "IR",
    "LEG",
    "LOCG",
    "MAP",
    "MATR",
    "MRI",
    "MS",
    "OP",
    "PAT",
    "RD",
    "SAR",
    "SARIQ",
    "SL",
    "TI",
    "VD",
    "VIS",
    "WIND",
    "XRAY",
]


class ImageSubHeader(Group):
    """
    Image SubHeader fields

    Args
    ----
    name: str
        Name to give this component

    Note
    ----
    See MIL-STD-2500C Table A-3

    """

    def __init__(self, name):
        super().__init__(name)

        self.append(
            Field(
                "IM",
                "File Part Type",
                2,
                BCSA,
                Constant("IM"),
                StringAscii,
                default="IM",
            )
        )
        self.append(
            Field("IID1", "Image Identifier 1", 10, BCSA, AnyRange(), StringAscii)
        )
        self.append(
            Field(
                "IDATIM", "Image Date and Time", 14, BCSN, DATETIME_REGEX, StringAscii
            )
        )
        self.append(
            Field(
                "TGTID",
                "Target Identifier",
                17,
                BCSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "IID2",
                "Image Identifier 2",
                80,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISCLAS",
                "Image Security Classification",
                1,
                ECSA,
                Enum(["T", "S", "C", "R", "U"]),
                StringISO8859_1,
            )
        )
        self.append(
            Field(
                "ISCLSY",
                "Image Security Classification System",
                2,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISCODE",
                "Image Codewords",
                11,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISCTLH",
                "Image Control and Handling",
                2,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISREL",
                "Image Release Instructions",
                20,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISDCTP",
                "Image Declassification Type",
                2,
                ECSA,
                Enum(["", "DD", "DE", "GD", "GE", "O", "X"]),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISDCDT",
                "Image Declassification Date",
                8,
                ECSA,
                AnyOf(Constant(""), DATE_REGEX),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISDCXM",
                "Image Declassification Exemption",
                4,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISDG",
                "Image Downgrade",
                1,
                ECSA,
                Enum(["", "S", "C", "R"]),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISDGDT",
                "Image Downgrade Date",
                8,
                ECSA,
                AnyOf(Constant(""), DATE_REGEX),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISCLTX",
                "Image Classification Text",
                43,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISCATP",
                "Image Classification Authority Type",
                1,
                ECSA,
                Enum(["", "O", "D", "M"]),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISCAUT",
                "Image Classification Authority",
                40,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISCRSN",
                "Image Classification Reason",
                1,
                ECSA,
                AnyOf(Constant(""), Regex("[A-G]")),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISSRDT",
                "Image Security Source Date",
                8,
                ECSA,
                AnyOf(Constant(""), DATE_REGEX),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "ISCTLN",
                "Image Security Control Number",
                15,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field("ENCRYP", "Encryption", 1, BCSN_PI, AnyRange(), Integer, default=0)
        )
        self.append(
            Field(
                "ISORCE",
                "Image Source",
                42,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "NROWS",
                "Number of Significant Rows in Image",
                8,
                BCSN_PI,
                MinMax(1, None),
                Integer,
            )
        )
        self.append(
            Field(
                "NCOLS",
                "Number of Significant Columns in Image",
                8,
                BCSN_PI,
                MinMax(1, None),
                Integer,
            )
        )
        self.append(
            Field(
                "PVTYPE",
                "Pixel Value Type",
                3,
                BCSA,
                Enum(["INT", "B", "SI", "R", "C"]),
                StringAscii,
            )
        )
        self.append(
            Field(
                "IREP",
                "Image Representation",
                8,
                BCSA,
                Enum(
                    [
                        "MONO",
                        "RGB",
                        "RGB/LUT",
                        "MULTI",
                        "NODISPLY",
                        "NVECTOR",
                        "POLAR",
                        "VPH",
                        "YCbCr601",
                    ]
                ),
                StringAscii,
            )
        )
        self.append(
            Field(
                "ICAT",
                "Image Category",
                8,
                BCSA,
                Enum(IMAGE_CATEGORIES),
                StringAscii,
                default="VIS",
            )
        )
        self.append(
            Field(
                "ABPP",
                "Actual Bits-Per-Pixel Per Band",
                2,
                BCSN_PI,
                MinMax(1, 96),
                Integer,
            )
        )
        self.append(
            Field(
                "PJUST",
                "Pixel Justification",
                1,
                BCSA,
                Enum(["L", "R"]),
                StringAscii,
                default="R",
            )
        )
        self.append(
            Field(
                "ICORDS",
                "Image Coordinate Representation",
                1,
                BCSA,
                Enum(["", "U", "G", "N", "S", "D"]),
                StringAscii,
                default="",
                setter_callback=self._icords_handler,
            )
        )
        # IGEOLO
        self.append(
            Field(
                "NICOM",
                "Number of Image Comments",
                1,
                BCSN_PI,
                AnyRange(),
                Integer,
                default=0,
                setter_callback=self._nicom_handler,
            )
        )
        # ICOMn
        self.append(
            Field(
                "IC",
                "Image Compression",
                2,
                BCSA,
                Enum(
                    [
                        "NC",
                        "NM",
                        "C1",
                        "C3",
                        "C4",
                        "C5",
                        "C6",
                        "C7",
                        "C8",
                        "I1",
                        "M1",
                        "M3",
                        "M4",
                        "M5",
                        "M6",
                        "M7",
                        "M8",
                    ]
                ),
                StringAscii,
                setter_callback=self._ic_handler,
            )
        )
        # COMRAT
        self.append(
            Field(
                "NBANDS",
                "Number of Bands",
                1,
                BCSN_PI,
                AnyRange(),
                Integer,
                setter_callback=self._nbands_handler,
            )
        )
        # XBANDS
        # IREPBANDn
        # ISUBCATn
        # IFCn
        # IMFLTn
        # NLUTSn
        # NELUTn
        # LUTDn
        self.append(
            Field(
                "ISYNC", "Image Sync Code", 1, BCSN_PI, Constant(0), Integer, default=0
            )
        )
        self.append(
            Field(
                "IMODE", "Image Mode", 1, BCSA, Enum(["B", "P", "R", "S"]), StringAscii
            )
        )
        self.append(
            Field(
                "NBPR", "Number of Blocks Per Row", 4, BCSN_PI, MinMax(1, None), Integer
            )
        )
        self.append(
            Field(
                "NBPC",
                "Number of Blocks Per Column",
                4,
                BCSN_PI,
                MinMax(1, None),
                Integer,
            )
        )
        self.append(
            Field(
                "NPPBH",
                "Number of Pixels Per Block Horizontal",
                4,
                BCSN_PI,
                MinMax(0, 8192),
                Integer,
            )
        )
        self.append(
            Field(
                "NPPBV",
                "Number of Pixels Per Block Vertical",
                4,
                BCSN_PI,
                MinMax(0, 8192),
                Integer,
            )
        )
        self.append(
            Field(
                "NBPP",
                "Number of Bits Per Pixel Per Band",
                2,
                BCSN_PI,
                MinMax(1, 96),
                Integer,
            )
        )
        self.append(
            Field(
                "IDLVL",
                "Image Display Level",
                3,
                BCSN_PI,
                MinMax(1, None),
                Integer,
                default=1,
            )
        )
        self.append(
            Field(
                "IALVL",
                "Attachment Level",
                3,
                BCSN_PI,
                MinMax(0, 998),
                Integer,
                default=0,
            )
        )
        self.append(
            Field(
                "ILOC",
                "Image Location",
                10,
                BCSN,
                AnyRange(),
                int_pair(5),
                default=(0, 0),
            )
        )
        self.append(
            Field(
                "IMAG",
                "Image Magnification",
                4,
                BCSA,
                Regex(r"(\d+\.?\d*)|(\d*\.?\d+)|(\/\d+)"),
                StringAscii,
                default="1.0 ",
            )
        )
        self.append(
            Field(
                "UDIDL",
                "User Defined Image Data Length",
                5,
                BCSN_PI,
                AnyOf(Constant(0), MinMax(3, None)),
                Integer,
                default=0,
                setter_callback=self._udidl_handler,
            )
        )
        self.append(
            Field(
                "IXSHDL",
                "Image Extended Subheader Data Length",
                5,
                BCSN_PI,
                AnyOf(Constant(0), MinMax(3, None)),
                Integer,
                default=0,
                setter_callback=self._ixshdl_handler,
            )
        )

    def _icords_handler(self, field):
        self.remove_all("IGEOLO")
        if field.value:
            self.insert_after(
                field,
                Field(
                    "IGEOLO",
                    "Image Geographic Location",
                    60,
                    BCSA,
                    AnyRange(),
                    StringAscii,
                    default="",
                ),
            )

    def _nicom_handler(self, field):
        self.remove_all("ICOM\\d+")
        after = self["NICOM"]
        for idx in range(1, field.value + 1):
            after = self.insert_after(
                after,
                Field(
                    f"ICOM{idx}",
                    "Image Comment {n}",
                    80,
                    ECSA,
                    AnyRange(),
                    StringISO8859_1,
                ),
            )

    def _ic_handler(self, field):
        self.remove_all("COMRAT")
        if field.value not in ("NC", "NM"):
            self.insert_after(
                self["IC"],
                Field(
                    "COMRAT", "Compression Rate Code", 4, BCSA, AnyRange(), StringAscii
                ),
            )

    def _nbands_handler(self, field):
        self.remove_all("XBANDS")
        if field.value == 0:
            self.insert_after(
                self["NBANDS"],
                Field(
                    "XBANDS",
                    "Number of Multispectral Bands",
                    5,
                    BCSN_PI,
                    MinMax(10, None),
                    Integer,
                    setter_callback=self._xbands_handler,
                ),
            )
        self._set_num_band_groups(field.value)

    def _xbands_handler(self, field):
        self._set_num_band_groups(field.value)

    def _set_num_band_groups(self, count):
        self.remove_all("IREPBAND\\d+")
        self.remove_all("ISUBCAT\\d+")
        self.remove_all("IFC\\d+")
        self.remove_all("IMFLT\\d+")
        self.remove_all("NLUTS\\d+")
        self.remove_all("NELUT\\d+")
        self.remove_all("LUTD\\d+")

        after = self.get("XBANDS", self["NBANDS"])
        for idx in range(1, count + 1):
            after = self.insert_after(
                after,
                Field(
                    f"IREPBAND{idx:05d}",
                    "nth Band Representation",
                    2,
                    BCSA,
                    AnyRange(),
                    StringAscii,
                    default="",
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    f"ISUBCAT{idx:05d}",
                    "nth Band Subcategory",
                    6,
                    BCSA,
                    AnyRange(),
                    StringAscii,
                    default="",
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    f"IFC{idx:05d}",
                    "nth Band Image Filter Condition",
                    1,
                    BCSA,
                    AnyRange(),
                    StringAscii,
                    default="N",
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    f"IMFLT{idx:05d}",
                    "nth Band Standard Image Filter Code",
                    3,
                    BCSA,
                    AnyRange(),
                    StringAscii,
                    default="",
                ),
            )
            after = self.insert_after(
                after,
                Field(
                    f"NLUTS{idx:05d}",
                    "Number of LUTS for the nth Image Band",
                    1,
                    BCSN_PI,
                    MinMax(0, 4),
                    Integer,
                    default=0,
                    setter_callback=self._nluts_handler,
                ),
            )

    def _udidl_handler(self, field):
        self.remove_all("UDOFL")
        self.remove_all("UDID")
        if field.value > 0:
            after = self.insert_after(
                field,
                Field(
                    "UDOFL", "User Defined Overflow", 3, BCSN_PI, AnyRange(), Integer
                ),
            )
            self.insert_after(
                after,
                Field(
                    "UDID",
                    "User Defined Image Data",
                    field.value - 3,
                    None,
                    AnyRange(),
                    Bytes,
                ),
            )

    def _ixshdl_handler(self, field):
        self.remove_all("IXSOFL")
        self.remove_all("IXSHD")
        if field.value > 0:
            after = self.insert_after(
                field,
                Field(
                    "IXSOFL",
                    "Image Extended Subheader Overflow",
                    3,
                    BCSN_PI,
                    AnyRange(),
                    Integer,
                ),
            )
            self.insert_after(
                after,
                Field(
                    "IXSHD",
                    "Image Extended Subheader Data",
                    field.value - 3,
                    None,
                    AnyRange(),
                    Bytes,
                ),
            )

    def _nluts_handler(self, field):
        idx = int(field.name.removeprefix("NLUTS"))
        self.remove_all(f"NELUT{idx:05d}\\d+")
        self.remove_all(f"LUTD{idx:05d}\\d+")
        if field.value > 0:
            after = self.insert_after(
                field,
                Field(
                    f"NELUT{idx:05d}",
                    "Number of LUT Entries for the nth Image Band",
                    5,
                    BCSN_PI,
                    MinMax(1, 65536),
                    Integer,
                    setter_callback=self._nelut_handler,
                ),
            )
            for lutidx in range(1, field.value + 1):
                after = self.insert_after(
                    after,
                    Field(
                        f"LUTD{idx:05d}{lutidx}",
                        "nth Image Band, mth LUT",
                        None,
                        None,
                        AnyRange(),
                        Bytes,
                    ),
                )

    def _nelut_handler(self, field):
        idx = int(field.name.removeprefix("NELUT"))
        for lutd in self.find_all(f"LUTD{idx:05d}\\d+"):
            lutd.size = field.value


class ImageSegment(Group):
    def __init__(self, name, data_size):
        super().__init__(name)
        self.append(ImageSubHeader("SubHeader"))
        self.append(BinaryPlaceholder("Data", data_size))

    def print(self):
        print(f"# ImageSegment {self.name}")
        super().print()


class GraphicSegment(Group):
    def __init__(self, name, subheader_size, data_size):
        super().__init__(name)
        self.append(
            Field("SubHeader", "Placeholder", subheader_size, None, AnyRange(), Bytes)
        )
        self.append(BinaryPlaceholder("Data", data_size))

    def print(self):
        print(f"# GraphicSegment {self.name}")
        super().print()


class TextSegment(Group):
    def __init__(self, name, subheader_size, data_size):
        super().__init__(name)
        self.append(
            Field("SubHeader", "Placeholder", subheader_size, None, AnyRange(), Bytes)
        )
        self.append(BinaryPlaceholder("Data", data_size))

    def print(self):
        print(f"# TextSegment {self.name}")
        super().print()


class RESegment(Group):
    def __init__(self, name, subheader_size, data_size):
        super().__init__(name)
        self.append(
            Field("SubHeader", "Placeholder", subheader_size, None, AnyRange(), Bytes)
        )
        self.append(BinaryPlaceholder("Data", data_size))

    def print(self):
        print(f"# RESegment {self.name}")
        super().print()


class DESubHeader(Group):
    def __init__(self, name):
        super().__init__(name)
        self.append(
            Field(
                "DE",
                "File Part Type",
                2,
                BCSA,
                Constant("DE"),
                StringAscii,
                default="DE",
            )
        )
        self.append(
            Field(
                "DESID",
                "Unique DES Type Identifier",
                25,
                BCSA,
                AnyRange(),
                StringAscii,
                setter_callback=self._desid_handler,
            )
        )
        self.append(
            Field(
                "DESVER",
                "Version of the Data Definition",
                2,
                BCSN_PI,
                MinMax(1, None),
                Integer,
            )
        )
        self.append(
            Field(
                "DESCLAS",
                "DES Security Classification",
                1,
                ECSA,
                Enum(["T", "S", "C", "R", "U"]),
                StringISO8859_1,
            )
        )
        self.append(
            Field(
                "DESCLSY",
                "DES Security Classification System",
                2,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESCODE",
                "DES Codewords",
                11,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESCTLH",
                "DES Control and Handling",
                2,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESREL",
                "DES Release Instructions",
                20,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESDCTP",
                "DES Declassification Type",
                2,
                ECSA,
                Enum(["", "DD", "DE", "GD", "GE", "O", "X"]),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESDCDT",
                "DES Declassification Date",
                8,
                ECSA,
                AnyOf(Constant(""), DATE_REGEX),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESDCXM",
                "DES Declassification Exemption",
                4,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESDG",
                "DES Downgrade",
                1,
                ECSA,
                Enum(["", "S", "C", "R"]),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESDGDT",
                "DES Downgrade Date",
                8,
                ECSA,
                AnyOf(Constant(""), DATE_REGEX),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESCLTX",
                "DES Classification Text",
                43,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESCATP",
                "DES Classification Authority Type",
                1,
                ECSA,
                Enum(["", "O", "D", "M"]),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESCAUT",
                "DES Classification Authority",
                40,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESCRSN",
                "DES Classification Reason",
                1,
                ECSA,
                AnyOf(Constant(""), Regex("[A-G]")),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESSRDT",
                "DES Security Source Date",
                8,
                ECSA,
                AnyOf(Constant(""), DATE_REGEX),
                StringISO8859_1,
                default="",
            )
        )
        self.append(
            Field(
                "DESCTLN",
                "DES Security Control Number",
                15,
                ECSA,
                AnyRange(),
                StringISO8859_1,
                default="",
            )
        )
        # DESOFLW
        # DESITEM
        self.append(
            Field(
                "DESSHL",
                "DES User-defined Subheader Length",
                4,
                BCSN_PI,
                AnyRange(),
                Integer,
                setter_callback=self._desshl_handler,
            )
        )
        # DESSHF

    def _desid_handler(self, field):
        self.remove_all("DESOFLW")
        self.remove_all("DESITEM")
        if field.value == "TRE_OVERFLOW":
            after = self.insert_after(
                self["DESCTLN"],
                Field(
                    "DESOFLW",
                    "DES Overflowed Header Type",
                    6,
                    BCSA,
                    Enum(["XHD", "IXSHD", "SXSHD", "TXSHD", "UDHD", "UDID"]),
                    StringAscii,
                ),
            )
            self.insert_after(
                after,
                Field(
                    "DESITEM",
                    "DES Data Item Overflowed",
                    3,
                    BCSN_PI,
                    AnyRange(),
                    Integer,
                ),
            )

    def _desshl_handler(self, field):
        self.remove_all("DESSHF")
        self.insert_after(field, DESSHF_Factory(self["DESID"], self["DESVER"], field))


class DESegment(Group):
    def __init__(self, name, data_size):
        super().__init__(name)
        self.append(DESubHeader("SubHeader"))
        self.append(BinaryPlaceholder("DESDATA", data_size))

    def print(self):
        print(f"# DESegment {self.name}")
        super().print()


class XmlDataContentSubheader(Group):
    def __init__(self, name, size):
        super().__init__(name)
        self.all_fields = [
            Field(
                "DESCRC",
                "Cyclic Redundancy Check",
                5,
                BCSN_PI,
                AnyOf(MinMax(0, 65535), Constant(99999)),
                Integer,
                default=0,
            ),
            Field("DESSHFT", "XML File Type", 8, BCSA, AnyRange(), StringAscii),
            Field("DESSHDT", "Date and Time", 20, BCSA, AnyRange(), StringAscii),
            Field(
                "DESSHRP",
                "Responsible Party",
                40,
                U8,
                AnyRange(),
                StringUtf8,
                default="",
            ),
            Field(
                "DESSHSI",
                "Specification Identifier",
                60,
                U8,
                AnyRange(),
                StringUtf8,
                default="",
            ),
            Field(
                "DESSHSV",
                "Specification Version",
                10,
                BCSA,
                AnyRange(),
                StringAscii,
                default="",
            ),
            Field("DESSHSD", "Specification Date", 20, BCSA, AnyRange(), StringAscii),
            Field(
                "DESSHTN",
                "Target Namespace",
                120,
                BCSA,
                AnyRange(),
                StringAscii,
                default="",
            ),
            Field(
                "DESSHLPG",
                "Location - Polygon",
                125,
                BCSA,
                AnyRange(),
                StringAscii,
                default="",
            ),
            Field(
                "DESSHLPT",
                "Location - Point",
                25,
                BCSA,
                AnyRange(),
                StringAscii,
                default="",
            ),
            Field(
                "DESSHLI",
                "Location - Identifier",
                20,
                BCSA,
                AnyRange(),
                StringAscii,
                default="",
            ),
            Field(
                "DESSHLIN",
                "Location Identifier Namespace URI",
                120,
                BCSA,
                AnyRange(),
                StringAscii,
                default="",
            ),
            Field("DESSHABS", "Abstract", 200, U8, AnyRange(), StringUtf8, default=""),
        ]
        allowed_sizes = {0, 5, 283, 773}
        if size not in allowed_sizes:
            logger.warning(
                f"Invalid user defined subheader length. {size} not in {allowed_sizes}"
            )

        current_size = 0
        for field in self.all_fields:
            if current_size == size:
                break
            elif current_size < size:
                self.append(field)
                current_size += field.size
            elif current_size > size:
                raise ValueError(f"Invalid XML_DATA_CONTENT header {size=}")


def DESSHF_Factory(desid_field, desver_field, desshl_field):  # noqa: N802
    """Create the DESSHF field based on the DES type

    Args
    ----
    desid_field: Field
        the DES SubHeader's DESID field
    desver_field: Field
        the DES SubHeader's DESVER field
    desshl_field: Field
        the DES SubHeader's DESSHL field

    Returns
    -------
    NitfIOComponent
        Either a XmlDataContentSubheader or a Field
    """
    if (desid_field.value, desver_field.value) == ("XML_DATA_CONTENT", 1):
        return XmlDataContentSubheader("DESSHF", desshl_field.value)
    return Field(
        "DESSHF",
        "DES User-defined Subheader Fields",
        desshl_field.value,
        BCSA,
        AnyRange(),
        StringAscii,
    )


class Nitf(Group):
    """Class representing an entire NITF

    Contains the following keys:
    * FileHeader
    * ImageSegments
    * GraphicSegments
    * TextSegments
    * DESegments
    * RESegments

    """

    def __init__(self):
        super().__init__("Root")
        self.append(
            FileHeader(
                "FileHeader",
                numi_callback=self._numi_handler,
                lin_callback=self._lin_handler,
                nums_callback=self._nums_handler,
                lsshn_callback=self._lsshn_handler,
                lsn_callback=self._lsn_handler,
                numt_callback=self._numt_handler,
                ltshn_callback=self._ltshn_handler,
                ltn_callback=self._ltn_handler,
                numdes_callback=self._numdes_handler,
                ldn_callback=self._ldn_handler,
                numres_callback=self._numres_handler,
                lreshn_callback=self._lreshn_handler,
                lren_callback=self._lren_handler,
            )
        )
        self.append(
            SegmentList(
                "ImageSegments",
                lambda n: ImageSegment(n, None),
                maximum=999,
            )
        )
        self.append(
            SegmentList(
                "GraphicSegments",
                lambda n: GraphicSegment(n, None, None),
                maximum=999,
            )
        )
        self.append(
            SegmentList(
                "TextSegments",
                lambda n: TextSegment(n, None, None),
                maximum=999,
            )
        )
        self.append(
            SegmentList(
                "DESegments",
                lambda n: DESegment(n, None),
                maximum=999,
            )
        )
        self.append(
            SegmentList(
                "RESegments",
                lambda n: RESegment(n, None, None),
                maximum=999,
            )
        )

    def _numi_handler(self, field):
        self["ImageSegments"].set_size(field.value)

    def _lin_handler(self, field):
        idx = int(field.name.removeprefix("LI")) - 1
        self["ImageSegments"][idx]["Data"].size = field.value

    def _nums_handler(self, field):
        self["GraphicSegments"].set_size(field.value)

    def _lsshn_handler(self, field):
        # this callback should be removed if the Graphic Subheader is implemented
        idx = int(field.name.removeprefix("LSSH")) - 1
        self["GraphicSegments"][idx]["SubHeader"].size = field.value

    def _lsn_handler(self, field):
        idx = int(field.name.removeprefix("LS")) - 1
        self["GraphicSegments"][idx]["Data"].size = field.value

    def _numt_handler(self, field):
        self["TextSegments"].set_size(field.value)

    def _ltshn_handler(self, field):
        # this callback should be removed if the Text Subheader is implemented
        idx = int(field.name.removeprefix("LTSH")) - 1
        self["TextSegments"][idx]["SubHeader"].size = field.value

    def _ltn_handler(self, field):
        idx = int(field.name.removeprefix("LT")) - 1
        self["TextSegments"][idx]["Data"].size = field.value

    def _numdes_handler(self, field):
        self["DESegments"].set_size(field.value)

    def _ldn_handler(self, field):
        idx = int(field.name.removeprefix("LD")) - 1
        self["DESegments"][idx]["DESDATA"].size = field.value

    def _numres_handler(self, field):
        self["RESegments"].set_size(field.value)

    def _lreshn_handler(self, field):
        # this callback should be removed if the Reserved Subheader is implemented
        idx = int(field.name.removeprefix("LRESH")) - 1
        self["RESegments"][idx]["SubHeader"].size = field.value

    def _lren_handler(self, field):
        idx = int(field.name.removeprefix("LRE")) - 1
        self["RESegments"][idx]["Data"].size = field.value

    def update_lengths(self):
        """Compute and set the segment lengths"""
        self["FileHeader"]["FL"].value = self.length()
        self["FileHeader"]["HL"].value = self["FileHeader"].length()

        for idx, seg in enumerate(self["ImageSegments"]):
            self["FileHeader"][f"LISH{idx + 1:03d}"].value = seg["SubHeader"].length()
            self["FileHeader"][f"LI{idx + 1:03d}"].value = seg["Data"].size

        for idx, seg in enumerate(self["GraphicSegments"]):
            self["FileHeader"][f"LSSH{idx + 1:03d}"].value = seg["SubHeader"].length()
            self["FileHeader"][f"LS{idx + 1:03d}"].value = seg["Data"].size

        for idx, seg in enumerate(self["TextSegments"]):
            self["FileHeader"][f"LTSH{idx + 1:03d}"].value = seg["SubHeader"].length()
            self["FileHeader"][f"LT{idx + 1:03d}"].value = seg["Data"].size

        for idx, seg in enumerate(self["DESegments"]):
            self["FileHeader"][f"LDSH{idx + 1:03d}"].value = seg["SubHeader"].length()
            self["FileHeader"][f"LD{idx + 1:03d}"].value = seg["DESDATA"].size

        for idx, seg in enumerate(self["RESegments"]):
            self["FileHeader"][f"LRESH{idx + 1:03d}"].value = seg["SubHeader"].length()
            self["FileHeader"][f"LRE{idx + 1:03d}"].value = seg["Data"].size

    def update_fdt(self):
        """Set the FDT field to the current time"""
        now = datetime.datetime.now(datetime.timezone.utc)
        self["FileHeader"]["FDT"].value = now.strftime("%Y%m%d%H%M%S")

    def finalize(self):
        """Compute derived values such as lengths, and CLEVEL"""
        self.update_lengths()
        self.update_fdt()
        self.update_clevel()  # must be after lengths

    def _clevel_ccs_extent(self):
        min_ccs = (0, 0)
        max_ccs = (0, 0)

        level_origin = {0: np.array([0, 0])}
        for imseg in self["ImageSegments"]:
            alvl = imseg["SubHeader"]["IALVL"].value
            dlvl = imseg["SubHeader"]["IDLVL"].value
            loc = imseg["SubHeader"]["ILOC"].value
            size = np.array(
                [imseg["SubHeader"]["NROWS"].value, imseg["SubHeader"]["NCOLS"].value]
            )
            level_origin[dlvl] = level_origin[alvl] + loc

            min_ccs = np.minimum(min_ccs, level_origin[dlvl])
            max_ccs = np.maximum(max_ccs, level_origin[dlvl] + size)

        if len(self["GraphicSegments"]):
            logger.warning("CLEVEL of NITFs with Graphic Segments is not supported")

        max_extent = max(np.asarray(max_ccs) - min_ccs)
        if max_extent <= 2047:
            return 3
        if max_extent <= 8191:
            return 5
        if max_extent <= 65535:
            return 6
        if max_extent <= 99_999_999:
            return 7
        return 9

    def _clevel_file_size(self):
        if self["FileHeader"]["FL"].value < 50 * (1 << 20):
            return 3
        if self["FileHeader"]["FL"].value < 1 * (1 << 30):
            return 5
        if self["FileHeader"]["FL"].value < 2 * (1 << 30):
            return 6
        if self["FileHeader"]["FL"].value < 10 * (1 << 30):
            return 7
        return 9

    def _clevel_image_size(self):
        clevel = 3
        for imseg in self["ImageSegments"]:
            nrows = imseg["SubHeader"]["NROWS"].value
            ncols = imseg["SubHeader"]["NCOLS"].value

            if nrows <= 2048 and ncols <= 2048:
                clevel = max(clevel, 3)
            elif nrows <= 8192 and ncols <= 8192:
                clevel = max(clevel, 5)
            elif nrows <= 65536 and ncols <= 65536:
                clevel = max(clevel, 6)
            elif nrows <= 99_999_999 and ncols <= 99_999_999:
                clevel = max(clevel, 7)
        return clevel

    def _clevel_image_blocking(self):
        clevel = 3
        for imseg in self["ImageSegments"]:
            horiz = imseg["SubHeader"]["NPPBH"].value
            vert = imseg["SubHeader"]["NPPBV"].value

            if horiz <= 2048 and vert <= 2048:
                clevel = max(clevel, 3)
            elif horiz <= 8192 and vert <= 8192:
                clevel = max(clevel, 5)
        return clevel

    def _clevel_irep(self):
        clevel = 0
        for imseg in self["ImageSegments"]:
            has_lut = bool(imseg["SubHeader"].find_all("NLUT.*"))
            num_bands = (
                imseg["SubHeader"].get("XBANDS", imseg["SubHeader"]["NBANDS"]).value
            )
            # Color (RGB) No Compression
            if (
                imseg["SubHeader"]["IREP"].value == "RGB"
                and num_bands == 3
                and not has_lut
                and imseg["SubHeader"]["IC"].value in ("NC", "NM")
                and imseg["SubHeader"]["IMODE"].value in ("B", "P", "R", "S")
            ):
                if imseg["SubHeader"]["NBPP"].value == 8:
                    clevel = max(clevel, 3)

                if imseg["SubHeader"]["NBPP"].value in (8, 16, 32):
                    clevel = max(clevel, 6)

            # Multiband (MULTI) No Compression
            if (
                imseg["SubHeader"]["IREP"].value == "MULTI"
                and imseg["SubHeader"]["NBPP"].value in (1, 8, 16, 32, 64)
                and imseg["SubHeader"]["IC"].value in ("NC", "NM")
                and imseg["SubHeader"]["IMODE"].value in ("B", "P", "R", "S")
            ):
                if 2 <= num_bands <= 9:
                    clevel = max(clevel, 3)

                if 10 <= num_bands <= 255:
                    clevel = max(clevel, 5)

                if 255 <= num_bands <= 999:
                    clevel = max(clevel, 7)

            # JPEG2000 Compression Multiband (MULTI)
            if (
                imseg["SubHeader"]["IREP"].value == "MULTI"
                and imseg["SubHeader"]["NBPP"].value <= 32
                and imseg["SubHeader"]["IC"].value in ("C8", "M8")
                and imseg["SubHeader"]["IMODE"].value == "B"
            ):
                if 1 <= num_bands <= 9:
                    clevel = max(clevel, 3)

                if 10 <= num_bands <= 255:
                    clevel = max(clevel, 5)

                if 256 <= num_bands <= 999:
                    clevel = max(clevel, 7)

            # Multiband (MULTI) Individual Band JPEG Compression
            if (
                imseg["SubHeader"]["IREP"].value == "MULTI"
                and imseg["SubHeader"]["NBPP"].value in (8, 12)
                and not has_lut
                and imseg["SubHeader"]["IC"].value in ("C3", "M3")
                and imseg["SubHeader"]["IMODE"].value in ("B", "S")
            ):
                if 2 <= num_bands <= 9:
                    clevel = max(clevel, 3)

                if 10 <= num_bands <= 255:
                    clevel = max(clevel, 5)

                if 256 <= num_bands <= 999:
                    clevel = max(clevel, 7)

            # Multiband (MULTI) Multi-Component Compression
            if (
                imseg["SubHeader"]["IREP"].value == "MULTI"
                and imseg["SubHeader"]["NBPP"].value in (8, 12)
                and not has_lut
                and imseg["SubHeader"]["IC"].value in ("C6", "M6")
                and imseg["SubHeader"]["IMODE"].value in ("B", "P", "S")
            ):
                if 2 <= num_bands <= 9:
                    clevel = max(clevel, 3)

                if 10 <= num_bands <= 255:
                    clevel = max(clevel, 5)

                if 256 <= num_bands <= 999:
                    clevel = max(clevel, 7)

            # Matrix Data (NODISPLY)
            if (
                imseg["SubHeader"]["IREP"].value == "NODISPLY"
                and imseg["SubHeader"]["NBPP"].value in (8, 16, 32, 64)
                and not has_lut
                and imseg["SubHeader"]["IMODE"].value in ("B", "P", "R", "S")
            ):
                if 2 <= num_bands <= 9:
                    clevel = max(clevel, 3)

                if 10 <= num_bands <= 255:
                    clevel = max(clevel, 5)

                if 256 <= num_bands <= 999:
                    clevel = max(clevel, 7)

        return clevel

    def _clevel_num_imseg(self):
        if len(self["ImageSegments"]) <= 20:
            return 3
        if 20 < len(self["ImageSegments"]) <= 100:
            return 5
        return 9

    def _clevel_aggregate_size_of_graphic_segments(self):
        size = 0
        for field in self["FileHeader"].find_all("LS\\d+"):
            size += field.value

        if size <= 1 * (1 << 20):
            return 3
        if size <= 2 * (1 << 20):
            return 5
        return 9

    def _clevel_cl9(self):
        """Explicit CLEVEL 9 checks"""
        # 1
        if self["FileHeader"]["FL"].value >= 10 * (1 << 30):
            return 9

        total_num_bands = 0
        for imseg in self["ImageSegments"]:
            # 2
            if (
                imseg["SubHeader"]["NPPBH"].value == 0
                or imseg["SubHeader"]["NPPBV"].value == 0
            ):
                return 9
            total_num_bands += imseg.get("XBANDS", imseg["SubHeader"]["NBANDS"]).value

        # 3
        if total_num_bands > 999:
            return 9

        # 4
        if len(self["ImageSegments"]) > 100:
            return 9

        # 5
        if len(self["GraphicSegments"]) > 100:
            return 9

        # 6
        size = 0
        for field in self["FileHeader"].find_all("LS\\d+"):
            size += field.value
        if size > 2 * (1 << 20):
            return 9

        # 7
        if len(self["TextSegments"]) > 32:
            return 9

        # 8
        if len(self["DESegments"]) > 100:
            return 9

        return 0

    def update_clevel(self):
        """Compute and update the CELVEL field.  See MIL-STD-2500C Table A-10"""
        clevel = 3
        helpers = [attrib for attrib in dir(self) if attrib.startswith("_clevel_")]
        for helper in helpers:
            clevel = max(clevel, getattr(self, helper)())

        self["FileHeader"]["CLEVEL"].value = clevel


def main(args=None):
    parser = argparse.ArgumentParser(description="Display NITF Header content")
    parser.add_argument("filename", type=pathlib.Path, help="Path to NITF file")
    config = parser.parse_args(args)

    ntf = Nitf()
    with config.filename.open("rb") as fd:
        ntf.load(fd)

    ntf.print()


if __name__ == "__main__":
    main()
