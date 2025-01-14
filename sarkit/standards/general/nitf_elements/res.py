"""The reserved extension subheader definitions."""

from .base import (
    BaseNITFElement,
    NITFElement,
    Unstructured,
    _IntegerDescriptor,
    _NITFElementDescriptor,
    _parse_nitf_element,
    _StringDescriptor,
    _StringEnumDescriptor,
)
from .security import NITFSecurityTags

# To help link the code to the NITF Standard, many variable and property names
# are intended to closely match the names used in the document.  As a result they
# may not adhere to the PEP8 convention used elsewhere in the code.


class RESUserHeader(Unstructured):
    _size_len = 4


class ReservedExtensionHeader(NITFElement):
    """
    The reserved extension subheader - see standards document Joint BIIF Profile (JBP) for more
    information.
    """

    _ordering = ("RE", "RESID", "RESVER", "Security", "UserHeader")
    _lengths = {"RE": 2, "RESID": 25, "RESVER": 2}
    RE = _StringEnumDescriptor(
        "RE",
        True,
        2,
        {
            "RE",
        },
        default_value="RE",
        docstring="File part type.",
    )  # type: str
    RESID = _StringDescriptor(
        "RESID",
        True,
        25,
        default_value="",
        docstring="Unique RES Type Identifier. This field shall contain a valid alphanumeric "
        "identifier properly registered with the ISMC.",
    )  # type: str
    RESVER = _IntegerDescriptor(
        "RESVER",
        True,
        2,
        default_value=1,
        docstring="Version of the Data Definition. This field shall contain the alphanumeric version "
        "number of the use of the tag. The version number is assigned as part of the "
        "registration process.",
    )  # type: int
    Security = _NITFElementDescriptor(
        "Security",
        True,
        NITFSecurityTags,
        default_args={},
        docstring="The security tags.",
    )  # type: NITFSecurityTags

    def __init__(self, **kwargs):
        self._RESID = None
        self._UserHeader = None
        super(ReservedExtensionHeader, self).__init__(**kwargs)

    @property
    def UserHeader(self) -> RESUserHeader:  # noqa N802
        """
        RESUserHeader: The RES user header.
        """

        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):  # noqa N802
        if not isinstance(value, BaseNITFElement):
            value = _parse_nitf_element(value, RESUserHeader, {}, "UserHeader", self)
        self._UserHeader = value
        self._load_header_data()

    def _load_header_data(self):
        """
        Load any user defined header specifics.

        Returns
        -------
        None
        """

        pass

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == "UserHeader":
            val = RESUserHeader.from_bytes(value, start)
            fields["UserHeader"] = val
            return start + val.get_bytes_length()
        else:
            return super(ReservedExtensionHeader, cls)._parse_attribute(
                fields, attribute, value, start
            )
