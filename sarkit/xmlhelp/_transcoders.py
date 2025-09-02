"""
Common XML Helper functionality
"""

import abc
import datetime
import inspect
from collections.abc import Iterator, Sequence
from typing import Any

import lxml.etree
import numpy as np
import numpy.typing as npt


def inheritdocstring(cls):
    """Decorator for inheriting docstring from parent class(es)"""
    for base in inspect.getmro(cls):
        if base.__doc__ is not None:
            cls.__doc__ = base.__doc__
            break
    return cls


class Type:
    """Base class for transcoders which provide methods for parsing, setting, and making XML elements."""

    def parse_elem(self, elem):
        """Returns the XML element ``elem``'s text."""
        return elem.text

    def set_elem(self, elem, val):
        """Set ``elem.text`` to the string version of ``val``."""
        elem.text = str(val)

    def make_elem(self, tag, val):
        """Create a new XML element."""
        elem = lxml.etree.Element(tag)
        self.set_elem(elem, val)
        return elem


class TxtType(Type):
    """
    Transcoder for text (TXT) XML parameter types.

    """

    def parse_elem(self, elem) -> str:
        """Returns a string constructed from ``elem.text``."""
        return "" if elem.text is None else elem.text

    def set_elem(self, elem, val: str) -> None:
        """Set ``elem.text`` to ``val``."""
        elem.text = val or None


class EnuType(TxtType):
    """Transcoder for enumeration (ENU) XML parameter types.

    Alias for `TxtType` (does not enforce allowable values).
    """


class BoolType(Type):
    """
    Transcoder for boolean (BOOL) XML parameter types.

    """

    def parse_elem(self, elem: lxml.etree.Element) -> bool:
        """Returns a Boolean value constructed from the string ``elem.text``."""
        val_collapsed = elem.text.strip()
        if val_collapsed in ("0", "false"):
            return False
        if val_collapsed in ("1", "true"):
            return True
        raise ValueError(f"{val_collapsed} is not in xs:boolean's lexical space")

    def set_elem(self, elem: lxml.etree.Element, val: bool) -> None:
        """Set ``elem.text`` to a string representation of the boolean ``val``."""
        elem.text = str(val).lower()


class IntType(Type):
    """
    Transcoder for integer (INT) XML parameter types.

    """

    def parse_elem(self, elem: lxml.etree.Element) -> int:
        """Returns an integer object constructed from the string ``elem.text``."""
        return int(elem.text)


class DblType(Type):
    """
    Transcoder for double-precision floating point (DBL) XML parameter types.

    """

    def parse_elem(self, elem: lxml.etree.Element) -> float:
        """Returns a floating point number constructed from the string ``elem.text``."""
        return float(elem.text)


class HexType(Type):
    """
    Transcoder for HEX XML parameter types.

    """

    def parse_elem(self, elem: lxml.etree.Element) -> bytes:
        """Returns a byte string constructed from the string ``elem.text``."""
        return bytes.fromhex(elem.text)

    def set_elem(self, elem: lxml.etree.Element, val: bytes) -> None:
        """Set ``elem.text`` to a hex string representation of the byte string ``val``."""
        elem.text = val.hex().upper()


class XdtType(Type):
    """
    Transcoder for XML dateTime (XDT) XML parameter types.

    Parameters
    ----------
    force_utc : bool, optional
        If ``True``, naive datetimes are treated as UTC times and
        non-UTC times are adjusted to UTC prior to transcoding.

    """

    def __init__(self, force_utc: bool = True) -> None:
        self.force_utc = force_utc

    def parse_elem(self, elem: lxml.etree.Element) -> datetime.datetime:
        """Returns a `datetime` constructed from the string ``elem.text``."""
        val = datetime.datetime.fromisoformat(elem.text)
        if self.force_utc:
            is_aware = val.tzinfo is not None and val.tzinfo.utcoffset(None) is not None
            val = (
                val.astimezone(datetime.UTC)
                if is_aware
                else val.replace(tzinfo=datetime.UTC)
            )
        return val

    def set_elem(self, elem: lxml.etree.Element, val: datetime.datetime) -> None:
        """Set ``elem.text`` to a string representation of the date and time from ``val``."""
        is_aware = val.tzinfo is not None and val.tzinfo.utcoffset(None) is not None

        format_str = "%Y-%m-%dT%H:%M:%S.%f"
        if self.force_utc or (is_aware and not val.utcoffset()):
            format_str += "Z"
        if self.force_utc and is_aware:
            val = val.astimezone(datetime.UTC)
        elem.text = val.strftime(format_str)


class PolyNdType(Type):
    """
    Transcoder for N-variate polynomial (POLY, 2D_POLY, etc.) XML parameter types.

    Parameters
    ----------
    nvar : int, optional
        Number of variables in polynomial.
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, nvar: int = 1, child_ns: str = "") -> None:
        self.nvar = nvar
        self.child_ns = child_ns

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns an array of coefficients describing the polynomial encoded in ``elem``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to parse

        Returns
        -------
        coefs : ndarray
            ``nvar``-dimensional array of coefficients ordered so that the coefficient of
            the term of multi-exponent n_1, n_2, ..., n_nvar is contained in ``val[n_1, n_2, ..., n_nvar]``.

        """
        coef_by_exponents = {
            tuple(
                int(coef.get(f"exponent{x}")) for x in range(1, self.nvar + 1)
            ): float(coef.text)
            for coef in elem
        }
        coefs = np.zeros(np.max(list(coef_by_exponents), axis=0) + 1, np.float64)
        for exponents, coef in coef_by_exponents.items():
            coefs[*exponents] = coef
        return coefs

    def set_elem(self, elem: lxml.etree.Element, val: npt.ArrayLike) -> None:
        """Set ``elem`` node using the polynomial coefficients from ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : array_like
            ``nvar``-dimensional array of coefficients ordered so that the coefficient of
            the term of multi-exponent n_1, n_2, ..., n_nvar is contained in ``val[n_1, n_2, ..., n_nvar]``.

        """
        coefs = np.asarray(val)
        if coefs.ndim != self.nvar:
            raise ValueError(f"Coefficient array must have ndim={self.nvar}")
        elem[:] = []
        elem_ns = self.child_ns if self.child_ns else lxml.etree.QName(elem).namespace
        ns = f"{{{elem_ns}}}" if elem_ns else ""
        for dim, ncoef in enumerate(coefs.shape):
            elem.set(f"order{dim + 1}", str(ncoef - 1))
        for coord, coef in np.ndenumerate(coefs):
            attribs = {f"exponent{d + 1}": str(c) for d, c in enumerate(coord)}
            lxml.etree.SubElement(elem, ns + "Coef", attrib=attribs).text = str(coef)


class PolyType(PolyNdType):
    """Transcoder for one-dimensional polynomial (POLY) XML parameter types."""

    def __init__(self, *, child_ns=""):
        super().__init__(child_ns=child_ns)


class Poly2dType(PolyNdType):
    """Transcoder for two-dimensional polynomial (2D_POLY) XML parameter types."""

    def __init__(self, *, child_ns=""):
        super().__init__(nvar=2, child_ns=child_ns)


class XyzPolyType(Type):
    """
    Transcoder for XYZ_POLY XML parameter types containing triplets of 1D polynomials.

    Parameters
    ----------
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, child_ns: str = "") -> None:
        self.child_ns = child_ns

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns an array of coefficients describing the polynomials encoded in ``elem``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to parse

        Returns
        -------
        coefs : (N, 3) ndarray
            Array of coefficients ordered so that the coefficients for terms of degree
            n are contained in ``val[n]``. Dimension two enumerates the X, Y, and Z
            polynomials.

        """
        xyz = [PolyType().parse_elem(elem.find(f"{{*}}{d}")) for d in "XYZ"]
        xyz_coefs = np.zeros_like(xyz[0], shape=(max(len(d) for d in xyz), len(xyz)))
        for dim, coefs in enumerate(xyz):
            xyz_coefs[: len(coefs), dim] = coefs
        return xyz_coefs

    def set_elem(self, elem: lxml.etree.Element, val: npt.ArrayLike) -> None:
        """Set ``elem`` node using the polynomial coefficients from ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : (N, 3) array_like
            Array of coefficients ordered so that the coefficients for terms of degree
            n are contained in ``val[n]``. Dimension two enumerates the X, Y, and Z
            polynomials.

        """
        coefs = np.asarray(val)
        if coefs.shape[1] != 3:
            raise ValueError(f"{coefs.shape[1]=} must be 3")
        elem[:] = []
        elem_ns = self.child_ns if self.child_ns else lxml.etree.QName(elem).namespace
        ns = f"{{{elem_ns}}}" if elem_ns else ""
        for index, tag in enumerate("XYZ"):
            subelem = lxml.etree.SubElement(elem, ns + tag)
            PolyType().set_elem(subelem, coefs[:, index])


class SequenceType(abc.ABC, Type):
    """
    Abstract base class for XML types containing a defined sequence of subelements.

    Parameters
    ----------
    subelements : dict of str: Type
        Mapping of subelement tags to transcoder type
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, subelements: dict[str, Type], child_ns: str = "") -> None:
        self.subelements = subelements
        self.child_ns = child_ns

    @abc.abstractmethod
    def parse_elem(self, elem: lxml.etree.Element) -> Any:
        pass

    @abc.abstractmethod
    def set_elem(self, elem: lxml.etree.Element, val: Any) -> None:
        pass

    def parse_subelements(self, elem: lxml.etree.Element) -> dict[str, Any]:
        """Returns an array containing the sub-elements encoded in ``elem``."""
        return {
            e_name: e_type.parse_elem(elem.find(f"{{*}}{e_name}"))
            for e_name, e_type in self.subelements.items()
        }

    def set_subelements(self, elem: lxml.etree.Element, val: dict[str, Any]) -> None:
        """Set ``elem`` node subelements using ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : dict
            Mapping of subelement tags to subelement values

        """
        if self.subelements.keys() != val.keys():
            raise ValueError(f"{(val.keys())=} must match {self.subelements.keys()=}")
        elem[:] = []
        elem_ns = self.child_ns if self.child_ns else lxml.etree.QName(elem).namespace
        ns = f"{{{elem_ns}}}" if elem_ns else ""
        for e_name, e_type in self.subelements.items():
            subelem = lxml.etree.SubElement(elem, ns + e_name)
            e_type.set_elem(subelem, val[e_name])


class ArrayType(SequenceType):
    """
    Base transcoder class for XML parameter types whose subelements can be treated as an array.

    """

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns an array containing the sub-elements encoded in ``elem``."""
        return np.array(list(super().parse_subelements(elem).values()))

    def set_elem(self, elem: lxml.etree.Element, val: Sequence[Any]) -> None:
        """Set ``elem`` node using ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : array_like
            Sequence of element values in ``subelements`` order

        """
        if len(self.subelements) != len(val):
            raise ValueError(
                f"{len(self.subelements)=} does not match expected {len(val)=}"
            )
        super().set_subelements(
            elem, dict((k, v) for k, v in zip(self.subelements, val, strict=True))
        )


class XyType(ArrayType):
    """
    Transcoder for XML parameter types containing scalar X and Y components.

    Parameters
    ----------
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, child_ns: str = "") -> None:
        super().__init__(
            subelements={c: DblType() for c in ("X", "Y")}, child_ns=child_ns
        )


class XyzType(ArrayType):
    """
    Transcoder for XML parameter types containing scalar X, Y, and Z components.

    Parameters
    ----------
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, child_ns: str = "") -> None:
        super().__init__(
            subelements={c: DblType() for c in ("X", "Y", "Z")}, child_ns=child_ns
        )


class LatLonType(ArrayType):
    """
    Transcoder for XML parameter types containing scalar Lat and Lon components.

    Parameters
    ----------
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, child_ns: str = "") -> None:
        super().__init__(
            subelements={c: DblType() for c in ("Lat", "Lon")}, child_ns=child_ns
        )


class LatLonHaeType(ArrayType):
    """
    Transcoder for XML parameter types containing scalar Lat, Lon, and HAE components.

    Parameters
    ----------
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, child_ns: str = "") -> None:
        super().__init__(
            subelements={c: DblType() for c in ("Lat", "Lon", "HAE")}, child_ns=child_ns
        )


class RowColType(ArrayType):
    """
    Transcoder for XML parameter types containing scalar, integer Row and Col components.

    Parameters
    ----------
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, child_ns: str = "") -> None:
        super().__init__(
            subelements={c: IntType() for c in ("Row", "Col")}, child_ns=child_ns
        )


class LineSampType(ArrayType):
    """
    Transcoder for XML parameter types containing scalar Line and Sample components.

    Parameters
    ----------
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, child_ns: str = "") -> None:
        super().__init__(
            subelements={c: DblType() for c in ("Line", "Sample")}, child_ns=child_ns
        )


class CmplxType(SequenceType):
    """
    Transcoder for double-precision floating point complex (CMPLX) XML parameter types.

    Parameters
    ----------
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.

    """

    def __init__(self, child_ns: str = "") -> None:
        super().__init__(
            subelements={c: DblType() for c in ("Real", "Imag")}, child_ns=child_ns
        )

    def parse_elem(self, elem: lxml.etree.Element) -> complex:
        """Returns the complex number encoded in ``elem``."""
        return complex(*super().parse_subelements(elem).values())

    def set_elem(self, elem: lxml.etree.Element, val: complex) -> None:
        """Set ``elem`` node to the complex number ``val``."""
        super().set_subelements(elem, {"Real": val.real, "Imag": val.imag})


class SizedType(abc.ABC, Type):
    """
    Abstract base class for XML parameter types containing ordered subelements with a common tag.

    Parameters
    ----------
    sub_tag : str
        Tag of subelement (excluding namespace)
    sub_type : Type
        Transcoder type of subelements
    include_size_attr : bool, optional
        If True, the size attribute is included. Otherwise it is not. Default is True.
    index_start : int, optional
        Starting index. Default is 1.
    child_ns : str, optional
        Namespace to use for child elements.  Parent namespace used if unspecified.
    """

    def __init__(
        self,
        sub_tag: str,
        sub_type: Type,
        *,
        include_size_attr: bool = True,
        index_start: int = 1,
        child_ns: str = "",
    ) -> None:
        self.sub_tag = sub_tag
        self.sub_type = sub_type
        self.include_size_attr = include_size_attr
        self.index_start = index_start
        self.child_ns = child_ns

    @abc.abstractmethod
    def parse_elem(self, elem: lxml.etree.Element) -> Any:
        pass

    def iter_parse(self, elem: lxml.etree.Element) -> Iterator:
        """Yield sub-elements encoded in ``elem`` in indexed order."""
        for x in sorted(elem, key=lambda x: int(x.get("index"))):
            yield self.sub_type.parse_elem(x)

    def set_elem(self, elem: lxml.etree.Element, val: Sequence[Any]) -> None:
        """Set ``elem`` node using ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : array_like
            Sequence of element values in ``subelements`` order

        """
        elem[:] = []
        elem_ns = self.child_ns if self.child_ns else lxml.etree.QName(elem).namespace
        ns = f"{{{elem_ns}}}" if elem_ns else ""
        if self.include_size_attr:
            elem.set("size", str(len(val)))
        for index, sub_val in enumerate(val):
            subelem = lxml.etree.SubElement(elem, ns + self.sub_tag)
            self.sub_type.set_elem(subelem, sub_val)
            subelem.set("index", str(index + self.index_start))


class ListType(SizedType):
    """
    Transcoder for XML parameter types containing an ordered list of subelements with a common tag.

    """

    def parse_elem(self, elem: lxml.etree.Element) -> list:
        """Returns an list containing the sub-elements encoded in ``elem``."""
        return list(self.iter_parse(elem))


class NdArrayType(SizedType):
    """
    Like `ListType`, but returns an `ndarray`
    """

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns an array containing the sub-elements encoded in ``elem``."""
        return np.array(list(self.iter_parse(elem)))


class ParameterType(Type):
    """
    Transcoder for TXT XML parameter types with a required "name" attribute.

    """

    def parse_elem(self, elem) -> tuple[str, str]:
        """Returns a tuple containing (``elem["name"]``, ``elem.text``)."""
        name = elem.get("name")
        val = TxtType().parse_elem(elem)
        return (name, val)

    def set_elem(self, elem, val: tuple[str, str]) -> None:
        """Set ``elem``'s name and value from a tuple of strings: (``name``, ``text``)"""
        elem.set("name", val[0])
        TxtType().set_elem(elem, val[1])


class MtxType(Type):
    """
    Transcoder for MTX XML parameter types containing a matrix.

    Attributes
    ----------
    shape : tuple of (int, int)
        Expected shape of the matrix.

    """

    def __init__(self, shape) -> None:
        self.shape = shape

    def parse_elem(self, elem: lxml.etree.Element) -> npt.NDArray:
        """Returns an array containing the matrix encoded in ``elem``."""
        shape = tuple(int(elem.get(f"size{d}")) for d in (1, 2))
        if self.shape != shape:
            raise ValueError(f"elem {shape=} does not match expected {self.shape}")
        val = np.zeros(shape)
        for entry in elem:
            val[*[int(entry.get(f"index{x}")) - 1 for x in (1, 2)]] = float(entry.text)
        return val

    def set_elem(self, elem: lxml.etree.Element, val: npt.ArrayLike) -> None:
        """Set ``elem`` node using ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : array_like
            matrix of shape= ``shape``

        """
        mtx = np.asarray(val)
        if self.shape != mtx.shape:
            raise ValueError(f"{mtx.shape=} does not match expected {self.shape}")
        elem[:] = []
        elem_ns = lxml.etree.QName(elem).namespace
        ns = f"{{{elem_ns}}}" if elem_ns else ""
        for d, nd in zip((1, 2), mtx.shape, strict=True):
            elem.set(f"size{d}", str(nd))
        for indices, entry in np.ndenumerate(mtx):
            attribs = {f"index{d + 1}": str(c + 1) for d, c in enumerate(indices)}
            lxml.etree.SubElement(elem, ns + "Entry", attrib=attribs).text = str(entry)
