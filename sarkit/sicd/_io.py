"""
Functions to read and write SICD files.
"""

import copy
import dataclasses
import datetime
import importlib.resources
import itertools
import logging
import os
import warnings
from typing import Any, Final, Self, TypedDict

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit._nitf_io
import sarkit.sicd._xml as sicd_xml
import sarkit.wgs84

SPECIFICATION_IDENTIFIER: Final[str] = (
    "SICD Volume 1 Design & Implementation Description Document"
)

SCHEMA_DIR = importlib.resources.files("sarkit.sicd.schemas")


class VersionInfoType(TypedDict):
    version: str
    date: str
    schema: importlib.resources.abc.Traversable


# Keys must be in ascending order
VERSION_INFO: Final[dict[str, VersionInfoType]] = {
    "urn:SICD:1.1.0": {
        "version": "1.1",
        "date": "2014-09-30T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.1.0_2014_09_30.xsd",
    },
    "urn:SICD:1.2.1": {
        "version": "1.2.1",
        "date": "2018-12-13T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.2.1_2018_12_13.xsd",
    },
    "urn:SICD:1.3.0": {
        "version": "1.3.0",
        "date": "2021-11-30T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.3.0_2021_11_30.xsd",
    },
    "urn:SICD:1.4.0": {
        "version": "1.4.0",
        "date": "2023-10-26T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.4.0_2023_10_26.xsd",
    },
}


PIXEL_TYPES: Final[dict[str, dict[str, Any]]] = {
    "RE32F_IM32F": {
        "bytes": 8,
        "pvtype": "R",
        "subcat": ("I", "Q"),
        "dtype": np.dtype(np.complex64),
    },
    "RE16I_IM16I": {
        "bytes": 4,
        "pvtype": "SI",
        "subcat": ("I", "Q"),
        "dtype": np.dtype([("real", np.int16), ("imag", np.int16)]),
    },
    "AMP8I_PHS8I": {
        "bytes": 2,
        "pvtype": "INT",
        "subcat": ("M", "P"),
        "dtype": np.dtype([("amp", np.uint8), ("phase", np.uint8)]),
    },
}


@dataclasses.dataclass(kw_only=True)
class NitfSecurityFields:
    """NITF Security Header/Subheader fields

    Attributes
    ----------
    clas : str
        Security Classification
    clsy : str
        Security Classification System
    code : str
        Codewords
    ctlh : str
        Control and Handling
    rel : str
        Releasing Instructions
    dctp : str
        Declassification Type
    dcdt : str
        Declassification Date
    dcxm : str
        Declassification Exemption
    dg : str
        Downgrade
    dgdt : str
        Downgrade Date
    cltx : str
        Classification Text
    catp : str
        Classification Authority Type
    caut : str
        Classification Authority
    crsn : str
        Classification Reason
    srdt : str
        Security Source Date
    ctln : str
        Security Control Number
    """

    clas: str
    clsy: str = ""
    code: str = ""
    ctlh: str = ""
    rel: str = ""
    dctp: str = ""
    dcdt: str = ""
    dcxm: str = ""
    dg: str = ""
    dgdt: str = ""
    cltx: str = ""
    catp: str = ""
    caut: str = ""
    crsn: str = ""
    srdt: str = ""
    ctln: str = ""

    @classmethod
    def _from_nitf_fields(
        cls,
        prefix: str,
        field_group: sarkit._nitf_io.Group,
    ) -> Self:
        """Construct from NITF security fields"""
        return cls(
            clas=field_group[f"{prefix}CLAS"].value,
            clsy=field_group[f"{prefix}CLSY"].value,
            code=field_group[f"{prefix}CODE"].value,
            ctlh=field_group[f"{prefix}CTLH"].value,
            rel=field_group[f"{prefix}REL"].value,
            dctp=field_group[f"{prefix}DCTP"].value,
            dcdt=field_group[f"{prefix}DCDT"].value,
            dcxm=field_group[f"{prefix}DCXM"].value,
            dg=field_group[f"{prefix}DG"].value,
            dgdt=field_group[f"{prefix}DGDT"].value,
            cltx=field_group[f"{prefix}CLTX"].value,
            catp=field_group[f"{prefix}CATP"].value,
            caut=field_group[f"{prefix}CAUT"].value,
            crsn=field_group[f"{prefix}CRSN"].value,
            srdt=field_group[f"{prefix}SRDT"].value,
            ctln=field_group[f"{prefix}CTLN"].value,
        )

    def _set_nitf_fields(self, prefix: str, field_group: sarkit._nitf_io.Group) -> None:
        """Set NITF security fields"""
        field_group[f"{prefix}CLAS"].value = self.clas
        field_group[f"{prefix}CLSY"].value = self.clsy
        field_group[f"{prefix}CODE"].value = self.code
        field_group[f"{prefix}CTLH"].value = self.ctlh
        field_group[f"{prefix}REL"].value = self.rel
        field_group[f"{prefix}DCTP"].value = self.dctp
        field_group[f"{prefix}DCDT"].value = self.dcdt
        field_group[f"{prefix}DCXM"].value = self.dcxm
        field_group[f"{prefix}DG"].value = self.dg
        field_group[f"{prefix}DGDT"].value = self.dgdt
        field_group[f"{prefix}CLTX"].value = self.cltx
        field_group[f"{prefix}CATP"].value = self.catp
        field_group[f"{prefix}CAUT"].value = self.caut
        field_group[f"{prefix}CRSN"].value = self.crsn
        field_group[f"{prefix}SRDT"].value = self.srdt
        field_group[f"{prefix}CTLN"].value = self.ctln


@dataclasses.dataclass(kw_only=True)
class NitfFileHeaderPart:
    """NITF header fields which are set according to a Program Specific Implementation Document

    Attributes
    ----------
    ostaid : str
        Originating Station ID
    ftitle : str
        File Title
    security : NitfSecurityFields
        Security Tags with "FS" prefix
    oname : str
        Originator's Name
    ophone : str
        Originator's Phone
    """

    ostaid: str
    ftitle: str = ""
    security: NitfSecurityFields
    oname: str = ""
    ophone: str = ""

    @classmethod
    def _from_header(cls, file_header: sarkit._nitf_io.FileHeader) -> Self:
        """Construct from a NITF File Header object"""
        return cls(
            ostaid=file_header["OSTAID"].value,
            ftitle=file_header["FTITLE"].value,
            security=NitfSecurityFields._from_nitf_fields("FS", file_header),
            oname=file_header["ONAME"].value,
            ophone=file_header["OPHONE"].value,
        )

    def __post_init__(self):
        if isinstance(self.security, dict):
            self.security = NitfSecurityFields(**self.security)


@dataclasses.dataclass(kw_only=True)
class NitfImSubheaderPart:
    """NITF image header fields which are set according to a Program Specific Implementation Document

    Attributes
    ----------
    tgtid : str
       Target Identifier
    iid2 : str
        Image Identifier 2
    security : NitfSecurityFields
        Security Tags with "IS" prefix
    isorce : str
        Image Source
    icom : list of str
        Image Comments
    """

    ## IS fields are applied to all segments
    tgtid: str = ""
    iid2: str = ""
    security: NitfSecurityFields
    isorce: str
    icom: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def _from_header(cls, image_header: sarkit._nitf_io.ImageSubHeader) -> Self:
        """Construct from a NITF ImageSubHeader object"""
        return cls(
            tgtid=image_header["TGTID"].value,
            iid2=image_header["IID2"].value,
            security=NitfSecurityFields._from_nitf_fields("IS", image_header),
            isorce=image_header["ISORCE"].value,
            icom=[val.value for val in image_header.find_all("ICOM\\d+")],
        )

    def __post_init__(self):
        if isinstance(self.security, dict):
            self.security = NitfSecurityFields(**self.security)


@dataclasses.dataclass(kw_only=True)
class NitfDeSubheaderPart:
    """NITF DES subheader fields which are set according to a Program Specific Implementation Document

    Attributes
    ----------
    security : NitfSecurityFields
        Security Tags with "DES" prefix
    desshrp : str
        Responsible Party - Organization Identifier
    desshli : str
        Location - Identifier
    desshlin : str
        Location Identifier Namespace URI
    desshabs : str
        Abstract. Brief narrative summary of the content of the DES.
    """

    security: NitfSecurityFields
    desshrp: str = ""
    desshli: str = ""
    desshlin: str = ""
    desshabs: str = ""

    @classmethod
    def _from_header(cls, de_header: sarkit._nitf_io.DESubHeader) -> Self:
        """Construct from a NITF DESubHeader object"""
        return cls(
            security=NitfSecurityFields._from_nitf_fields("DES", de_header),
            desshrp=de_header["DESSHF"]["DESSHRP"].value,
            desshli=de_header["DESSHF"]["DESSHLI"].value,
            desshlin=de_header["DESSHF"]["DESSHLIN"].value,
            desshabs=de_header["DESSHF"]["DESSHABS"].value,
        )

    def __post_init__(self):
        if isinstance(self.security, dict):
            self.security = NitfSecurityFields(**self.security)


@dataclasses.dataclass(kw_only=True)
class NitfMetadata:
    """Settable SICD NITF metadata

    Attributes
    ----------
    xmltree : lxml.etree.ElementTree
        SICD XML
    file_header_part : NitfFileHeaderPart
        NITF File Header fields which can be set
    im_subheader_part : NitfImSubheaderPart
        NITF image subheader fields which can be set
    de_subheader_part : NitfDeSubheaderPart
        NITF DES subheader fields which can be set
    """

    xmltree: lxml.etree.ElementTree
    file_header_part: NitfFileHeaderPart
    im_subheader_part: NitfImSubheaderPart
    de_subheader_part: NitfDeSubheaderPart

    def __post_init__(self):
        if isinstance(self.file_header_part, dict):
            self.file_header_part = NitfFileHeaderPart(**self.file_header_part)
        if isinstance(self.im_subheader_part, dict):
            self.im_subheader_part = NitfImSubheaderPart(**self.im_subheader_part)
        if isinstance(self.de_subheader_part, dict):
            self.de_subheader_part = NitfDeSubheaderPart(**self.de_subheader_part)

    def __eq__(self, other):
        if isinstance(other, NitfMetadata):
            self_parts = (
                lxml.etree.tostring(self.xmltree, method="c14n"),
                self.file_header_part,
                self.im_subheader_part,
                self.de_subheader_part,
            )
            other_parts = (
                lxml.etree.tostring(other.xmltree, method="c14n"),
                other.file_header_part,
                other.im_subheader_part,
                other.de_subheader_part,
            )
            return self_parts == other_parts
        return False


class NitfReader:
    """Read a SICD NITF

    A NitfReader object can be used as a context manager in a ``with`` statement.
    Attributes, but not methods, can be safely accessed outside of the context manager's context.

    Parameters
    ----------
    file : `file object`
        SICD NITF file to read

    Attributes
    ----------
    metadata : NitfMetadata
        SICD NITF metadata

    See Also
    --------
    NitfWriter

    Examples
    --------
    .. testsetup::

        import lxml.etree
        import numpy as np

        import sarkit.sicd as sksicd

        file = tmppath / "example.sicd"
        sec = {"security": {"clas": "U"}}
        example_sicd_xmltree = lxml.etree.parse("data/example-sicd-1.4.0.xml")
        sicd_meta = sksicd.NitfMetadata(
            xmltree=example_sicd_xmltree,
            file_header_part={"ostaid": "nowhere", "ftitle": "SARkit example SICD FTITLE"} | sec,
            im_subheader_part={"isorce": "this sensor"} | sec,
            de_subheader_part=sec,
        )
        with open(file, "wb") as f, sksicd.NitfWriter(f, sicd_meta):
            pass  # don't currently care about the pixels

    .. doctest::

        >>> import sarkit.sicd as sicd
        >>> with file.open("rb") as f, sksicd.NitfReader(f) as r:
        ...     img = r.read_image()

        >>> print(r.metadata.xmltree.getroot().tag)
        {urn:SICD:1.4.0}SICD

        >>> print(r.metadata.im_subheader_part.isorce)
        this sensor
    """

    def __init__(self, file):
        self._file_object = file

        self._ntf = sarkit._nitf_io.Nitf().load(file)

        deseg = self._ntf["DESegments"][0]  # SICD XML must be in first DES
        if not deseg["SubHeader"]["DESSHF"]["DESSHTN"].value.startswith("urn:SICD"):
            raise ValueError(f"Unable to find SICD DES in {file}")

        file.seek(deseg["DESDATA"].get_offset(), os.SEEK_SET)
        sicd_xmltree = lxml.etree.fromstring(
            file.read(deseg["DESDATA"].size)
        ).getroottree()

        nitf_header_fields = NitfFileHeaderPart._from_header(self._ntf["FileHeader"])
        nitf_image_fields = NitfImSubheaderPart._from_header(
            self._ntf["ImageSegments"][0]["SubHeader"],
        )
        nitf_de_fields = NitfDeSubheaderPart._from_header(deseg["SubHeader"])

        self.metadata = NitfMetadata(
            xmltree=sicd_xmltree,
            file_header_part=nitf_header_fields,
            im_subheader_part=nitf_image_fields,
            de_subheader_part=nitf_de_fields,
        )

    def read_image(self) -> npt.NDArray:
        """Read the entire pixel array

        Returns
        -------
        ndarray
            SICD image array
        """
        self._file_object.seek(0, os.SEEK_SET)
        nrows = int(self.metadata.xmltree.findtext("{*}ImageData/{*}NumRows"))
        ncols = int(self.metadata.xmltree.findtext("{*}ImageData/{*}NumCols"))
        pixel_type = self.metadata.xmltree.findtext("{*}ImageData/{*}PixelType")
        dtype = PIXEL_TYPES[pixel_type]["dtype"].newbyteorder(">")
        sicd_pixels = np.empty((nrows, ncols), dtype)

        imsegs = sorted(
            [
                imseg
                for imseg in self._ntf["ImageSegments"]
                if imseg["SubHeader"]["IID1"].value.startswith("SICD")
            ],
            key=lambda seg: seg["SubHeader"]["IID1"].value,
        )

        for imseg in imsegs:
            ic_value = imseg["SubHeader"]["IC"].value
            if ic_value != "NC":
                raise RuntimeError(
                    f"SICDs with Compression and/or Masking not supported. IC={ic_value}"
                )

        imseg_sizes = np.asarray([imseg["Data"].size for imseg in imsegs])
        imseg_offsets = np.asarray([imseg["Data"].get_offset() for imseg in imsegs])
        splits = np.cumsum(imseg_sizes // (ncols * dtype.itemsize))[:-1]
        for split, sz, offset in zip(
            np.array_split(sicd_pixels, splits, axis=0), imseg_sizes, imseg_offsets
        ):
            this_os = offset - self._file_object.tell()
            split[...] = np.fromfile(
                self._file_object, dtype, count=sz // dtype.itemsize, offset=this_os
            ).reshape(split.shape)
        return sicd_pixels

    def read_sub_image(
        self,
        start_row: int = 0,
        start_col: int = 0,
        end_row: int = -1,
        end_col: int = -1,
    ) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
        """Read a sub-image from the file

        Parameters
        ----------
        start_row : int
        start_col : int
        end_row : int
        end_col : int

        Returns
        -------
        ndarray
            SICD sub-image array
        lxml.etree.ElementTree
            SICD sub-image XML ElementTree
        """
        raise NotImplementedError()

    def done(self):
        "Indicates to the reader that the user is done with it"
        self._file_object = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.done()


@dataclasses.dataclass(kw_only=True)
class SizingImhdr:
    """Per segment values computed by the SICD Image Sizing Algorithm"""

    idlvl: int
    ialvl: int
    iloc_rows: int
    nrows: int
    igeolo: str


def _format_igeolo(iscc):
    def _format_dms(value, lon_or_lat):
        if lon_or_lat == "lat":
            dirs = {1: "N", -1: "S"}
            deg_digits = 2
        else:
            dirs = {1: "E", -1: "W"}
            deg_digits = 3

        direction = dirs[np.sign(value)]
        secs = np.abs(round(value * 3600))
        degrees = secs // 3600
        minutes = (secs // 60) % 60
        seconds = secs % 60

        return f"{int(degrees):0{deg_digits}d}{int(minutes):02d}{int(seconds):02d}{direction}"

    return "".join(
        [
            _format_dms(iscc[0][0], "lat"),
            _format_dms(iscc[0][1], "lon"),
            _format_dms(iscc[1][0], "lat"),
            _format_dms(iscc[1][1], "lon"),
            _format_dms(iscc[2][0], "lat"),
            _format_dms(iscc[2][1], "lon"),
            _format_dms(iscc[3][0], "lat"),
            _format_dms(iscc[3][1], "lon"),
        ]
    )


def image_segment_sizing_calculations(
    sicd_xmltree: lxml.etree.ElementTree,
) -> tuple[int, list[SizingImhdr]]:
    """3.2 Image Segment Sizing Calculations

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree

    Returns
    -------
    int
        Number of Image Segments (NumIS)
    list of :py:class:`SizingImhdr`
        One per Image Segment

    """

    xml_helper = sicd_xml.XmlHelper(sicd_xmltree)

    # 3.2.1 Image Segment Parameters and Equations
    pixel_type = xml_helper.load("./{*}ImageData/{*}PixelType")
    num_rows = xml_helper.load("{*}ImageData/{*}NumRows")
    num_cols = xml_helper.load("{*}ImageData/{*}NumCols")

    bytes_per_pixel = {"RE32F_IM32F": 8, "RE16I_IM16I": 4, "AMP8I_PHS8I": 2}[pixel_type]

    is_size_max = 9_999_999_998
    iloc_max = 99_999
    bytes_per_row = bytes_per_pixel * num_cols
    product_size = bytes_per_pixel * num_rows * num_cols
    limit1 = int(np.floor(is_size_max / bytes_per_row))
    num_rows_limit = min(limit1, iloc_max)
    if product_size <= is_size_max:
        num_is = 1
        num_rows_is = [num_rows]
        first_row_is = [0]
        row_offset_is = [0]
    else:
        num_is = int(np.ceil(num_rows / num_rows_limit))
        num_rows_is = [0] * num_is
        first_row_is = [0] * num_is
        row_offset_is = [0] * num_is
        for n in range(num_is - 1):
            num_rows_is[n] = num_rows_limit
            first_row_is[n + 1] = (n + 1) * num_rows_limit
            row_offset_is[n + 1] = num_rows_limit
        num_rows_is[-1] = num_rows - (num_is - 1) * num_rows_limit

    icp_latlon = xml_helper.load("./{*}GeoData/{*}ImageCorners")

    icp_ecef = [
        sarkit.wgs84.geodetic_to_cartesian([np.deg2rad(lat), np.deg2rad(lon), 0])
        for lat, lon in icp_latlon
    ]

    iscp_ecef = np.zeros((num_is, 4, 3))
    for imidx in range(num_is):
        wgt1 = (num_rows - 1 - first_row_is[imidx]) / (num_rows - 1)
        wgt2 = first_row_is[imidx] / (num_rows - 1)
        iscp_ecef[imidx][0] = wgt1 * icp_ecef[0] + wgt2 * icp_ecef[3]
        iscp_ecef[imidx][1] = wgt1 * icp_ecef[1] + wgt2 * icp_ecef[2]

    for imidx in range(num_is - 1):
        iscp_ecef[imidx][2] = iscp_ecef[imidx + 1][1]
        iscp_ecef[imidx][3] = iscp_ecef[imidx + 1][0]
    iscp_ecef[num_is - 1][2] = icp_ecef[2]
    iscp_ecef[num_is - 1][3] = icp_ecef[3]

    iscp_latlon = np.rad2deg(sarkit.wgs84.cartesian_to_geodetic(iscp_ecef)[:, :, :2])

    # 3.2.2 File Header and Image Sub-Header Parameters
    seginfos = []
    for n in range(num_is):
        seginfos.append(
            SizingImhdr(
                nrows=num_rows_is[n],
                iloc_rows=row_offset_is[n],
                idlvl=n + 1,
                ialvl=n,
                igeolo=_format_igeolo(iscp_latlon[n]),
            )
        )

    return num_is, seginfos


class NitfWriter:
    """Write a SICD NITF

    A NitfWriter object can be used as a context manager in a ``with`` statement.

    Parameters
    ----------
    file : `file object`
        SICD NITF file to write
    metadata : NitfMetadata
        SICD NITF metadata to write (copied on construction)

    See Also
    --------
    NitfReader

    Examples
    --------
    Construct a SICD metadata object...

    .. doctest::

        >>> import lxml.etree
        >>> import sarkit.sicd as sksicd
        >>> sicd_xml = lxml.etree.parse("data/example-sicd-1.4.0.xml")
        >>> sec = sksicd.NitfSecurityFields(clas="U")
        >>> meta = sksicd.NitfMetadata(
        ...     xmltree=sicd_xml,
        ...     file_header_part=sksicd.NitfFileHeaderPart(ostaid="my station", security=sec),
        ...     im_subheader_part=sksicd.NitfImSubheaderPart(isorce="my sensor", security=sec),
        ...     de_subheader_part=sksicd.NitfDeSubheaderPart(security=sec),
        ... )

    ... and associated complex image array.

    .. doctest::

        >>> import numpy as np
        >>> img_to_write = np.zeros(
        ...     (
        ...         sksicd.XmlHelper(sicd_xml).load("{*}ImageData/{*}NumRows"),
        ...         sksicd.XmlHelper(sicd_xml).load("{*}ImageData/{*}NumCols"),
        ...     ),
        ...     dtype=sksicd.PIXEL_TYPES[sicd_xml.findtext("{*}ImageData/{*}PixelType")]["dtype"],
        ... )

    Write the SICD NITF to a file

    .. doctest::

        >>> from tempfile import NamedTemporaryFile
        >>> outfile = NamedTemporaryFile()
        >>> with sksicd.NitfWriter(outfile, meta) as w:
        ...     w.write_image(img_to_write)
    """

    def __init__(self, file, metadata: NitfMetadata):
        self._file_object = file

        self._metadata = copy.deepcopy(metadata)
        sicd_xmltree = self._metadata.xmltree
        xmlns = lxml.etree.QName(sicd_xmltree.getroot()).namespace
        schema = lxml.etree.XMLSchema(file=VERSION_INFO[xmlns]["schema"])
        if not schema.validate(sicd_xmltree):
            warnings.warn(str(schema.error_log))

        xml_helper = sicd_xml.XmlHelper(sicd_xmltree)
        cols = xml_helper.load("./{*}ImageData/{*}NumCols")
        pixel_type = sicd_xmltree.findtext("./{*}ImageData/{*}PixelType")
        bits_per_element = PIXEL_TYPES[pixel_type]["bytes"] * 8 / 2

        num_is, seginfos = image_segment_sizing_calculations(sicd_xmltree)

        self._ntf = sarkit._nitf_io.Nitf()
        self._ntf["FileHeader"]["OSTAID"].value = self._metadata.file_header_part.ostaid
        self._ntf["FileHeader"]["FTITLE"].value = self._metadata.file_header_part.ftitle
        self._metadata.file_header_part.security._set_nitf_fields(
            "FS", self._ntf["FileHeader"]
        )
        self._ntf["FileHeader"]["ONAME"].value = self._metadata.file_header_part.oname
        self._ntf["FileHeader"]["OPHONE"].value = self._metadata.file_header_part.ophone
        self._ntf["FileHeader"]["NUMI"].value = num_is

        for idx, seginfo in enumerate(seginfos):
            subhdr = self._ntf["ImageSegments"][idx]["SubHeader"]
            if len(seginfos) > 1:
                subhdr["IID1"].value = f"SICD{idx + 1:03d}"
            else:
                subhdr["IID1"].value = "SICD000"
            subhdr["IDATIM"].value = xml_helper.load(
                "./{*}Timeline/{*}CollectStart"
            ).strftime("%Y%m%d%H%M%S")
            subhdr["TGTID"].value = self._metadata.im_subheader_part.tgtid
            subhdr["IID2"].value = self._metadata.im_subheader_part.iid2
            self._metadata.im_subheader_part.security._set_nitf_fields("IS", subhdr)
            subhdr["ISORCE"].value = self._metadata.im_subheader_part.isorce
            subhdr["NROWS"].value = seginfo.nrows
            subhdr["NCOLS"].value = cols
            subhdr["PVTYPE"].value = PIXEL_TYPES[pixel_type]["pvtype"]
            subhdr["IREP"].value = "NODISPLY"
            subhdr["ICAT"].value = "SAR"
            subhdr["ABPP"].value = bits_per_element
            subhdr["PJUST"].value = "R"
            subhdr["ICORDS"].value = "G"
            subhdr["IGEOLO"].value = seginfo.igeolo
            subhdr["IC"].value = "NC"
            subhdr["NICOM"].value = len(self._metadata.im_subheader_part.icom)
            for icomidx, icom in enumerate(self._metadata.im_subheader_part.icom):
                subhdr[f"ICOM{icomidx + 1}"].value = icom
            subhdr["NBANDS"].value = 2
            subhdr["ISUBCAT00001"].value = PIXEL_TYPES[pixel_type]["subcat"][0]
            subhdr["ISUBCAT00002"].value = PIXEL_TYPES[pixel_type]["subcat"][1]
            subhdr["IMODE"].value = "P"
            subhdr["NBPR"].value = 1
            subhdr["NBPC"].value = 1

            if subhdr["NCOLS"].value > 8192:
                subhdr["NPPBH"].value = 0
            else:
                subhdr["NPPBH"].value = subhdr["NCOLS"].value

            if subhdr["NROWS"].value > 8192:
                subhdr["NPPBV"].value = 0
            else:
                subhdr["NPPBV"].value = subhdr["NROWS"].value

            subhdr["NBPP"].value = bits_per_element
            subhdr["IDLVL"].value = idx + 1
            subhdr["IALVL"].value = idx
            subhdr["ILOC"].value = (seginfo.iloc_rows, 0)
            subhdr["IMAG"].value = "1.0 "

            self._ntf["ImageSegments"][idx]["Data"].size = (
                # No compression, no masking, no blocking
                subhdr["NROWS"].value
                * subhdr["NCOLS"].value
                * subhdr["NBANDS"].value
                * subhdr["NBPP"].value
                // 8
            )

        sicd_xml_bytes = lxml.etree.tostring(sicd_xmltree)
        self._ntf["FileHeader"]["NUMDES"].value = 1
        self._ntf["DESegments"][0]["DESDATA"].size = len(sicd_xml_bytes)
        self._set_de_segment(
            self._ntf["DESegments"][0], sicd_xmltree, self._metadata.de_subheader_part
        )

        self._ntf.finalize()  # compute lengths, CLEVEL, etc...
        self._ntf.dump(file)
        file.seek(self._ntf["DESegments"][0]["DESDATA"].get_offset(), os.SEEK_SET)
        file.write(sicd_xml_bytes)

    @staticmethod
    def _set_de_segment(de_segment, sicd_xmltree, de_subheader_part):
        subhdr = de_segment["SubHeader"]
        subhdr["DESID"].value = "XML_DATA_CONTENT"
        subhdr["DESVER"].value = 1
        de_subheader_part.security._set_nitf_fields("DES", subhdr)
        subhdr["DESSHL"].value = 773
        subhdr["DESSHF"]["DESCRC"].value = 99999
        subhdr["DESSHF"]["DESSHFT"].value = "XML"
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        subhdr["DESSHF"]["DESSHDT"].value = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        subhdr["DESSHF"]["DESSHRP"].value = de_subheader_part.desshrp
        subhdr["DESSHF"][
            "DESSHSI"
        ].value = "SICD Volume 1 Design & Implementation Description Document"

        xml_helper = sicd_xml.XmlHelper(sicd_xmltree)
        xmlns = lxml.etree.QName(sicd_xmltree.getroot()).namespace
        if xmlns not in VERSION_INFO:
            logging.warning(f"Unknown SICD version: {xmlns}")
            spec_date = "0000-00-00T00:00:00Z"
            spec_version = "unknown"
        else:
            spec_date = VERSION_INFO[xmlns]["date"]
            spec_version = VERSION_INFO[xmlns]["version"]

        subhdr["DESSHF"]["DESSHSD"].value = spec_date
        subhdr["DESSHF"]["DESSHSV"].value = spec_version
        subhdr["DESSHF"]["DESSHTN"].value = xmlns

        icp = xml_helper.load("./{*}GeoData/{*}ImageCorners")
        desshlpg = ""
        for icp_lat, icp_lon in itertools.chain(icp, [icp[0]]):
            desshlpg += f"{icp_lat:0=+12.8f}{icp_lon:0=+13.8f}"
        subhdr["DESSHF"]["DESSHLPG"].value = desshlpg
        subhdr["DESSHF"]["DESSHLI"].value = de_subheader_part.desshli
        subhdr["DESSHF"]["DESSHLIN"].value = de_subheader_part.desshlin
        subhdr["DESSHF"]["DESSHABS"].value = de_subheader_part.desshabs

    def write_image(self, array: npt.NDArray, start: None | tuple[int, int] = None):
        """Write pixel data to a NITF file

        Parameters
        ----------
        array : ndarray
            2D array of complex pixels
        start : tuple of (int, int), optional
            The start index (first_row, first_col) of `array` in the SICD image.
            If not given, `array` must be the full SICD image.

        """
        pixel_type = self._metadata.xmltree.findtext("./{*}ImageData/{*}PixelType")
        if PIXEL_TYPES[pixel_type]["dtype"] != array.dtype.newbyteorder("="):
            raise ValueError(
                f"Array dtype ({array.dtype}) does not match expected dtype ({PIXEL_TYPES[pixel_type]['dtype']}) "
                f"for PixelType={pixel_type}"
            )

        xml_helper = sicd_xml.XmlHelper(self._metadata.xmltree)
        rows = xml_helper.load("./{*}ImageData/{*}NumRows")
        cols = xml_helper.load("./{*}ImageData/{*}NumCols")
        sicd_shape = np.asarray((rows, cols))

        if start is None:
            # require array to be full image
            if np.any(array.shape != sicd_shape):
                raise ValueError(
                    f"Array shape {array.shape} does not match sicd shape {sicd_shape}."
                    "If writing only a portion of the image, use the 'start' argument"
                )
            start = (0, 0)
        else:
            raise NotImplementedError("start argument not yet supported")
        startarr = np.asarray(start)

        if not np.issubdtype(startarr.dtype, np.integer):
            raise ValueError(f"Start index must be integers {startarr=}")

        if np.any(startarr < 0):
            raise ValueError(f"Start index must be non-negative {startarr=}")

        stop = startarr + array.shape
        if np.any(stop > sicd_shape):
            raise ValueError(
                f"array goes beyond end of sicd. start + array.shape = {stop} sicd shape={sicd_shape}"
            )

        if pixel_type == "RE32F_IM32F":
            raw_dtype = array.real.dtype
        else:
            assert array.dtype.names is not None  # placate mypy
            raw_dtype = array.dtype[array.dtype.names[0]]

        imsegs = sorted(
            [
                imseg
                for imseg in self._ntf["ImageSegments"]
                if imseg["SubHeader"]["IID1"].value.startswith("SICD")
            ],
            key=lambda seg: seg["SubHeader"]["IID1"].value,
        )
        first_rows = np.cumsum(
            [0] + [imseg["SubHeader"]["NROWS"].value for imseg in imsegs[:-1]]
        )
        for imseg, first_row in zip(self._ntf["ImageSegments"], first_rows):
            self._file_object.seek(imseg["Data"].get_offset(), os.SEEK_SET)

            # Could break this into blocks to reduce memory usage from byte swapping
            raw_array = array[
                first_row : first_row + imseg["SubHeader"]["NROWS"].value
            ].view((raw_dtype, 2))
            raw_array = raw_array.astype(raw_dtype.newbyteorder(">"), copy=False)
            raw_array.tofile(self._file_object)

    def close(self):
        """
        Flush to disk and close any opened file descriptors.

        Called automatically when used as a context manager
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
