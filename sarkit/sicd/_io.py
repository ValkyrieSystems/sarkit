"""
Functions to read and write SICD files.
"""

import copy
import dataclasses
import datetime
import importlib.resources
import itertools
import warnings
from typing import Any, Final, Self

import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit._nitf.nitf
import sarkit._nitf.nitf_elements.des
import sarkit._nitf.nitf_elements.image
import sarkit._nitf.nitf_elements.nitf_head
import sarkit._nitf.nitf_elements.security
import sarkit._nitf.utils
import sarkit.sicd._xml as sicd_xml

SPECIFICATION_IDENTIFIER: Final[str] = (
    "SICD Volume 1 Design & Implementation Description Document"
)

SCHEMA_DIR = importlib.resources.files("sarkit.sicd.schemas")

# Keys in ascending order
VERSION_INFO: Final[dict] = {
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
    def _from_security_tags(
        cls, security: sarkit._nitf.nitf_elements.security.NITFSecurityTags
    ) -> Self:
        """Construct from a NITFSecurityTags object"""
        return cls(
            clas=security.CLAS,
            clsy=security.CLSY,
            code=security.CODE,
            ctlh=security.CTLH,
            rel=security.REL,
            dctp=security.DCTP,
            dcdt=security.DCDT,
            dcxm=security.DCXM,
            dg=security.DG,
            dgdt=security.DGDT,
            cltx=security.CLTX,
            catp=security.CATP,
            caut=security.CAUT,
            crsn=security.CRSN,
            srdt=security.SRDT,
            ctln=security.CTLN,
        )

    def _as_security_tags(
        self,
    ) -> sarkit._nitf.nitf_elements.security.NITFSecurityTags:
        """Construct a NITFSecurityTags object"""
        return sarkit._nitf.nitf_elements.security.NITFSecurityTags(
            CLAS=self.clas,
            CLSY=self.clsy,
            CODE=self.code,
            CTLH=self.ctlh,
            REL=self.rel,
            DCTP=self.dctp,
            DCDT=self.dcdt,
            DCXM=self.dcxm,
            DG=self.dg,
            DGDT=self.dgdt,
            CLTX=self.cltx,
            CATP=self.catp,
            CAUT=self.caut,
            CRSN=self.crsn,
            SRDT=self.srdt,
            CTLN=self.ctln,
        )


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
    def _from_header(cls, file_header: sarkit._nitf.nitf.NITFHeader) -> Self:
        """Construct from a NITFHeader object"""
        return cls(
            ostaid=file_header.OSTAID,
            ftitle=file_header.FTITLE,
            security=NitfSecurityFields._from_security_tags(file_header.Security),
            oname=file_header.ONAME,
            ophone=file_header.OPHONE,
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
    def _from_header(cls, image_header: sarkit._nitf.nitf.ImageSegmentHeader) -> Self:
        """Construct from a NITF ImageSegmentHeader object"""
        return cls(
            tgtid=image_header.TGTID,
            iid2=image_header.IID2,
            security=NitfSecurityFields._from_security_tags(image_header.Security),
            isorce=image_header.ISORCE,
            icom=[
                val.to_bytes().decode().rstrip() for val in image_header.Comments.values
            ],
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
    def _from_header(cls, de_header: sarkit._nitf.nitf.DataExtensionHeader) -> Self:
        """Construct from a NITF DataExtensionHeader object"""
        return cls(
            security=NitfSecurityFields._from_security_tags(de_header.Security),
            desshrp=de_header.UserHeader.DESSHRP,
            desshli=de_header.UserHeader.DESSHLI,
            desshlin=de_header.UserHeader.DESSHLIN,
            desshabs=de_header.UserHeader.DESSHABS,
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

        self._initial_offset = self._file_object.tell()
        if self._initial_offset != 0:
            raise RuntimeError(
                "seek(0) must be the start of the NITF"
            )  # this is a NITFDetails limitation

        nitf_details = sarkit._nitf.nitf.NITFDetails(self._file_object)
        image_segment_collections = [
            [
                n
                for n, imghdr in enumerate(nitf_details.img_headers)
                if imghdr.IID1.startswith("SICD")
            ]
        ]
        self._nitf_reader = sarkit._nitf.nitf.NITFReader(
            nitf_details=nitf_details,
            image_segment_collections=image_segment_collections,
        )
        des_header = sarkit._nitf.nitf.DataExtensionHeader.from_bytes(
            self._nitf_reader.nitf_details.get_des_subheader_bytes(0), 0
        )
        if not des_header.UserHeader.DESSHTN.startswith("urn:SICD"):
            raise ValueError(f"Unable to find SICD DES in {file}")

        sicd_xmltree = lxml.etree.fromstring(
            self._nitf_reader.nitf_details.get_des_bytes(0)
        ).getroottree()
        nitf_header_fields = NitfFileHeaderPart._from_header(nitf_details.nitf_header)
        nitf_image_fields = NitfImSubheaderPart._from_header(
            nitf_details.img_headers[0]
        )
        nitf_de_fields = NitfDeSubheaderPart._from_header(des_header)

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
        self._file_object.seek(self._initial_offset)
        nrows = int(self.metadata.xmltree.findtext("{*}ImageData/{*}NumRows"))
        ncols = int(self.metadata.xmltree.findtext("{*}ImageData/{*}NumCols"))
        pixel_type = self.metadata.xmltree.findtext("{*}ImageData/{*}PixelType")
        dtype = PIXEL_TYPES[pixel_type]["dtype"].newbyteorder(">")
        sicd_pixels = np.empty((nrows, ncols), dtype)
        imseg_sizes = self._nitf_reader.nitf_details.img_segment_sizes[
            self._nitf_reader.image_segment_collections
        ]
        imseg_offsets = self._nitf_reader.nitf_details.img_segment_offsets[
            self._nitf_reader.image_segment_collections
        ]
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
        _ = self._nitf_reader.read(slice(start_row, end_row), slice(start_col, end_col))
        # TODO update XML
        raise NotImplementedError()

    def done(self):
        "Indicates to the reader that the user is done with it"
        self._file_object = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.done()


def _create_des_manager(sicd_xmltree, des_fields):
    """DES Manager for SICD XML DES"""
    xmlns = lxml.etree.QName(sicd_xmltree.getroot()).namespace
    xml_helper = sicd_xml.XmlHelper(sicd_xmltree)
    now_dt = datetime.datetime.now(datetime.timezone.utc)

    icp = xml_helper.load("./{*}GeoData/{*}ImageCorners")
    desshlpg = ""
    for icp_lat, icp_lon in itertools.chain(icp, [icp[0]]):
        desshlpg += f"{icp_lat:0=+12.8f}{icp_lon:0=+13.8f}"

    deshead = sarkit._nitf.nitf_elements.des.DataExtensionHeader(
        Security=des_fields.security._as_security_tags(),
        UserHeader=sarkit._nitf.nitf_elements.des.XMLDESSubheader(
            DESSHSI=SPECIFICATION_IDENTIFIER,
            DESSHSV=VERSION_INFO[xmlns]["version"],
            DESSHSD=VERSION_INFO[xmlns]["date"],
            DESSHTN=xmlns,
            DESSHDT=now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            DESSHLPG=desshlpg,
            DESSHRP=des_fields.desshrp,
            DESSHLI=des_fields.desshli,
            DESSHLIN=des_fields.desshlin,
            DESSHABS=des_fields.desshabs,
        ),
    )
    sicd_des = sarkit._nitf.nitf.DESSubheaderManager(
        deshead, lxml.etree.tostring(sicd_xmltree)
    )
    return sicd_des


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

        self._initial_offset = self._file_object.tell()
        if self._initial_offset != 0:
            raise RuntimeError(
                "seek(0) must be the start of the NITF"
            )  # this is a NITFDetails limitation

        self._metadata = copy.deepcopy(metadata)
        sicd_xmltree = self._metadata.xmltree
        xmlns = lxml.etree.QName(sicd_xmltree.getroot()).namespace
        schema = lxml.etree.XMLSchema(file=VERSION_INFO[xmlns]["schema"])
        if not schema.validate(sicd_xmltree):
            warnings.warn(str(schema.error_log))

        xml_helper = sicd_xml.XmlHelper(sicd_xmltree)
        rows = xml_helper.load("./{*}ImageData/{*}NumRows")
        cols = xml_helper.load("./{*}ImageData/{*}NumCols")
        pixel_type = sicd_xmltree.findtext("./{*}ImageData/{*}PixelType")

        # CLEVEL and FL will be corrected...
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        header = sarkit._nitf.nitf_elements.nitf_head.NITFHeader(
            CLEVEL=3,
            OSTAID=self._metadata.file_header_part.ostaid,
            FDT=now_dt.strftime("%Y%m%d%H%M%S"),
            FTITLE=self._metadata.file_header_part.ftitle,
            Security=self._metadata.file_header_part.security._as_security_tags(),
            ONAME=self._metadata.file_header_part.oname,
            OPHONE=self._metadata.file_header_part.ophone,
            FL=0,
        )

        # Create image segments
        bits_per_element = PIXEL_TYPES[pixel_type]["bytes"] * 8 / 2
        icp = xml_helper.load("./{*}GeoData/{*}ImageCorners")

        is_size_max = 10**10 - 2  # allowable image segment size
        iloc_max = 99999
        bytes_per_row = cols * PIXEL_TYPES[pixel_type]["bytes"]
        product_size = bytes_per_row * rows
        limit_1 = int(np.floor(is_size_max / bytes_per_row))
        num_rows_limit = min(iloc_max, limit_1)

        if product_size <= is_size_max:
            image_segment_limits = [(0, rows, 0, cols)]
        else:
            image_segment_limits = []
            row_offset = 0
            while row_offset < rows:
                next_rows = min(rows, row_offset + num_rows_limit)
                image_segment_limits.append((row_offset, next_rows, 0, cols))
                row_offset = next_rows

        image_segment_collections = (tuple(range(len(image_segment_limits))),)
        image_segment_coordinates = (tuple(image_segment_limits),)
        image_managers = []
        for i, entry in enumerate(image_segment_limits):
            this_rows = entry[1] - entry[0]
            subhead = sarkit._nitf.nitf_elements.image.ImageSegmentHeader(
                IID1=f"SICD{0 if len(image_segment_limits) == 1 else i + 1:03d}",
                IDATIM=xml_helper.load("./{*}Timeline/{*}CollectStart").strftime(
                    "%Y%m%d%H%M%S"
                ),
                TGTID=self._metadata.im_subheader_part.tgtid,
                IID2=self._metadata.im_subheader_part.iid2,
                Security=self._metadata.im_subheader_part.security._as_security_tags(),
                ISORCE=self._metadata.im_subheader_part.isorce,
                NROWS=this_rows,
                NCOLS=cols,
                PVTYPE=PIXEL_TYPES[pixel_type]["pvtype"],
                IREP="NODISPLY",
                ICAT="SAR",
                ABPP=bits_per_element,
                IGEOLO=sarkit._nitf.utils._interpolate_corner_points_string(
                    np.array(entry, dtype=np.int64), rows, cols, icp
                ),
                Comments=sarkit._nitf.nitf_elements.image.ImageComments(
                    [
                        sarkit._nitf.nitf_elements.image.ImageComment(COMMENT=comment)
                        for comment in self._metadata.im_subheader_part.icom
                    ]
                ),
                IC="NC",
                NPPBH=0 if cols > 8192 else cols,
                NPPBV=0 if this_rows > 8192 else this_rows,
                NBPP=bits_per_element,
                NBPC=1,
                NBPR=1,
                IDLVL=i + 1,
                IALVL=i,
                ILOC=f"{0 if i == 0 else num_rows_limit:05d}00000",
                Bands=sarkit._nitf.nitf_elements.image.ImageBands(
                    values=[
                        sarkit._nitf.nitf_elements.image.ImageBand(ISUBCAT=entry)
                        for entry in PIXEL_TYPES[pixel_type]["subcat"]
                    ]
                ),
            )
            image_managers.append(sarkit._nitf.nitf.ImageSubheaderManager(subhead))

        sicd_des = _create_des_manager(sicd_xmltree, self._metadata.de_subheader_part)

        sicd_details = sarkit._nitf.nitf.NITFWritingDetails(
            header,
            image_managers=tuple(image_managers),
            image_segment_collections=image_segment_collections,
            image_segment_coordinates=image_segment_coordinates,
            des_managers=(sicd_des,),
        )

        self._nitf_writer = sarkit._nitf.nitf.NITFWriter(
            file_object=self._file_object,
            writing_details=sicd_details,
        )

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
        raw_array = array.view((raw_dtype, 2))
        raw_array = raw_array.astype(raw_dtype.newbyteorder(">"), copy=False)
        self._nitf_writer.write_raw(raw_array, start_indices=tuple(startarr))

    def close(self):
        """
        Flush to disk and close any opened file descriptors.

        Called automatically when used as a context manager
        """
        self._nitf_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
