"""
Functions to read and write SICD files.
"""

import copy
import dataclasses
import datetime
import itertools
import logging
import os
import warnings
from typing import Self

import jbpy
import jbpy.core
import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.sicd._xml as sicd_xml
import sarkit.wgs84
from sarkit import _iohelp

from . import _constants as sicdconst

_NOMINAL_CHUNK_SIZE = 128 * 1024 * 1024


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
        field_group: jbpy.core.Group,
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

    def _set_nitf_fields(self, prefix: str, field_group: jbpy.core.Group) -> None:
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
    def _from_header(cls, file_header: jbpy.core.FileHeader) -> Self:
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
    def _from_header(cls, image_header: jbpy.core.ImageSubheader) -> Self:
        """Construct from a NITF ImageSubheader object"""
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
    def _from_header(cls, de_header: jbpy.core.DataExtensionSubheader) -> Self:
        """Construct from a NITF DataExtensionSubheader object"""
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
    jbp : ``jbpy.Jbp``
        NITF file object

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

        >>> print(r.jbp["FileHeader"]["FTITLE"].value)
        SARkit example SICD FTITLE
    """

    def __init__(self, file):
        self._file_object = file

        self.jbp = jbpy.Jbp().load(file)

        deseg = self.jbp["DataExtensionSegments"][0]  # SICD XML must be in first DES
        if not deseg["subheader"]["DESSHF"]["DESSHTN"].value.startswith("urn:SICD"):
            raise ValueError(f"Unable to find SICD DES in {file}")

        file.seek(deseg["DESDATA"].get_offset(), os.SEEK_SET)
        sicd_xmltree = lxml.etree.fromstring(
            file.read(deseg["DESDATA"].size)
        ).getroottree()

        nitf_header_fields = NitfFileHeaderPart._from_header(self.jbp["FileHeader"])
        nitf_image_fields = NitfImSubheaderPart._from_header(
            self.jbp["ImageSegments"][0]["subheader"],
        )
        nitf_de_fields = NitfDeSubheaderPart._from_header(deseg["subheader"])

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
        return self.read_sub_image()[0]

    def read_sub_image(
        self,
        start_row: int | None = None,
        start_col: int | None = None,
        stop_row: int | None = None,
        stop_col: int | None = None,
    ) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
        """Read a 2D slice / sub-image from the file

        Parameters
        ----------
        start_row : int or None
            Lowest row index to retrieve (inclusive). If None, defaults to first row.
        start_col : int or None
            Lowest column index to retrieve (inclusive). If None, defaults to first column.
        stop_row : int or None
            Highest row index to retrieve (exclusive). If None, defaults to one after last row.
        stop_col : int or None
            Highest col index to retrieve (exclusive). If None, defaults to one after last column.

        Returns
        -------
        ndarray
            SICD sub-image array
        lxml.etree.ElementTree
            SICD XML updated to describe sub-image
        """

        file_nrows = int(self.metadata.xmltree.findtext("{*}ImageData/{*}NumRows"))
        file_ncols = int(self.metadata.xmltree.findtext("{*}ImageData/{*}NumCols"))
        pixel_type = self.metadata.xmltree.findtext("{*}ImageData/{*}PixelType")
        dtype = sicdconst.PIXEL_TYPES[pixel_type]["dtype"].newbyteorder(">")

        # Convert None and negative values to absolute indices
        start_row, stop_row, _ = slice(start_row, stop_row).indices(file_nrows)
        start_col, stop_col, _ = slice(start_col, stop_col).indices(file_ncols)

        out_nrows = stop_row - start_row
        out_ncols = stop_col - start_col

        out_shape = (out_nrows, out_ncols)
        if np.any(np.less(out_shape, 1)):
            raise ValueError(f"Invalid shape requested: ({out_shape})")

        out = np.empty(out_shape, dtype)

        out_xmltree = _update_sicd_subimage_xml(
            self.metadata.xmltree, start_row, start_col, out_nrows, out_ncols
        )

        imsegs = sorted(
            [
                imseg
                for imseg in self.jbp["ImageSegments"]
                if imseg["subheader"]["IID1"].value.startswith("SICD")
            ],
            key=lambda seg: seg["subheader"]["IID1"].value,
        )

        for imseg in imsegs:
            ic_value = imseg["subheader"]["IC"].value
            if ic_value != "NC":
                raise RuntimeError(
                    f"SICDs with Compression and/or Masking not supported. IC={ic_value}"
                )

        imseg_sizes = np.asarray([imseg["Data"].size for imseg in imsegs])
        imseg_shapes = [
            (imseg["subheader"]["NROWS"].value, imseg["subheader"]["NCOLS"].value)
            for imseg in imsegs
        ]
        imseg_offsets = np.asarray([imseg["Data"].get_offset() for imseg in imsegs])
        segment_row_boundaries = [0] + np.cumsum(
            imseg_sizes // (file_ncols * dtype.itemsize)
        ).tolist()

        for seg_idx, (seg_start_row, seg_stop_row) in enumerate(
            itertools.pairwise(segment_row_boundaries)
        ):
            if stop_row <= seg_start_row or seg_stop_row <= start_row:  # no overlap
                continue

            input_start_row = max(start_row, seg_start_row) - seg_start_row
            out_start_row = max(start_row, seg_start_row) - start_row

            input_stop_row = min(stop_row, seg_stop_row) - seg_start_row
            out_stop_row = min(stop_row, seg_stop_row) - start_row

            out_segment = out[out_start_row:out_stop_row]
            try:
                mmap = np.memmap(
                    self._file_object,
                    dtype=dtype,
                    mode="r",
                    offset=imseg_offsets[seg_idx],  # np.memmap seeks to 0 first
                    shape=imseg_shapes[seg_idx],
                )
                out_segment[...] = mmap[
                    input_start_row:input_stop_row, start_col:stop_col
                ]
            except Exception:  # mmap can raise many different exceptions
                offset_within_segment = input_start_row * file_ncols * dtype.itemsize
                self._file_object.seek(
                    imseg_offsets[seg_idx] + offset_within_segment, os.SEEK_SET
                )
                # Read contiguous rows to cut down on seek/read overhead
                num_chunks = (
                    out_segment.shape[0] * file_ncols * dtype.itemsize
                ) // _NOMINAL_CHUNK_SIZE
                num_chunks = np.clip(num_chunks, 1, out_segment.shape[0])

                for split in np.array_split(out_segment, num_chunks, axis=0):
                    split[...] = _iohelp.fromfile(
                        self._file_object, dtype, split.shape[0] * file_ncols
                    ).reshape(split.shape[0], -1)[:, start_col:stop_col]

        return out, out_xmltree

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

    bytes_per_row = bytes_per_pixel * num_cols
    product_size = bytes_per_pixel * num_rows * num_cols
    limit1 = int(np.floor(sicdconst.IS_SIZE_MAX / bytes_per_row))
    num_rows_limit = min(limit1, sicdconst.ILOC_MAX)
    if product_size <= sicdconst.IS_SIZE_MAX:
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


def jbp_from_nitf_metadata(metadata: NitfMetadata) -> jbpy.Jbp:
    """Create a Jbp object from NitfMetadata"""
    sicd_xmltree = metadata.xmltree

    xml_helper = sicd_xml.XmlHelper(sicd_xmltree)
    cols = xml_helper.load("./{*}ImageData/{*}NumCols")
    pixel_type = sicd_xmltree.findtext("./{*}ImageData/{*}PixelType")
    bits_per_element = sicdconst.PIXEL_TYPES[pixel_type]["bytes"] * 8 / 2

    num_is, seginfos = image_segment_sizing_calculations(sicd_xmltree)

    jbp = jbpy.Jbp()
    jbp["FileHeader"]["OSTAID"].value = metadata.file_header_part.ostaid
    jbp["FileHeader"]["FTITLE"].value = metadata.file_header_part.ftitle
    metadata.file_header_part.security._set_nitf_fields("FS", jbp["FileHeader"])
    jbp["FileHeader"]["ONAME"].value = metadata.file_header_part.oname
    jbp["FileHeader"]["OPHONE"].value = metadata.file_header_part.ophone
    jbp["FileHeader"]["NUMI"].value = num_is

    for idx, seginfo in enumerate(seginfos):
        subhdr = jbp["ImageSegments"][idx]["subheader"]
        if len(seginfos) > 1:
            subhdr["IID1"].value = f"SICD{idx + 1:03d}"
        else:
            subhdr["IID1"].value = "SICD000"
        subhdr["IDATIM"].value = xml_helper.load(
            "./{*}Timeline/{*}CollectStart"
        ).strftime("%Y%m%d%H%M%S")
        subhdr["TGTID"].value = metadata.im_subheader_part.tgtid
        subhdr["IID2"].value = metadata.im_subheader_part.iid2
        metadata.im_subheader_part.security._set_nitf_fields("IS", subhdr)
        subhdr["ISORCE"].value = metadata.im_subheader_part.isorce
        subhdr["NROWS"].value = seginfo.nrows
        subhdr["NCOLS"].value = cols
        subhdr["PVTYPE"].value = sicdconst.PIXEL_TYPES[pixel_type]["pvtype"]
        subhdr["IREP"].value = "NODISPLY"
        subhdr["ICAT"].value = "SAR"
        subhdr["ABPP"].value = bits_per_element
        subhdr["PJUST"].value = "R"
        subhdr["ICORDS"].value = "G"
        subhdr["IGEOLO"].value = seginfo.igeolo
        subhdr["IC"].value = "NC"
        subhdr["NICOM"].value = len(metadata.im_subheader_part.icom)
        for icomidx, icom in enumerate(metadata.im_subheader_part.icom):
            subhdr[f"ICOM{icomidx + 1}"].value = icom
        subhdr["NBANDS"].value = 2
        subhdr["ISUBCAT00001"].value = sicdconst.PIXEL_TYPES[pixel_type]["subcat"][0]
        subhdr["ISUBCAT00002"].value = sicdconst.PIXEL_TYPES[pixel_type]["subcat"][1]
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

        jbp["ImageSegments"][idx]["Data"].size = (
            # No compression, no masking, no blocking
            subhdr["NROWS"].value
            * subhdr["NCOLS"].value
            * subhdr["NBANDS"].value
            * subhdr["NBPP"].value
            // 8
        )

    sicd_xml_bytes = lxml.etree.tostring(sicd_xmltree)
    jbp["FileHeader"]["NUMDES"].value = 1
    jbp["DataExtensionSegments"][0]["DESDATA"].size = len(sicd_xml_bytes)
    _populate_de_segment(
        jbp["DataExtensionSegments"][0],
        sicd_xmltree,
        metadata.de_subheader_part,
    )

    jbp.finalize()  # compute lengths, CLEVEL, etc...
    return jbp


def _populate_de_segment(de_segment, sicd_xmltree, de_subheader_part):
    subhdr = de_segment["subheader"]
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
    if xmlns not in sicdconst.VERSION_INFO:
        logging.warning(f"Unknown SICD version: {xmlns}")
        spec_date = "0000-00-00T00:00:00Z"
        spec_version = "unknown"
    else:
        spec_date = sicdconst.VERSION_INFO[xmlns]["date"]
        spec_version = sicdconst.VERSION_INFO[xmlns]["version"]

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


class NitfWriter:
    """Write a SICD NITF

    A NitfWriter object can be used as a context manager in a ``with`` statement.

    Parameters
    ----------
    file : `file object`
        SICD NITF file to write
    metadata : NitfMetadata
        SICD NITF metadata to write (copied on construction)
    jbp_override : ``jbpy.Jbp`` or ``None``, optional
        Jbp (NITF) object to use.  If not provided, one will be created using `jbp_from_nitf_metadata`.

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

    def __init__(
        self, file, metadata: NitfMetadata, jbp_override: jbpy.Jbp | None = None
    ):
        self._file_object = file
        self._metadata = copy.deepcopy(metadata)
        self._jbp = jbp_override or jbp_from_nitf_metadata(metadata)

        sicd_xmltree = self._metadata.xmltree
        xmlns = lxml.etree.QName(sicd_xmltree.getroot()).namespace
        schema = lxml.etree.XMLSchema(file=sicdconst.VERSION_INFO[xmlns]["schema"])
        if not schema.validate(sicd_xmltree):
            warnings.warn(str(schema.error_log))

        self._jbp.finalize()  # compute lengths, CLEVEL, etc...
        self._jbp.dump(file)
        desdata = self._jbp["DataExtensionSegments"][0]["DESDATA"]

        file.seek(desdata.get_offset(), os.SEEK_SET)
        sicd_xml_bytes = lxml.etree.tostring(sicd_xmltree)
        assert desdata.size == len(sicd_xml_bytes)
        file.write(sicd_xml_bytes)

    def write_image(self, array: npt.NDArray):
        """Write pixel data to a NITF file

        Parameters
        ----------
        array : ndarray
            2D array of complex pixels

        """
        pixel_type = self._metadata.xmltree.findtext("./{*}ImageData/{*}PixelType")
        if sicdconst.PIXEL_TYPES[pixel_type]["dtype"] != array.dtype.newbyteorder("="):
            raise ValueError(
                f"Array dtype ({array.dtype}) does not match expected dtype ({sicdconst.PIXEL_TYPES[pixel_type]['dtype']}) "
                f"for PixelType={pixel_type}"
            )

        xml_helper = sicd_xml.XmlHelper(self._metadata.xmltree)
        rows = xml_helper.load("./{*}ImageData/{*}NumRows")
        cols = xml_helper.load("./{*}ImageData/{*}NumCols")
        sicd_shape = np.asarray((rows, cols))

        # require array to be full image
        if np.any(array.shape != sicd_shape):
            raise ValueError(
                f"Array shape {array.shape} does not match sicd shape {sicd_shape}."
            )

        if pixel_type == "RE32F_IM32F":
            raw_dtype = array.real.dtype
        else:
            assert array.dtype.names is not None  # placate mypy
            raw_dtype = array.dtype[array.dtype.names[0]]

        imsegs = sorted(
            [
                imseg
                for imseg in self._jbp["ImageSegments"]
                if imseg["subheader"]["IID1"].value.startswith("SICD")
            ],
            key=lambda seg: seg["subheader"]["IID1"].value,
        )
        first_rows = np.cumsum(
            [0] + [imseg["subheader"]["NROWS"].value for imseg in imsegs[:-1]]
        )
        for imseg, first_row in zip(self._jbp["ImageSegments"], first_rows):
            self._file_object.seek(imseg["Data"].get_offset(), os.SEEK_SET)

            # Could break this into blocks to reduce memory usage from byte swapping
            raw_array = array[
                first_row : first_row + imseg["subheader"]["NROWS"].value
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


def _update_sicd_subimage_xml(
    sicd_xmltree: lxml.etree.ElementTree,
    first_row: int,
    first_col: int,
    num_rows: int,
    num_cols: int,
) -> lxml.etree.ElementTree:
    """Update SICD XML to describe a sub-image

    Updates the ImageData fields as expected and the GeoData/ImageCorners
    using a straight-line projection approximation to a plane.

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree
    first_row : int
        first row of sub-image, relative to ImageData/FirstRow
    first_col : int
        first column of sub-image, relative to ImageData/FirstCol
    num_rows : int
        number of rows in sub-image
    num_cols : int
        number of columns in sub-image

    Returns
    -------
    sicd_xmltree_out : lxml.etree.ElementTree
        Updated SICD XML ElementTree
    """
    if not first_row >= 0:
        raise ValueError("first_row must be >= 0, not {first_row}")
    if not first_col >= 0:
        raise ValueError("first_col must be >= 0, not {first_col}")
    if not num_rows > 0:
        raise ValueError("num_rows must be > 0, not {num_rows}")
    if not num_cols > 0:
        raise ValueError("num_cols must be > 0, not {num_cols}")

    sicd_xmltree_out = copy.deepcopy(sicd_xmltree)
    xml_helper = sicd_xml.XmlHelper(sicd_xmltree_out)

    end_row = first_row + num_rows
    end_col = first_col + num_cols
    if end_row > xml_helper.load("./{*}ImageData/{*}NumRows"):
        raise RuntimeError("Requested sub-image goes beyond edge of input SICD")
    if end_col > xml_helper.load("./{*}ImageData/{*}NumCols"):
        raise RuntimeError("Requested sub-image goes beyond edge of input SICD")

    first_row_abs = first_row + xml_helper.load("./{*}ImageData/{*}FirstRow")
    first_col_abs = first_col + xml_helper.load("./{*}ImageData/{*}FirstCol")

    xml_helper.set("./{*}ImageData/{*}FirstRow", first_row_abs)
    xml_helper.set("./{*}ImageData/{*}FirstCol", first_col_abs)
    xml_helper.set("./{*}ImageData/{*}NumRows", num_rows)
    xml_helper.set("./{*}ImageData/{*}NumCols", num_cols)

    scp = xml_helper.load("./{*}GeoData/{*}SCP/{*}ECF")
    scp_llh = xml_helper.load("./{*}GeoData/{*}SCP/{*}LLH")
    urow = xml_helper.load("./{*}Grid/{*}Row/{*}UVectECF")
    ucol = xml_helper.load("./{*}Grid/{*}Col/{*}UVectECF")
    row_ss = xml_helper.load("./{*}Grid/{*}Row/{*}SS")
    col_ss = xml_helper.load("./{*}Grid/{*}Col/{*}SS")
    scp_row = xml_helper.load("./{*}ImageData/{*}SCPPixel/{*}Row")
    scp_col = xml_helper.load("./{*}ImageData/{*}SCPPixel/{*}Col")
    arp_scp_coa = xml_helper.load("./{*}SCPCOA/{*}ARPPos")
    varp_scp_coa = xml_helper.load("./{*}SCPCOA/{*}ARPVel")
    look = {"L": 1, "R": -1}[xml_helper.load("./{*}SCPCOA/{*}SideOfTrack")]

    ugpn = sarkit.wgs84.up(scp_llh)

    spn = look * np.cross(varp_scp_coa, scp - arp_scp_coa)
    uspn = spn / np.linalg.norm(spn)

    sf_proj = np.dot(uspn, ugpn)

    sicp_row = np.array(
        [
            first_row_abs,
            first_row_abs,
            first_row_abs + num_rows - 1,
            first_row_abs + num_rows - 1,
        ]
    )

    sicp_col = np.array(
        [
            first_col_abs,
            first_col_abs + num_cols - 1,
            first_col_abs + num_cols - 1,
            first_col_abs,
        ]
    )

    # The SICD Sub-Image Extraction document dated 2009-06-15 dictates a straight-line projection
    row_coord = (sicp_row - scp_row) * row_ss
    col_coord = (sicp_col - scp_col) * col_ss
    delta_ipp = row_coord[..., np.newaxis] * urow + col_coord[..., np.newaxis] * ucol
    dist_proj = 1 / sf_proj * np.inner(delta_ipp, ugpn)
    gpp = scp + delta_ipp - dist_proj[..., np.newaxis] * uspn
    gpp_llh = sarkit.wgs84.cartesian_to_geodetic(gpp)
    xml_helper.set("./{*}GeoData/{*}ImageCorners", gpp_llh[:, :-1])
    return sicd_xmltree_out
