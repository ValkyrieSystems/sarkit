"""
Functions to read and write SIDD files.
"""

import collections
import dataclasses
import datetime
import itertools
import logging
import os
import re
import warnings
from typing import Self

import jbpy
import jbpy.core
import lxml.etree
import numpy as np
import numpy.typing as npt

import sarkit.sicd as sksicd
import sarkit.sicd._io
import sarkit.sidd as sksidd
import sarkit.wgs84
from sarkit import _iohelp

from . import _constants as siddconst

logger = logging.getLogger(__name__)


# SICD implementation happens to match, reuse it
class NitfSecurityFields(sksicd.NitfSecurityFields):
    __doc__ = sksicd.NitfSecurityFields.__doc__


# SICD implementation happens to match, reuse it
class NitfFileHeaderPart(sksicd.NitfFileHeaderPart):
    __doc__ = sksicd.NitfFileHeaderPart.__doc__


@dataclasses.dataclass(kw_only=True)
class NitfImSubheaderPart:
    """NITF image subheader fields which are set according to a Program Specific Implementation Document

    Attributes
    ----------
    tgtid : str
        Target Identifier
    iid2 : str
        Image Identifier 2
    security : NitfSecurityFields
        Security Tags with "IS" prefix
    icom : list of str
        Image Comments
    """

    ## IS fields are applied to all segments
    tgtid: str = ""
    iid2: str = ""
    security: NitfSecurityFields
    icom: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def _from_header(cls, image_header: jbpy.core.ImageSubheader) -> Self:
        """Construct from a NITF ImageSubheader object"""
        return cls(
            tgtid=image_header["TGTID"].value,
            iid2=image_header["IID2"].value,
            security=NitfSecurityFields._from_nitf_fields("IS", image_header),
            icom=[val.value for val in image_header.find_all("ICOM\\d+")],
        )

    def __post_init__(self):
        if isinstance(self.security, dict):
            self.security = NitfSecurityFields(**self.security)


# SICD implementation happens to match, reuse it
class NitfDeSubheaderPart(sksicd.NitfDeSubheaderPart):
    __doc__ = sksicd.NitfDeSubheaderPart.__doc__


@dataclasses.dataclass
class NitfLegendMetadata:
    """SIDD NITF legend metadata"""

    def __post_init__(self):
        raise NotImplementedError()


@dataclasses.dataclass(kw_only=True)
class NitfProductImageMetadata:
    """SIDD NITF product image metadata

    Attributes
    ----------
    xmltree : lxml.etree.ElementTree
        SIDD product metadata XML ElementTree
    im_subheader_part : NitfImSubheaderPart
        NITF Image Segment Header fields which can be set
    de_subheader_part : :NitfDeSubheaderPart
        NITF DE Segment Header fields which can be set
    legends : list of NitfLegendMetadata
        Metadata for legend(s) attached to this image
    lookup_table : ndarray or None
        Mapping from raw to display pixel values. Required for "LU" pixel types.
        Table must be 256 elements.
        For MONO8LU, table must have dtype of np.uint8 or np.uint16.
        For RGB8LU, table must have dtype of ``PIXEL_TYPES["RGB24I"]["dtype"]``.
    """

    xmltree: lxml.etree.ElementTree
    im_subheader_part: NitfImSubheaderPart
    de_subheader_part: NitfDeSubheaderPart
    legends: list[NitfLegendMetadata] = dataclasses.field(default_factory=list)
    lookup_table: npt.NDArray | None = None

    def __post_init__(self):
        _validate_xml(self.xmltree)

        xml_helper = sksidd.XmlHelper(self.xmltree)
        pixel_type = xml_helper.load("./{*}Display/{*}PixelType")

        if self.lookup_table is not None:
            lookup_table = np.asarray(self.lookup_table)
            if lookup_table.shape != (256,):
                raise ValueError("lookup_table must contain exactly 256 elements")
            lut_dtype = lookup_table.dtype
        else:
            lut_dtype = None

        mismatch = False
        if ("LU" in pixel_type) != (lut_dtype is not None):
            mismatch = True
        elif pixel_type == "MONO8LU" and lut_dtype not in (np.uint8, np.uint16):
            mismatch = True
        elif (
            pixel_type == "RGB8LU"
            and lut_dtype != siddconst.PIXEL_TYPES["RGB24I"]["dtype"]
        ):
            mismatch = True

        if mismatch:
            raise RuntimeError(
                f"lookup_table type mismatch.  {pixel_type=}  {lut_dtype=}"
            )

        if isinstance(self.im_subheader_part, dict):
            self.im_subheader_part = NitfImSubheaderPart(**self.im_subheader_part)
            self.de_subheader_part = NitfDeSubheaderPart(**self.de_subheader_part)


@dataclasses.dataclass
class NitfDedMetadata:
    """SIDD NITF DED metadata"""

    def __post_init__(self):
        raise NotImplementedError()


@dataclasses.dataclass
class NitfProductSupportXmlMetadata:
    """SIDD NITF product support XML metadata

    Attributes
    ----------
    xmltree : lxml.etree.ElementTree
        SIDD product support XML
    de_subheader_part : NitfDeSubheaderPart
        NITF DES subheader fields which can be set
    """

    xmltree: lxml.etree.ElementTree
    de_subheader_part: NitfDeSubheaderPart

    def __post_init__(self):
        if isinstance(self.de_subheader_part, dict):
            self.de_subheader_part = NitfDeSubheaderPart(**self.de_subheader_part)


@dataclasses.dataclass
class NitfSicdXmlMetadata:
    """SIDD NITF SICD XML metadata

    Attributes
    ----------
    xmltree : lxml.etree.ElementTree
        SICD XML
    de_subheader_part : NitfDeSubheaderPart
        NITF DES subheader fields which can be set
    """

    xmltree: lxml.etree.ElementTree
    de_subheader_part: sksicd.NitfDeSubheaderPart

    def __post_init__(self):
        if isinstance(self.de_subheader_part, dict):
            self.de_subheader_part = sksicd.NitfDeSubheaderPart(
                **self.de_subheader_part
            )


@dataclasses.dataclass(kw_only=True)
class NitfMetadata:
    """Settable SIDD NITF metadata

    Attributes
    ----------
    file_header_part : NitfFileHeaderPart
        NITF file header fields which can be set
    images : list of NitfProductImageMetadata
        Settable metadata for the product image(s)
    ded : NitfDedMetadata or None
        Settable metadata for the Digital Elevation Data
    product_support_xmls : list of NitfProductSupportXmlMetadata
        Settable metadata for the product support XML(s)
    sicd_xmls : list of NitfSicdXmlMetadata
        Settable metadata for the SICD XML(s)
    """

    file_header_part: NitfFileHeaderPart
    images: list[NitfProductImageMetadata] = dataclasses.field(default_factory=list)
    ded: NitfDedMetadata | None = None
    product_support_xmls: list[NitfProductSupportXmlMetadata] = dataclasses.field(
        default_factory=list
    )
    sicd_xmls: list[NitfSicdXmlMetadata] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.file_header_part, dict):
            self.file_header_part = NitfFileHeaderPart(**self.file_header_part)


class NitfReader:
    """Read a SIDD NITF

    A NitfReader object should be used as a context manager in a ``with`` statement.
    Attributes, but not methods, can be safely accessed outside of the context manager's context.

    Parameters
    ----------
    file : `file object`
        SIDD NITF file to read

    Attributes
    ----------
    metadata : NitfMetadata
        SIDD NITF metadata
    jbp : ``jbpy.Jbp``
        NITF file object

    See Also
    --------
    NitfWriter

    Examples
    --------

    .. testsetup:: sidd_io

        import lxml.etree
        import numpy as np

        import sarkit.sidd as sksidd

        sidd_xml = lxml.etree.parse("data/example-sidd-3.0.0.xml")
        sec = {"security": {"clas": "U"}}
        meta = sksidd.NitfMetadata(
            file_header_part={"ostaid": "sksidd stn", "ftitle": "sarkit example", **sec},
            images=[
                sksidd.NitfProductImageMetadata(
                    xmltree=sidd_xml,
                    im_subheader_part=sec,
                    de_subheader_part=sec,
                )
            ],
        )
        img_to_write = np.zeros(
            sksidd.XmlHelper(sidd_xml).load("{*}Measurement/{*}PixelFootprint"),
            dtype=sksidd.PIXEL_TYPES[sidd_xml.findtext("{*}Display/{*}PixelType")]["dtype"],
        )
        file = pathlib.Path(tmpdir.name) / "foo"
        with file.open("wb") as f, sksidd.NitfWriter(f, meta) as w:
            w.write_image(0, img_to_write)

    .. doctest:: sidd_io

        >>> import sarkit.sidd as sksidd
        >>> with file.open("rb") as f, sksidd.NitfReader(f) as r:
        ...     img = r.read_image(0)

        >>> print(r.metadata.images[0].xmltree.getroot().tag)
        {urn:SIDD:3.0.0}SIDD

        >>> print(r.metadata.file_header_part.ftitle)
        sarkit example

        >>> print(r.jbp["FileHeader"]["FTITLE"].value)
        sarkit example
    """

    def __init__(self, file):
        self._file_object = file

        self.jbp = jbpy.Jbp().load(file)

        im_segments = {}
        for imseg_index, imseg in enumerate(self.jbp["ImageSegments"]):
            img_header = imseg["subheader"]
            if img_header["IID1"].value.startswith("SIDD"):
                if img_header["ICAT"].value == "SAR":
                    image_number = int(img_header["IID1"].value[4:7]) - 1
                    im_segments.setdefault(image_number, [])
                    im_segments[image_number].append(imseg_index)
                else:
                    raise NotImplementedError("Non SAR images not supported")  # TODO
            elif img_header["IID1"].value.startswith("DED"):
                raise NotImplementedError("DED not supported")  # TODO

        image_segment_collections = {}
        for idx, imseg in enumerate(self.jbp["ImageSegments"]):
            imghdr = imseg["subheader"]
            if not imghdr["IID1"].value.startswith("SIDD"):
                continue
            image_num = int(imghdr["IID1"].value[4:7]) - 1
            image_segment_collections.setdefault(image_num, [])
            image_segment_collections[image_num].append(idx)

        file_header_part = NitfFileHeaderPart._from_header(self.jbp["FileHeader"])
        self.metadata = NitfMetadata(file_header_part=file_header_part)

        image_number = 0
        for idx, deseg in enumerate(self.jbp["DataExtensionSegments"]):
            des_header = deseg["subheader"]
            if des_header["DESID"].value == "XML_DATA_CONTENT":
                file.seek(deseg["DESDATA"].get_offset(), os.SEEK_SET)
                try:
                    xmltree = lxml.etree.fromstring(
                        file.read(deseg["DESDATA"].size)
                    ).getroottree()
                except lxml.etree.XMLSyntaxError:
                    logger.error(f"Failed to parse DES {idx} as XML")
                    continue

                if "SIDD" in xmltree.getroot().tag:
                    de_subheader_part = NitfDeSubheaderPart._from_header(des_header)
                    if len(self.metadata.images) < len(image_segment_collections):
                        # user settable fields should be the same for all image segments
                        im_idx = im_segments[image_number][0]
                        im_subhdr = self.jbp["ImageSegments"][im_idx]["subheader"]
                        im_subhdeader_part = NitfImSubheaderPart._from_header(im_subhdr)
                        pixel_type = xmltree.findtext("./{*}Display/{*}PixelType")
                        lookup_table = None
                        if "LU" in pixel_type:
                            assert im_subhdr["NBANDS"].value == 1
                            assert im_subhdr["NELUT00001"].value == 256

                        if pixel_type == "RGB8LU":
                            assert im_subhdr["NLUTS00001"].value == 3
                            lookup_table = np.empty(
                                256, siddconst.PIXEL_TYPES["RGB24I"]["dtype"]
                            )
                            lookup_table["R"] = np.frombuffer(
                                im_subhdr["LUTD000011"].value, dtype=np.uint8
                            )
                            lookup_table["G"] = np.frombuffer(
                                im_subhdr["LUTD000012"].value, dtype=np.uint8
                            )
                            lookup_table["B"] = np.frombuffer(
                                im_subhdr["LUTD000013"].value, dtype=np.uint8
                            )
                        elif pixel_type == "MONO8LU":
                            msbs = np.frombuffer(
                                im_subhdr["LUTD000011"].value, dtype=np.uint8
                            )
                            if im_subhdr["NLUTS00001"].value == 1:
                                lookup_table = msbs
                            elif im_subhdr["NLUTS00001"].value == 2:
                                lsbs = np.frombuffer(
                                    im_subhdr["LUTD000012"].value, dtype=np.uint8
                                )
                                lookup_table = (msbs.astype(np.uint16) << 8) + lsbs
                            else:
                                raise ValueError(
                                    f"Unsupported NLUTS={im_subhdr['NLUTS00001'].value}"
                                )
                        self.metadata.images.append(
                            NitfProductImageMetadata(
                                xmltree=xmltree,
                                im_subheader_part=im_subhdeader_part,
                                de_subheader_part=de_subheader_part,
                                lookup_table=lookup_table,
                            )
                        )
                        image_number += 1
                    else:
                        # No matching product image, treat it as a product support XML
                        self.metadata.product_support_xmls.append(
                            NitfProductSupportXmlMetadata(xmltree, de_subheader_part)
                        )
                elif "SICD" in xmltree.getroot().tag:
                    de_subheader_part = sksicd.NitfDeSubheaderPart._from_header(
                        des_header
                    )
                    self.metadata.sicd_xmls.append(
                        NitfSicdXmlMetadata(xmltree, de_subheader_part)
                    )
                else:
                    de_subheader_part = NitfDeSubheaderPart._from_header(des_header)
                    self.metadata.product_support_xmls.append(
                        NitfProductSupportXmlMetadata(xmltree, de_subheader_part)
                    )

        # TODO Legends
        # TODO DED
        assert not any(x.legends for x in self.metadata.images)
        assert not self.metadata.ded

    def read_image(self, image_number: int) -> npt.NDArray:
        """Read the entire pixel array

        Parameters
        ----------
        image_number : int
            index of SIDD Product image to read

        Returns
        -------
        ndarray
            SIDD image array
        """
        self._file_object.seek(0)
        xml_helper = sksidd.XmlHelper(self.metadata.images[image_number].xmltree)
        shape = xml_helper.load("{*}Measurement/{*}PixelFootprint")
        dtype = siddconst.PIXEL_TYPES[xml_helper.load("{*}Display/{*}PixelType")][
            "dtype"
        ].newbyteorder(">")

        imseg_indices = product_image_segment_mapping(self.jbp)[
            f"SIDD{image_number + 1:03d}"
        ]
        imsegs = [self.jbp["ImageSegments"][idx] for idx in imseg_indices]

        image_pixels = np.empty(shape, dtype)
        imseg_sizes = np.asarray([imseg["Data"].size for imseg in imsegs])
        imseg_offsets = np.asarray([imseg["Data"].get_offset() for imseg in imsegs])
        splits = np.cumsum(imseg_sizes // (shape[-1] * dtype.itemsize))[:-1]
        for split, offset in zip(
            np.array_split(image_pixels, splits, axis=0), imseg_offsets
        ):
            self._file_object.seek(offset)
            split[...] = _iohelp.fromfile(
                self._file_object, dtype, np.prod(split.shape)
            ).reshape(split.shape)

        return image_pixels

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return


def jbp_from_nitf_metadata(metadata: NitfMetadata) -> jbpy.Jbp:
    """Create a Jbp object from NitfMetadata"""

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    jbp = jbpy.Jbp()
    jbp["FileHeader"]["OSTAID"].value = metadata.file_header_part.ostaid
    jbp["FileHeader"]["FTITLE"].value = metadata.file_header_part.ftitle
    metadata.file_header_part.security._set_nitf_fields("FS", jbp["FileHeader"])
    jbp["FileHeader"]["ONAME"].value = metadata.file_header_part.oname
    jbp["FileHeader"]["OPHONE"].value = metadata.file_header_part.ophone

    _, _, seginfos = segmentation_algorithm((img.xmltree for img in metadata.images))
    jbp["FileHeader"]["NUMI"].value = len(seginfos)  # TODO + num DES + num LEG

    for idx, seginfo in enumerate(seginfos):
        subhdr = jbp["ImageSegments"][idx]["subheader"]
        image_num = int(seginfo.iid1[4:7]) - 1

        imageinfo = metadata.images[image_num]
        xml_helper = sarkit.sidd._xml.XmlHelper(imageinfo.xmltree)
        pixel_type = xml_helper.load("./{*}Display/{*}PixelType")
        pixel_info = siddconst.PIXEL_TYPES[pixel_type]

        icp = xml_helper.load("./{*}GeoData/{*}ImageCorners")

        subhdr["IID1"].value = seginfo.iid1
        subhdr["IDATIM"].value = xml_helper.load(
            "./{*}ExploitationFeatures/{*}Collection/{*}Information/{*}CollectionDateTime"
        ).strftime("%Y%m%d%H%M%S")
        subhdr["TGTID"].value = imageinfo.im_subheader_part.tgtid
        subhdr["IID2"].value = imageinfo.im_subheader_part.iid2
        imageinfo.im_subheader_part.security._set_nitf_fields("IS", subhdr)
        subhdr["ISORCE"].value = xml_helper.load(
            "./{*}ExploitationFeatures/{*}Collection/{*}Information/{*}SensorName"
        )
        subhdr["NROWS"].value = seginfo.nrows
        subhdr["NCOLS"].value = seginfo.ncols
        subhdr["PVTYPE"].value = "INT"
        subhdr["IREP"].value = pixel_info["IREP"]
        subhdr["ICAT"].value = "SAR"
        subhdr["ABPP"].value = pixel_info["NBPP"]
        subhdr["PJUST"].value = "R"
        subhdr["ICORDS"].value = "G"
        subhdr["IGEOLO"].value = seginfo.igeolo
        subhdr["IC"].value = "NC"
        subhdr["NICOM"].value = len(imageinfo.im_subheader_part.icom)
        for icomidx, icom in enumerate(imageinfo.im_subheader_part.icom):
            subhdr[f"ICOM{icomidx + 1}"].value = icom
        subhdr["NBANDS"].value = len(pixel_info["IREPBANDn"])
        for bandnum, irepband in enumerate(pixel_info["IREPBANDn"]):
            subhdr[f"IREPBAND{bandnum + 1:05d}"].value = irepband

        if "LU" in pixel_type:
            if imageinfo.lookup_table is None:
                raise ValueError(f"lookup table must be set for PixelType={pixel_type}")

            if pixel_type == "RGB8LU":
                subhdr["NLUTS00001"].value = 3
                subhdr["NELUT00001"].value = 256
                subhdr["LUTD000011"].value = imageinfo.lookup_table["R"].tobytes()
                subhdr["LUTD000012"].value = imageinfo.lookup_table["G"].tobytes()
                subhdr["LUTD000013"].value = imageinfo.lookup_table["B"].tobytes()
            elif pixel_type == "MONO8LU":
                if imageinfo.lookup_table.dtype == np.uint8:
                    subhdr["NLUTS00001"].value = 1
                    subhdr["NELUT00001"].value = 256
                    subhdr["LUTD000011"].value = imageinfo.lookup_table.tobytes()
                elif imageinfo.lookup_table.dtype == np.uint16:
                    subhdr["NLUTS00001"].value = 2
                    subhdr["NELUT00001"].value = 256
                    subhdr["LUTD000011"].value = (
                        (imageinfo.lookup_table >> 8).astype(np.uint8).tobytes()
                    )  # MSB
                    subhdr["LUTD000012"].value = (
                        (imageinfo.lookup_table & 0xFF).astype(np.uint8).tobytes()
                    )  # LSB

        subhdr["IMODE"].value = pixel_info["IMODE"]
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

        subhdr["NBPP"].value = pixel_info["NBPP"]
        subhdr["IDLVL"].value = seginfo.idlvl
        subhdr["IALVL"].value = seginfo.ialvl
        subhdr["ILOC"].value = (int(seginfo.iloc[:5]), int(seginfo.iloc[5:]))
        subhdr["IMAG"].value = "1.0 "

        jbp["ImageSegments"][idx]["Data"].size = (
            # No compression, no masking, no blocking
            subhdr["NROWS"].value
            * subhdr["NCOLS"].value
            * subhdr["NBANDS"].value
            * subhdr["NBPP"].value
            // 8
        )

    # TODO add image_managers for legends
    assert not any(x.legends for x in metadata.images)
    # TODO add image_managers for DED
    assert not metadata.ded

    # DE Segments
    jbp["FileHeader"]["NUMDES"].value = (
        len(metadata.images)
        + len(metadata.product_support_xmls)
        + len(metadata.sicd_xmls)
    )

    desidx = 0
    to_write = []
    for imageinfo in metadata.images:
        xmlns = lxml.etree.QName(imageinfo.xmltree.getroot()).namespace
        xml_helper = sksidd.XmlHelper(imageinfo.xmltree)

        deseg = jbp["DataExtensionSegments"][desidx]
        subhdr = deseg["subheader"]
        subhdr["DESID"].value = "XML_DATA_CONTENT"
        subhdr["DESVER"].value = 1
        imageinfo.de_subheader_part.security._set_nitf_fields("DES", subhdr)
        subhdr["DESSHL"].value = 773
        subhdr["DESSHF"]["DESCRC"].value = 99999
        subhdr["DESSHF"]["DESSHFT"].value = "XML"
        subhdr["DESSHF"]["DESSHDT"].value = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        subhdr["DESSHF"]["DESSHRP"].value = imageinfo.de_subheader_part.desshrp
        subhdr["DESSHF"]["DESSHSI"].value = siddconst.SPECIFICATION_IDENTIFIER
        subhdr["DESSHF"]["DESSHSV"].value = siddconst.VERSION_INFO[xmlns]["version"]
        subhdr["DESSHF"]["DESSHSD"].value = siddconst.VERSION_INFO[xmlns]["date"]
        subhdr["DESSHF"]["DESSHTN"].value = xmlns

        icp = xml_helper.load("./{*}GeoData/{*}ImageCorners")
        desshlpg = ""
        for icp_lat, icp_lon in itertools.chain(icp, [icp[0]]):
            desshlpg += f"{icp_lat:0=+12.8f}{icp_lon:0=+13.8f}"
        subhdr["DESSHF"]["DESSHLPG"].value = desshlpg
        subhdr["DESSHF"]["DESSHLI"].value = imageinfo.de_subheader_part.desshli
        subhdr["DESSHF"]["DESSHLIN"].value = imageinfo.de_subheader_part.desshlin
        subhdr["DESSHF"]["DESSHABS"].value = imageinfo.de_subheader_part.desshabs

        xml_bytes = lxml.etree.tostring(imageinfo.xmltree)
        deseg["DESDATA"].size = len(xml_bytes)
        to_write.append((deseg["DESDATA"].get_offset(), xml_bytes))

        desidx += 1

    # Product Support XML DES
    for prodinfo in metadata.product_support_xmls:
        deseg = jbp["DataExtensionSegments"][desidx]
        subhdr = deseg["subheader"]
        sidd_uh = jbp["DataExtensionSegments"][0]["subheader"]["DESSHF"]

        xmlns = lxml.etree.QName(prodinfo.xmltree.getroot()).namespace or ""

        subhdr["DESID"].value = "XML_DATA_CONTENT"
        subhdr["DESVER"].value = 1
        prodinfo.de_subheader_part.security._set_nitf_fields("DES", subhdr)
        subhdr["DESSHL"].value = 773
        subhdr["DESSHF"]["DESCRC"].value = 99999
        subhdr["DESSHF"]["DESSHFT"].value = "XML"
        subhdr["DESSHF"]["DESSHDT"].value = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        subhdr["DESSHF"]["DESSHRP"].value = prodinfo.de_subheader_part.desshrp
        subhdr["DESSHF"]["DESSHSI"].value = sidd_uh["DESSHSI"].value
        subhdr["DESSHF"]["DESSHSV"].value = "v" + sidd_uh["DESSHSV"].value
        subhdr["DESSHF"]["DESSHSD"].value = sidd_uh["DESSHSD"].value
        subhdr["DESSHF"]["DESSHTN"].value = xmlns
        subhdr["DESSHF"]["DESSHLPG"].value = ""
        subhdr["DESSHF"]["DESSHLI"].value = prodinfo.de_subheader_part.desshli
        subhdr["DESSHF"]["DESSHLIN"].value = prodinfo.de_subheader_part.desshlin
        subhdr["DESSHF"]["DESSHABS"].value = prodinfo.de_subheader_part.desshabs

        xml_bytes = lxml.etree.tostring(prodinfo.xmltree)
        deseg["DESDATA"].size = len(xml_bytes)

        desidx += 1

    # SICD XML DES
    for sicd_xml_info in metadata.sicd_xmls:
        deseg = jbp["DataExtensionSegments"][desidx]
        sarkit.sicd._io._populate_de_segment(
            deseg, sicd_xml_info.xmltree, sicd_xml_info.de_subheader_part
        )

        xml_bytes = lxml.etree.tostring(sicd_xml_info.xmltree)
        deseg["DESDATA"].size = len(xml_bytes)

        desidx += 1

    jbp.finalize()
    return jbp


def _is_sidd_product_image_segment(segment):
    if segment["subheader"]["ICAT"].value != "SAR":
        return False

    iid1 = segment["subheader"]["IID1"].value
    if re.fullmatch(r"SIDD\d{6}", iid1):
        product_image_number = int(iid1[4:7])
        segment_of_image = int(iid1[7:])
        if product_image_number >= 1 and segment_of_image >= 1:
            return True

    return False


def product_image_segment_mapping(jbp: jbpy.Jbp) -> dict[str, list[int]]:
    """Determine which JBP segments comprise each SIDD product image

    Parameters
    ----------
    jbp : ``jbpy.Jbp``
        JBP/NITF object

    Returns
    -------
    dict
        Mapping of partial SIDD IID1 identifier to ImageSegment indices.

    """
    mapping: dict[str, list[int]] = {}
    sorted_by_iid1 = sorted(
        enumerate(jbp["ImageSegments"]),
        key=lambda pair: pair[1]["subheader"]["IID1"].value,
    )
    for im_idx, imseg in sorted_by_iid1:
        iid1 = imseg["subheader"]["IID1"].value
        if _is_sidd_product_image_segment(imseg):
            name = iid1[:7]
            mapping.setdefault(name, []).append(im_idx)
    return mapping


class NitfWriter:
    """Write a SIDD NITF

    A NitfWriter object should be used as a context manager in a ``with`` statement.

    Parameters
    ----------
    file : `file object`
        SIDD NITF file to write
    metadata : NitfMetadata
        SIDD NITF metadata to write (copied on construction)
    jbp_override : ``jbpy.Jbp`` or ``None``, optional
        Jbp (NITF) object to use.  If not provided, one will be created using `jbp_from_nitf_metadata`.

    See Also
    --------
    NitfReader

    Examples
    --------
    Write a SIDD NITF with a single product image

    .. doctest::

        >>> import sarkit.sidd as sksidd

    Build the product image description and pixels

    .. doctest::

        >>> import lxml.etree
        >>> sidd_xml = lxml.etree.parse("data/example-sidd-3.0.0.xml")

        >>> sec = sksidd.NitfSecurityFields(clas="U")
        >>> img_meta = sksidd.NitfProductImageMetadata(
        ...     xmltree=sidd_xml,
        ...     im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
        ...     de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
        ... )

        >>> import numpy as np
        >>> img_to_write = np.zeros(
        ...     sksidd.XmlHelper(sidd_xml).load("{*}Measurement/{*}PixelFootprint"),
        ...     dtype=sksidd.PIXEL_TYPES[sidd_xml.findtext("{*}Display/{*}PixelType")]["dtype"],
        ... )

    Place the product image in a NITF metadata object

    .. doctest::

        >>> meta = sksidd.NitfMetadata(
        ...     file_header_part=sksidd.NitfFileHeaderPart(ostaid="my station", security=sec),
        ...     images=[img_meta],
        ... )

    Write the SIDD NITF to a file

    .. doctest::

        >>> from tempfile import NamedTemporaryFile
        >>> outfile = NamedTemporaryFile()
        >>> with sksidd.NitfWriter(outfile, meta) as w:
        ...     w.write_image(0, img_to_write)
    """

    def __init__(
        self, file, metadata: NitfMetadata, jbp_override: jbpy.Jbp | None = None
    ):
        self._file = file
        self._metadata = metadata
        self._jbp = jbp_override or jbp_from_nitf_metadata(metadata)
        self._images_written: set[int] = set()

        self._jbp.finalize()
        self._jbp.dump(file)

        to_write = []
        desidx = 0
        for imageinfo in metadata.images:
            deseg = self._jbp["DataExtensionSegments"][desidx]
            xml_bytes = lxml.etree.tostring(imageinfo.xmltree)
            assert deseg["DESDATA"].size == len(xml_bytes)
            to_write.append((deseg["DESDATA"].get_offset(), xml_bytes))

            desidx += 1

        for prodinfo in metadata.product_support_xmls:
            deseg = self._jbp["DataExtensionSegments"][desidx]
            xml_bytes = lxml.etree.tostring(prodinfo.xmltree)
            assert deseg["DESDATA"].size == len(xml_bytes)
            to_write.append((deseg["DESDATA"].get_offset(), xml_bytes))

            desidx += 1

        for sicd_xml_info in metadata.sicd_xmls:
            deseg = self._jbp["DataExtensionSegments"][desidx]
            xml_bytes = lxml.etree.tostring(sicd_xml_info.xmltree)
            assert deseg["DESDATA"].size == len(xml_bytes)
            to_write.append((deseg["DESDATA"].get_offset(), xml_bytes))

            desidx += 1

        for offset, xml_bytes in to_write:
            file.seek(offset, os.SEEK_SET)
            file.write(xml_bytes)

    def _product_image_info(self, image_number):
        shape = np.array([0, 0], dtype=np.int64)
        imseg_indices = product_image_segment_mapping(self._jbp)[
            f"SIDD{image_number + 1:03d}"
        ]
        imsegs = [self._jbp["ImageSegments"][idx] for idx in imseg_indices]

        shape[0] = sum(imseg["subheader"]["NROWS"].value for imseg in imsegs)
        shape[1] = imsegs[0]["subheader"]["NCOLS"].value

        irep = imsegs[0]["subheader"]["IREP"].value
        irepband0 = imsegs[0]["subheader"]["IREPBAND00001"].value
        nbands = imsegs[0]["subheader"]["NBANDS"].value
        abpp = imsegs[0]["subheader"]["ABPP"].value
        pixel_type = {
            ("MONO", "M", 1, 8): "MONO8I",
            ("MONO", "LU", 1, 8): "MONO8LU",
            ("MONO", "M", 1, 16): "MONO16I",
            ("RGB/LUT", "LU", 1, 8): "RGB8LU",
            ("RGB", "R", 3, 8): "RGB24I",
        }[(irep, irepband0, nbands, abpp)]

        return shape, pixel_type, imsegs

    def write_image(
        self,
        image_number: int,
        array: npt.NDArray,
    ):
        """Write product pixel data to a NITF file

        Parameters
        ----------
        image_number : int
            index of SIDD Product image to write
        array : ndarray
            2D array of pixels
        """
        shape, pixel_type, imsegs = self._product_image_info(image_number)

        # require array to be full image
        if np.any(array.shape != shape):
            raise ValueError(
                f"Array shape {array.shape} does not match SIDD shape {shape}."
            )

        first_rows = np.cumsum(
            [0] + [imseg["subheader"]["NROWS"].value for imseg in imsegs[:-1]]
        )

        if pixel_type == "RGB24I":
            assert array.dtype.names is not None  # placate mypy
            raw_dtype = array.dtype[array.dtype.names[0]]
            input_array = array.view((raw_dtype, 3))
        else:
            raw_dtype = siddconst.PIXEL_TYPES[pixel_type]["dtype"].newbyteorder(">")
            input_array = array

        for imseg, first_row in zip(imsegs, first_rows):
            self._file.seek(imseg["Data"].get_offset(), os.SEEK_SET)

            # Could break this into blocks to reduce memory usage from byte swapping
            raw_array = input_array[
                first_row : first_row + imseg["subheader"]["NROWS"].value
            ]
            raw_array = raw_array.astype(raw_dtype.newbyteorder(">"), copy=False)
            raw_array.tofile(self._file)

        self._images_written.add(image_number)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        images_expected = set(range(len(product_image_segment_mapping(self._jbp))))
        images_missing = images_expected - self._images_written
        if images_missing:
            logger.warning(
                f"SIDD Writer closed without writing all images. Missing: {images_missing}"
            )
        # TODO check legends, DED
        return


@dataclasses.dataclass(kw_only=True)
class SegmentationImhdr:
    """Per segment values computed by the SIDD Segmentation Algorithm"""

    idlvl: int
    ialvl: int
    iloc: str
    iid1: str
    nrows: int
    ncols: int
    igeolo: str


def segmentation_algorithm(
    sidd_xmltrees: collections.abc.Iterable[lxml.etree.ElementTree],
) -> tuple[int, list[int], list[SegmentationImhdr]]:
    """Implementation of section 2.4.2.1 Segmentation Algorithm and 2.4.2.2 Image Segment Corner Coordinate Parameters

    Parameters
    ----------
    sidd_xmltrees : iterable of lxml.etree.ElementTree
        SIDD XML Metadata instances

    Returns
    -------
    fhdr_numi: int
        Number of NITF image segments
    fhdr_li: list of int
        Length of each NITF image segment
    imhdr: list of SegmentationImhdr
        Image Segment subheader information
    """
    z = 0
    fhdr_numi = 0
    fhdr_li = []
    seginfos = []

    for k, sidd_xmltree in enumerate(sidd_xmltrees):
        xml_helper = sksidd.XmlHelper(sidd_xmltree)
        pixel_info = siddconst.PIXEL_TYPES[xml_helper.load("./{*}Display/{*}PixelType")]
        num_rows_k = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Row")
        num_cols_k = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Col")

        pcc = xml_helper.load(
            "./{*}GeoData/{*}ImageCorners"
        )  # Document says /SIDD/GeographicAndTarget/GeogrpahicCoverage/Footprint, but that was renamed in v2.0

        bytes_per_pixel = pixel_info[
            "dtype"
        ].itemsize  # Document says NBANDS, but that doesn't work for 16bit
        bytes_per_row = (
            bytes_per_pixel * num_cols_k
        )  # Document says NumRows(k), but that doesn't make sense
        num_rows_limit_k = min(siddconst.LI_MAX // bytes_per_row, siddconst.ILOC_MAX)

        product_size = bytes_per_pixel * num_rows_k * num_cols_k
        if product_size <= siddconst.LI_MAX:
            z += 1
            fhdr_numi += 1
            fhdr_li.append(product_size)
            seginfos.append(
                SegmentationImhdr(
                    idlvl=z,
                    ialvl=0,
                    iloc="0000000000",
                    iid1=f"SIDD{k + 1:03d}001",  # Document says 'm', but there is no m variable
                    nrows=num_rows_k,
                    ncols=num_cols_k,
                    igeolo=sarkit.sicd._io._format_igeolo(pcc),
                )
            )
        else:
            num_seg_per_image_k = int(np.ceil(num_rows_k / num_rows_limit_k))
            z += 1
            fhdr_numi += num_seg_per_image_k
            fhdr_li.append(bytes_per_pixel * num_rows_limit_k * num_cols_k)
            this_image_seginfos = []
            this_image_seginfos.append(
                SegmentationImhdr(
                    idlvl=z,
                    ialvl=0,
                    iloc="0000000000",
                    iid1=f"SIDD{k + 1:03d}001",  # Document says 'm', but there is no m variable
                    nrows=num_rows_limit_k,
                    ncols=num_cols_k,
                    igeolo="",
                )
            )
            for n in range(1, num_seg_per_image_k - 1):
                z += 1
                fhdr_li.append(bytes_per_pixel * num_rows_limit_k * num_cols_k)
                this_image_seginfos.append(
                    SegmentationImhdr(
                        idlvl=z,
                        ialvl=z - 1,
                        iloc=f"{num_rows_limit_k:05d}00000",
                        iid1=f"SIDD{k + 1:03d}{n + 1:03d}",
                        nrows=num_rows_limit_k,
                        ncols=num_cols_k,
                        igeolo="",
                    )
                )
            z += 1
            last_seg_rows = num_rows_k - (num_seg_per_image_k - 1) * num_rows_limit_k
            fhdr_li.append(bytes_per_pixel * last_seg_rows * num_cols_k)
            this_image_seginfos.append(
                SegmentationImhdr(
                    idlvl=z,
                    ialvl=z - 1,
                    iloc=f"{num_rows_limit_k:05d}00000",  # Document says "lastSegRows", but we need the number of rows in the previous IS
                    iid1=f"SIDD{k + 1:03d}{num_seg_per_image_k:03d}",
                    nrows=last_seg_rows,
                    ncols=num_cols_k,
                    igeolo="",
                )
            )
            seginfos.extend(this_image_seginfos)

            pcc_ecef = sarkit.wgs84.geodetic_to_cartesian(
                np.hstack((pcc, [[0], [0], [0], [0]]))
            )
            for geo_z, seginfo in enumerate(this_image_seginfos):
                wgt1 = geo_z * num_rows_limit_k / num_rows_k
                wgt2 = 1 - wgt1
                wgt3 = (geo_z * num_rows_limit_k + seginfo.nrows) / num_rows_k
                wgt4 = 1 - wgt3
                iscc_ecef = [
                    wgt2 * pcc_ecef[0] + wgt1 * pcc_ecef[3],
                    wgt2 * pcc_ecef[1] + wgt1 * pcc_ecef[2],
                    wgt4 * pcc_ecef[1] + wgt3 * pcc_ecef[2],
                    wgt4 * pcc_ecef[0] + wgt3 * pcc_ecef[3],
                ]
                iscc = sarkit.wgs84.cartesian_to_geodetic(iscc_ecef)[:, :2]
                seginfo.igeolo = sarkit.sicd._io._format_igeolo(iscc)

    return fhdr_numi, fhdr_li, seginfos


def _validate_xml(sidd_xmltree):
    """Validate a SIDD XML tree against the schema"""

    xmlns = lxml.etree.QName(sidd_xmltree.getroot()).namespace
    if xmlns not in siddconst.VERSION_INFO:
        latest_xmlns = list(siddconst.VERSION_INFO.keys())[-1]
        logger.warning(f"Unknown SIDD namespace {xmlns}, assuming {latest_xmlns}")
        xmlns = latest_xmlns
    schema = lxml.etree.XMLSchema(file=siddconst.VERSION_INFO[xmlns]["schema"])
    valid = schema.validate(sidd_xmltree)
    if not valid:
        warnings.warn(str(schema.error_log))
    return valid
