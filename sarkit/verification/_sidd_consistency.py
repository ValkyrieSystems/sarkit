import datetime
import functools
import itertools
import os
import pprint
import re
from typing import Any, Optional

import jbpy
import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import shapely.geometry as shg

import sarkit.sidd as sksidd
import sarkit.verification._consistency as con
from sarkit import wgs84

_PIXEL_INFO = {
    "MONO8I": {
        "IREP": "MONO",
        "NBANDS": 1,
        "IREPBAND": ["M"],
        "NLUTS": [(0,)],
        "IMODE": "B",
    },
    "MONO8LU": {
        "IREP": "MONO",
        "NBANDS": 1,
        "IREPBAND": ["LU"],
        "NLUTS": [(1, 2)],
        "IMODE": "B",
    },
    "MONO16I": {
        "IREP": "MONO",
        "NBANDS": 1,
        "IREPBAND": ["M"],
        "NLUTS": [(0,)],
        "IMODE": "B",
    },
    "RGB8LU": {
        "IREP": "RGB/LUT",
        "NBANDS": 1,
        "IREPBAND": ["LU"],
        "NLUTS": [(3,)],
        "IMODE": "B",
    },
    "RGB24I": {
        "IREP": "RGB",
        "NBANDS": 3,
        "IREPBAND": ["R", "G", "B"],
        "NLUTS": [(0,), (0,), (0,)],  # document says each is "1", but that seems wrong
        "IMODE": "P",
    },
}


def per_image(method):
    method.per_image = True
    return method


def _is_v1(con_obj) -> bool:
    """Return ``True`` if first SIDD XML tree is v1.0"""
    return (
        lxml.etree.QName(con_obj.xml_trees[0].getroot()).namespace == "urn:SIDD:1.0.0"
    )


def _get_corners(xmlhelp):
    ns = lxml.etree.QName(xmlhelp.element_tree.getroot()).namespace
    if ns == "urn:SIDD:1.0.0":
        corners_path = "{*}GeographicAndTarget/{*}GeographicCoverage/{*}Footprint"
    else:
        corners_path = "{*}GeoData/{*}ImageCorners"
    return xmlhelp.load(corners_path)


class SiddConsistency(con.ConsistencyChecker):
    """Check SIDD file structure and metadata for internal consistency

    `SiddConsistency` objects should be instantiated using `from_file` or `from_parts`.

    Parameters
    ----------
    xml_trees : lxml.etree.Element or lxml.etree.ElementTree
        SIDD XML
    schema_override : `path-like object`, optional
        Path to XML Schema. If None, tries to find a version-specific schema
    file : `file object`, optional
        SIDD NITF file; when specified, NITF headers are extracted during object instantiation
    """

    def __init__(self, xml_trees, schema_override=None, file=None):
        super().__init__()

        self.schema_override = schema_override
        if file is not None:
            file.seek(0, os.SEEK_SET)
            self.ntf = jbpy.Jbp().load(file)
        else:
            self.ntf = None

        self.xml_trees = [
            item.getroottree() if hasattr(item, "getroottree") else item
            for item in xml_trees
        ]

        # process decorated methods to generate per-image tests
        # reverse the enumerated list so that we don't disturb indices on later iterations as we insert into the list
        for index, func in reversed(list(enumerate(self.funcs))):
            if getattr(func, "per_image", False):
                subfuncs = []
                for image_number, xml_tree in enumerate(xml_trees):
                    subfunc = functools.partial(func, image_number, xml_tree)
                    subfunc.__doc__ = (
                        f"{func.__doc__.removesuffix('.')} for image {image_number}."
                    )
                    subfunc.__name__ = f"{func.__name__}_image{image_number:03d}"
                    subfuncs.append(subfunc)
                self.funcs[index : index + 1] = subfuncs

    @classmethod
    def from_file(cls, file, schema: Optional[str] = None) -> "SiddConsistency":
        """Create a SiddConsistency object from a file

        Parameters
        ----------
        file : `file object`
            SIDD NITF or SIDD XML file to check
        schema : str, optional
            Path to XML Schema. If None, tries to find a version-specific schema

        Returns
        -------
        SiddConsistency
            The initialized consistency checker object

        See Also
        --------
        from_parts

        Examples
        --------
        Use `from_file` to check an XML file:

        .. doctest::

            >>> import sarkit.verification as skver

            >>> with open("data/example-sidd-3.0.0.xml", "r") as f:
            ...     con = skver.SiddConsistency.from_file(f)
            >>> con.check()
            >>> bool(con.passes())
            True
            >>> bool(con.failures())
            False

        Use `from_file` to check a SIDD NITF file:

        .. testsetup::

            import lxml.etree
            import numpy as np

            import sarkit.sidd as sksidd

            file = tmppath / "example.sidd"
            sec = {"security": {"clas": "U"}}
            example_sidd_xmltree = lxml.etree.parse("data/example-sidd-3.0.0.xml")
            sidd_meta = sksidd.NitfMetadata(
                file_header_part={"ostaid": "nowhere"} | sec,
                images=[sksidd.NitfProductImageMetadata(
                    xmltree=example_sidd_xmltree,
                    im_subheader_part=sec,
                    de_subheader_part=sec,
                )]
            )
            with open(file, "wb") as f, sksidd.NitfWriter(f, sidd_meta):
                pass  # don't currently care about the pixels

        .. doctest::

            >>> with file.open("rb") as f:
            ...     con = skver.SiddConsistency.from_file(f)
            >>> con.check()  # open file only used for construction
            >>> bool(con.passes())
            True
            >>> bool(con.failures())
            False

        """
        kwargs: dict[str, Any] = {"schema_override": schema}
        try:
            xml_trees = [lxml.etree.parse(file)]
        except lxml.etree.XMLSyntaxError:
            ntf = jbpy.Jbp()
            xml_trees = []
            file.seek(0, os.SEEK_SET)
            ntf.load(file)
            for deseg in ntf["DataExtensionSegments"]:
                if not deseg["subheader"]["DESID"].value == "XML_DATA_CONTENT":
                    continue
                desshtn = getattr(
                    deseg["subheader"].get("DESSHTN"), "encoded_value", b""
                )
                if not desshtn.startswith(b"urn:SIDD"):
                    continue
                file.seek(deseg["DESDATA"].get_offset())
                xml_bytes = file.read(deseg["DESDATA"].size)
                xml_trees.append(
                    lxml.etree.ElementTree(lxml.etree.fromstring(xml_bytes))
                )
            kwargs["file"] = file

        return cls(xml_trees, **kwargs)

    @classmethod
    def from_parts(
        cls,
        xml_trees: list["lxml.etree.Element | lxml.etree.ElementTree"],
        schema: Optional[str] = None,
    ) -> "SiddConsistency":
        """Create a SiddConsistency object from assorted parts

        Parameters
        ----------
        sidd_xml : lxml.etree.Element or lxml.etree.ElementTree
            SIDD XML
        schema : `path-like object`, optional
            Path to XML Schema. If None, tries to find a version-specific schema

        Returns
        -------
        SiddConsistency
            The initialized consistency checker object

        See Also
        --------
        from_file

        Examples
        --------
        Use `from_parts` to check a parsed XML element tree:

        .. doctest::

            >>> import lxml.etree
            >>> import sarkit.verification as skver
            >>> sidd_xmltree = lxml.etree.parse("data/example-sidd-3.0.0.xml")
            >>> con = skver.SiddConsistency.from_parts([sidd_xmltree])
            >>> con.check()
            >>> bool(con.passes())
            True
            >>> bool(con.failures())
            False
        """
        return cls(xml_trees, schema)

    @per_image
    def check_against_schema(self, image_number, xml_tree):
        """The XML matches the schema."""
        with self.precondition():
            ns = lxml.etree.QName(xml_tree.getroot()).namespace
            schema_filename = self.schema_override or sksidd.VERSION_INFO.get(
                ns, {}
            ).get("schema")
            assert schema_filename is not None
            schema = lxml.etree.XMLSchema(file=schema_filename)

            with self.need("XML passes schema"):
                assert schema.validate(xml_tree), schema.error_log

    def check_sidd_version(self) -> None:
        """SIDD XMLs have same namespace"""
        with self.need("All SIDD XMLs have same namespace"):
            namespaces = set(
                lxml.etree.QName(xml_tree.getroot()).namespace
                for xml_tree in self.xml_trees
            )
            assert len(namespaces) == 1

    def check_nitf_des_headers(self) -> None:
        """DES Subheader fields match spec"""
        with self.precondition():
            assert self.ntf is not None
            actual: Any = None
            expected: Any = None

            def _de_segment_type(segment):
                deseg_type = "OTHER"
                if segment["subheader"]["DESID"].value == "XML_DATA_CONTENT":
                    if segment["subheader"]["DESSHTN"].encoded_value.startswith(
                        b"urn:SIDD"
                    ):
                        deseg_type = "SIDD"
                    if segment["subheader"]["DESSHTN"].encoded_value.startswith(
                        b"urn:SICD"
                    ):
                        deseg_type = "SICD"
                    else:
                        deseg_type = "SUPPORT"
                elif segment["subheader"]["DESID"].value == "SICD_XML":
                    # SIDD v1.0 uses this nonstandard DESID
                    deseg_type = "SICD"
                return deseg_type

            deseg_types = [
                _de_segment_type(deseg) for deseg in self.ntf["DataExtensionSegments"]
            ]
            with self.need("DE segments are in SIDD->SICD->OTHER order"):
                expected = sorted(
                    deseg_types, key=["SIDD", "SUPPORT", "SICD", "OTHER"].index
                )
                assert deseg_types == expected

            with self.precondition():
                assert not _is_v1(self)
                desids = set(
                    x["subheader"]["DESID"].value
                    for x in self.ntf["DataExtensionSegments"]
                )
                with self.want("Outmoded DESID=SICD_XML not present"):
                    assert "SICD_XML" not in desids

            for des_idx, xml_tree in enumerate(self.xml_trees):
                subhdr = self.ntf["DataExtensionSegments"][des_idx]["subheader"]
                with self.need("SIDD XML is in a XML_DATA_CONTENT DES"):
                    assert subhdr["DESID"].value == "XML_DATA_CONTENT"
                with self.need("DESVER is 1"):
                    assert subhdr["DESVER"].value == 1
                with self.need("DESCRC is not used"):
                    assert subhdr["DESCRC"].value == 99999
                with self.need("DESSHFT is XML"):
                    assert subhdr["DESSHFT"].value == "XML"
                with self.need("DESSHSI specifies SIDD standard"):
                    expected = (
                        "SIDD Volume 1 Design & Implementation Description Document"
                    )
                    assert subhdr["DESSHSI"].value == expected

                actual = {
                    name: subhdr[name].value
                    for name in ("DESSHSV", "DESSHSD", "DESSHTN")
                }

                version_fields = [
                    {
                        "DESSHSV": info["version"],
                        "DESSHSD": info["date"],
                        "DESSHTN": namespace,
                    }
                    for namespace, info in sksidd.VERSION_INFO.items()
                ]

                with self.need("version information is a complete set"):
                    assert actual in version_fields

                with self.need("header matches instance namespace"):
                    assert (
                        actual["DESSHTN"]
                        == lxml.etree.QName(xml_tree.getroot()).namespace
                    )

                def _parse_ll_pair(string):
                    return float(string[:12]), float(string[12:])

                desshlpg = subhdr["DESSHLPG"].value
                actual = [
                    _parse_ll_pair(desshlpg[pair * 25 : (pair + 1) * 25])
                    for pair in range(5)
                ]

                xmlhelp = sksidd.XmlHelper(xml_tree)
                icp_ll = _get_corners(xmlhelp)

                found = False
                # Starting vertex isn't specified in document, try them all
                for attempt in range(4):
                    expected = np.roll(icp_ll, attempt, axis=0).tolist()
                    expected.append(expected[0])

                    if np.all(np.isclose(actual, expected, atol=1e-7)):
                        found = True

                with self.need("DESSHLPG matches an ImageCorners ordering"):
                    assert found, f"{actual=}\n{icp_ll=}"

                with self.need("DESSHLPT is space-filled"):
                    assert subhdr["DESSHLPT"].isnull()

    @staticmethod
    def _im_segment_type(segment):
        imseg_type = "OTHER"
        if segment["subheader"]["IID1"].value.startswith("SIDD"):
            if segment["subheader"]["ICAT"].value == "SAR":
                imseg_type = "SIDD_SAR"
            if segment["subheader"]["ICAT"].value == "LEG":
                imseg_type = "SIDD_LEGEND"
            if segment["subheader"]["ICAT"].value == "DED":
                imseg_type = "SIDD_DED"
        return imseg_type

    def check_nitf_image_segment_order(self) -> None:
        """Image Segment categories must be in order"""
        with self.precondition():
            assert self.ntf is not None

            # iterate and for each segment create a list of what is allowed to be next
            #    SIDD{n}{m} SAR -> SIDD{n}{m+1} SAR | SIDD{n}{m+1} LEG | SIDD{n+1}001 SAR | DED001 | OTHER
            #    SIDD{n}{m} LEG -> SIDD{n}{m+1} LEG | SIDD{n+1}001 SAR | DED001 | OTHER
            #    DED001 -> OTHER
            #    OTHER -> OTHER
            def _segment_desc(segment):
                iid1 = segment["subheader"]["IID1"].value
                icat = segment["subheader"]["ICAT"].value

                if (iid1.startswith("SIDD") and icat in ("SAR", "LEG")) or (
                    iid1 == "DED001" and icat == "DED"
                ):
                    return (iid1, icat)
                return ("OTHER", "")

            allowed = [("SIDD001001", "SAR")]  # must start with first SAR image segment
            for seg in self.ntf["ImageSegments"]:
                desc = _segment_desc(seg)
                with self.need("Image segments in correct order"):
                    assert desc in allowed

                if desc == ("OTHER", ""):
                    allowed = [("OTHER", "")]
                elif desc[1] == "DED":
                    allowed = [("OTHER", "")]
                elif desc[1] == "SAR":
                    n = int(desc[0][4:7])
                    m = int(desc[0][7:10])
                    allowed = [
                        (f"SIDD{n:03d}{m + 1:03d}", "SAR"),
                        (f"SIDD{n:03d}{m + 1:03d}", "LEG"),
                        (f"SIDD{n + 1:03d}001", "SAR"),
                        ("DED001", "DED"),
                        ("OTHER", ""),
                    ]
                elif desc[1] == "LEG":
                    n = int(desc[0][4:7])
                    m = int(desc[0][7:10])
                    allowed = [
                        (f"SIDD{n:03d}{m + 1:03d}", "LEG"),
                        (f"SIDD{n + 1:03d}001", "SAR"),
                        ("DED001", "DED"),
                        ("OTHER", ""),
                    ]

    def check_nitf_display_levels(self):
        """Image segments have reasonable display and attachment levels"""
        with self.precondition():
            assert self.ntf is not None

            idlvls = [
                seg["subheader"]["IDLVL"].value for seg in self.ntf["ImageSegments"]
            ]
            ialvls = [
                seg["subheader"]["IALVL"].value for seg in self.ntf["ImageSegments"]
            ]
            with self.need("IDLVLs range from 1...num image segments"):
                assert sorted(idlvls) == sorted(
                    range(1, len(self.ntf["ImageSegments"]) + 1)
                )
            with self.need("IALVLs reference an existing IDLVL"):
                assert set(ialvls) <= set(idlvls) | set([0])

            with self.need("image segment is attached to a lower display level"):
                for idlvl, ialvl in zip(idlvls, ialvls):
                    assert ialvl < idlvl

    @per_image
    def check_nitf_sidd_image_segments(self, image_number, xml_tree):
        """Check relationships between segments of a single image"""
        with self.precondition():
            assert self.ntf is not None

            headers = [
                seg["subheader"]
                for seg in self.ntf["ImageSegments"]
                if seg["subheader"]["IID1"].value.startswith(
                    f"SIDD{image_number + 1:03d}"
                )
            ]
            sar_headers = [hdr for hdr in headers if hdr["ICAT"].value == "SAR"]
            sar_idlvls = sorted([hdr["IDLVL"].value for hdr in sar_headers])
            sar_ialvls = sorted([hdr["IALVL"].value for hdr in sar_headers])
            with self.need("First SAR segment is attached to CCS"):
                assert sar_ialvls[0] == 0
            with self.need("SAR segments have sequential IDLVLS"):
                assert np.all(np.diff(sar_idlvls)) == 1
            with self.need("SAR segments are attached to the previous segment"):
                assert sar_ialvls[1:] == sar_idlvls[:-1]

            pixel_footprint = sksidd.ElementWrapper(xml_tree.getroot())["Measurement"][
                "PixelFootprint"
            ]
            sum_nrows = sum(hdr["NROWS"].value for hdr in sar_headers)
            all_ncols = np.asarray([hdr["NCOLS"].value for hdr in sar_headers])
            with self.need("Combined SAR segments shape matches pixel footprint"):
                assert np.all(all_ncols == all_ncols[0])
                assert all_ncols[0] == pixel_footprint[1]
                assert sum_nrows == pixel_footprint[0]

            leg_headers = [hdr for hdr in headers if hdr["ICAT"].value == "LEG"]
            leg_idlvls = sorted([hdr["IDLVL"].value for hdr in leg_headers])
            leg_ialvls = sorted([hdr["IALVL"].value for hdr in leg_headers])
            with self.need("Legends are displayed immediately after the SAR segments"):
                expected_idlvls = list(
                    range(sar_idlvls[-1] + 1, sar_idlvls[-1] + 1 + len(leg_headers))
                )
                assert sorted(leg_idlvls) == expected_idlvls
            with self.need("Legends are attached to SAR segments"):
                assert set(leg_ialvls) <= set(sar_idlvls)

    @per_image
    def check_nitf_image_segmentation(self, image_number, xml_tree):
        """NITF Image Subheaders match SIDD segmentation algorithm"""
        with self.precondition():
            assert self.ntf is not None

            segments = [
                seg
                for seg in self.ntf["ImageSegments"]
                if seg["subheader"]["IID1"].value.startswith(
                    f"SIDD{image_number + 1:03d}"
                )
            ]

            # seginfos are for a single image, update based on other images in the SIDD
            _, _, seginfos = sksidd.segmentation_algorithm([xml_tree])
            for info in seginfos:
                info.iid1 = f"SIDD{image_number + 1:03d}{info.iid1[7:10]}"
                info.idlvl += segments[0]["subheader"]["IDLVL"].value - 1
                if info.ialvl != 0:
                    info.ialvl += segments[0]["subheader"]["IDLVL"].value - 1

            for idx, (segment, expected) in enumerate(zip(segments, seginfos)):
                with self.need(f"Image Segment {idx} must be SAR"):
                    assert segment["subheader"]["ICAT"].value == "SAR"

                with self.need(f"Image Segment {idx} has expected IDLVL"):
                    assert segment["subheader"]["IDLVL"].value == expected.idlvl
                with self.need(f"Image Segment {idx} has expected IALVL"):
                    assert segment["subheader"]["IALVL"].value == expected.ialvl
                with self.need(f"Image Segment {idx} has expected ILOC"):
                    assert segment["subheader"]["ILOC"].value == (
                        int(expected.iloc[:5]),
                        int(expected.iloc[5:]),
                    )
                with self.need(f"Image Segment {idx} has expected IID1"):
                    assert segment["subheader"]["IID1"].value == expected.iid1
                with self.need(f"Image Segment {idx} has expected NROWS"):
                    assert segment["subheader"]["NROWS"].value == expected.nrows
                with self.need(f"Image Segment {idx} has expected NCOLS"):
                    assert segment["subheader"]["NCOLS"].value == expected.ncols

                def _dms_to_dd(dms_str):
                    direction = dms_str[-1]
                    ss = int(dms_str[-3:-1])
                    mm = int(dms_str[-5:-3])
                    dd = int(dms_str[:-5])
                    sign = 1 if direction in ["E", "N"] else -1
                    return sign * (dd + mm / 60 + ss / 60 / 60)

                igeolo = self.ntf["ImageSegments"][idx]["subheader"]["IGEOLO"].value
                igeolo_ll = [
                    [_dms_to_dd(igeolo[0:7]), _dms_to_dd(igeolo[7:15])],
                    [_dms_to_dd(igeolo[15:22]), _dms_to_dd(igeolo[22:30])],
                    [_dms_to_dd(igeolo[30:37]), _dms_to_dd(igeolo[37:45])],
                    [_dms_to_dd(igeolo[45:52]), _dms_to_dd(igeolo[52:60])],
                ]

                igeolo = expected.igeolo
                expected_ll = [
                    [_dms_to_dd(igeolo[0:7]), _dms_to_dd(igeolo[7:15])],
                    [_dms_to_dd(igeolo[15:22]), _dms_to_dd(igeolo[22:30])],
                    [_dms_to_dd(igeolo[30:37]), _dms_to_dd(igeolo[37:45])],
                    [_dms_to_dd(igeolo[45:52]), _dms_to_dd(igeolo[52:60])],
                ]

                with self.need(f"Image Segment {idx} has expected IGEOLO"):
                    np.testing.assert_allclose(
                        igeolo_ll, expected_ll, atol=1.0 / 60 / 60, rtol=0
                    )

            for segment in segments[len(seginfos) :]:
                with self.need("Legends located after SAR image"):
                    assert segment["subheader"]["ICAT"].value == "LEG"

    @per_image
    def check_nitf_image_subheaders(self, image_number, xml_tree):
        """Image Subheader fields match SIDD XML"""
        helper = sksidd.XmlHelper(xml_tree)
        with self.precondition():
            assert self.ntf is not None

            iid1prefix = f"SIDD{image_number + 1:03d}"
            imsegs = [
                imseg
                for imseg in self.ntf["ImageSegments"]
                if imseg["subheader"]["IID1"].value.startswith(iid1prefix)
            ]
            for imseg in imsegs:
                icat = imseg["subheader"]["ICAT"].value
                with self.need("Valid ICAT"):
                    assert icat in ("SAR", "LEG")

                if icat != "SAR":
                    continue  # checks for Legends not implemented

                idatim = datetime.datetime.strptime(
                    imseg["subheader"]["IDATIM"].value, "%Y%m%d%H%M%S"
                )
                idatim = idatim.replace(tzinfo=datetime.timezone.utc)
                collection_date_time = helper.load(
                    "./{*}ExploitationFeatures/{*}Collection/{*}Information/{*}CollectionDateTime"
                )  # Volme 2 says to use './AdvancedExploitation/... , which doesn't exist
                with self.need("IDATIM matches 1st CollectionDateTime"):
                    assert abs(collection_date_time - idatim) < datetime.timedelta(
                        seconds=1
                    )

                sensor_name = xml_tree.findtext(
                    "./{*}ExploitationFeatures/{*}Collection/{*}Information/{*}SensorName"
                )
                with self.need("ISORCE matches 1st SensorName"):
                    assert imseg["subheader"]["ISORCE"].value == sensor_name

                with self.need("PVTYPE is INT for SIDD product images"):
                    assert imseg["subheader"]["PVTYPE"].value == "INT"

                pixel_info = _PIXEL_INFO[xml_tree.findtext("./{*}Display/{*}PixelType")]

                with self.need("IREP is valid for product images"):
                    assert imseg["subheader"]["IREP"].value == pixel_info["IREP"]

                with self.need("ABPP is either 8 or 16"):
                    assert imseg["subheader"]["ABPP"].value in (8, 16)

                with self.need("PJUST is R"):
                    assert imseg["subheader"]["PJUST"].value == "R"

                with self.need("ICORDS is G for product images"):
                    assert imseg["subheader"]["ICORDS"].value == "G"

                with self.need("IC is valid"):
                    assert imseg["subheader"]["IC"].value in ("NC", "C8", "M8")

                if imseg["subheader"]["IC"].value in ("C8", "M8"):
                    with self.need("COMRAT describes J2K"):
                        comrat = imseg["subheader"]["COMRAT"].value

                        def is_numerically_lossless():
                            return re.fullmatch("N[0-9]{3}", comrat) is not None

                        def is_visually_lossless():
                            return re.fullmatch("V[0-9]{3}", comrat) is not None

                        def is_lossy():
                            valid_charset = (
                                re.fullmatch("[0-9.]{4}", comrat) is not None
                            )
                            return valid_charset and (comrat.count(".") <= 1)

                        assert (
                            is_numerically_lossless()
                            or is_visually_lossless()
                            or is_lossy()
                        )

                with self.need("NBANDS matches PixelType"):
                    assert imseg["subheader"]["NBANDS"].value == pixel_info["NBANDS"]

                for idx, (irepband, nlut_options) in enumerate(
                    zip(pixel_info["IREPBAND"], pixel_info["NLUTS"])
                ):
                    with self.need("IREPBAND consistent with PixelType"):
                        assert (
                            imseg["subheader"][f"IREPBAND{idx + 1:05d}"].value
                            == irepband
                        )
                    with self.need("ISUBCAT is space-filled"):
                        assert not imseg["subheader"][f"ISUBCAT{idx + 1:05d}"].value
                    with self.need('IFC is "N"'):
                        assert imseg["subheader"][f"IFC{idx + 1:05d}"].value == "N"
                    with self.need("IMFLT is space-filled"):
                        assert not imseg["subheader"][f"IMFLT{idx + 1:05d}"].value
                    with self.need("NLUTS consistent with PixelType"):
                        assert (
                            imseg["subheader"][f"NLUTS{idx + 1:05d}"].value
                            in nlut_options
                        )

                with self.need("ISYNC is 0"):
                    assert imseg["subheader"]["ISYNC"].value == 0

                with self.need("IMODE consistent with PixelType"):
                    assert imseg["subheader"]["IMODE"].value == pixel_info["IMODE"]

                with self.need("NBPP is either 8 or 16"):
                    assert imseg["subheader"]["NBPP"].value in (8, 16)

                with self.want("NBPP matches ABPP"):
                    assert (
                        imseg["subheader"]["NBPP"].value
                        == imseg["subheader"]["ABPP"].value
                    )

    @per_image
    def check_datetime_fields_are_utc(self, image_number, xml_tree) -> None:
        """Datetime fields should be followed by Z to indicate UTC."""
        for field in (
            "./{*}ProductCreation/{*}ProcessorInformation/{*}ProcessingDateTime",
            "./{*}ExploitationFeatures/{*}Collection/{*}Information/{*}CollectionDateTime",
            "./{*}DownstreamReprocessing/{*}ProcessingEvent/{*}AppliedDateTime",
        ):
            for element in xml_tree.findall(field):
                with self.want(f"{field} ends in Z to indicate UTC"):
                    assert element.text.strip().endswith("Z")

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_display_numbands(self, image_number, xml_tree) -> None:
        """Display/NumBands is consistent with Display/PixelType."""
        pixel_type = xml_tree.findtext("./{*}Display/{*}PixelType")
        expected_num_bands = {
            "MONO8I": 1,
            "MONO8LU": 1,
            "MONO16I": 1,
            "RGB8LU": 3,
            "RGB24I": 3,
        }[pixel_type]
        with self.need(
            f"Display/PixelType: {pixel_type} has {expected_num_bands} bands"
        ):
            assert (
                int(xml_tree.findtext("./{*}Display/{*}NumBands")) == expected_num_bands
            )

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_display_processing_bands(self, image_number, xml_tree) -> None:
        """Display/[Non]InteractiveProcessing nodes are present for each band."""
        num_bands = int(xml_tree.findtext("./{*}Display/{*}NumBands"))
        expected_bands = list(range(1, num_bands + 1))
        for tag in (
            "./{*}Display/{*}NonInteractiveProcessing",
            "./{*}Display/{*}InteractiveProcessing",
        ):
            actual_bands = sorted(
                int(elem.get("band", "-1")) for elem in xml_tree.findall(tag)
            )
            with self.need(
                f"{tag} repeated NumBands times with band attribute set from 1 to NumBands"
            ):
                assert actual_bands == expected_bands

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_display_antialias_filter_operation(self, image_number, xml_tree) -> None:
        """Display/.../(RRDS|Scaling)/AntiAlias/Operation is set correctly."""
        elems = itertools.chain(
            xml_tree.findall(
                "./{*}Display/{*}NonInteractiveProcessing/{*}RRDS/{*}AntiAlias/{*}Operation"
            ),
            xml_tree.findall(
                "./{*}Display/{*}InteractiveProcessing/{*}GeometricTransform"
                "/{*}Scaling/{*}AntiAlias/{*}Operation"
            ),
        )
        expected_operation = "CONVOLUTION"
        for elem in elems:
            with self.need(
                f'{elem.getroottree().getpath(elem)} = "{expected_operation}"'
            ):
                assert elem.text == expected_operation

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_display_interpolation_filter_operation(
        self, image_number, xml_tree
    ) -> None:
        """Display/.../(RRDS|Scaling)/Interpolation/Operation is set correctly."""
        elems = itertools.chain(
            xml_tree.findall(
                "./{*}Display/{*}NonInteractiveProcessing/{*}RRDS/{*}Interpolation/{*}Operation"
            ),
            xml_tree.findall(
                "./{*}Display/{*}InteractiveProcessing/{*}GeometricTransform"
                "/{*}Scaling/{*}Interpolation/{*}Operation"
            ),
        )
        expected_operation = "CORRELATION"
        for elem in elems:
            with self.need(
                f'{elem.getroottree().getpath(elem)} = "{expected_operation}"'
            ):
                assert elem.text == expected_operation

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_display_dra_bandstatssource(self, image_number, xml_tree) -> None:
        """Display/InteractiveProcessing/DynamicRangeAdjustment/BandStatsSource is a valid band index."""
        num_bands = int(xml_tree.findtext("./{*}Display/{*}NumBands"))
        path = "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment/{*}BandStatsSource"
        with self.need(f"{path} ∈ [1, {num_bands}]"):
            assert 1 <= int(xml_tree.findtext(path)) <= num_bands

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_display_auto_dra_parameters(self, image_number, xml_tree) -> None:
        """Display/InteractiveProcessing/DynamicRangeAdjustment/DRAParameters included if AlgorithmType = AUTO."""
        dra_path = "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment"
        for elem in xml_tree.findall(f'{dra_path}/{{*}}AlgorithmType[.="AUTO"]'):
            with self.need(
                f"{dra_path}/DRAParameters included if AlgorithmType = AUTO"
            ):
                assert elem.find("../{*}DRAParameters") is not None

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_display_valid_dra_parameters(self, image_number, xml_tree) -> None:
        """Display/InteractiveProcessing/DynamicRangeAdjustment/DRAParameters: 0.0 <= min <= max <= 1.0"""
        for elem in xml_tree.findall(
            "./{*}Display/{*}InteractiveProcessing"
            "/{*}DynamicRangeAdjustment/{*}DRAParameters"
        ):
            for min_param, max_param in [
                ("{*}Pmin", "{*}Pmax"),
                ("{*}EminModifier", "{*}EmaxModifier"),
            ]:
                with self.need(f"0.0 <= {min_param} <= {max_param} <= 1.0"):
                    assert (
                        0.0
                        <= float(elem.findtext(min_param))
                        <= float(elem.findtext(max_param))
                        <= 1.0
                    )

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_display_none_dra_overrides(self, image_number, xml_tree) -> None:
        """Display/InteractiveProcessing/DynamicRangeAdjustment/DRAOverrides excluded if AlgorithmType = NONE."""
        dra_path = "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment"
        for elem in xml_tree.findall(f'{dra_path}/{{*}}AlgorithmType[.="NONE"]'):
            with self.need(f"{dra_path}/DRAOverrides excluded if AlgorithmType = NONE"):
                assert elem.find("../{*}DRAOverrides") is None

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_display_valid_dra_overrides(self, image_number, xml_tree) -> None:
        """Display/InteractiveProcessing/DynamicRangeAdjustment/DRAOverrides ∈ [0.0, 2047.0]"""
        for elem in xml_tree.findall(
            "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment/{*}DRAOverrides"
        ):
            for child in elem:
                with self.need(f"{child.tag} ∈ [0.0, 2047.0]"):
                    assert 0.0 <= float(child.text) <= 2047.0

    @per_image
    def check_measurement_productplane_unit_vectors(
        self, image_number, xml_tree
    ) -> None:
        """Measurement/PlaneProjection/ProductPlane/(Row|Col)UnitVectors are unit-length and orthogonal"""
        with self.precondition():
            product_plane = xml_tree.find(
                "./{*}Measurement/{*}PlaneProjection/{*}ProductPlane"
            )
            assert product_plane is not None
            u_row = sksidd.XyzType().parse_elem(
                product_plane.find("./{*}RowUnitVector")
            )
            u_col = sksidd.XyzType().parse_elem(
                product_plane.find("./{*}ColUnitVector")
            )
            with self.need("RowUnitVector is unit-length"):
                assert np.linalg.norm(u_row) == con.Approx(1)
            with self.need("ColUnitVector is unit-length"):
                assert np.linalg.norm(u_col) == con.Approx(1)
            with self.need("RowUnitVector and ColUnitVector are orthogonal"):
                assert np.dot(u_row, u_col) == con.Approx(0, atol=1e-6)

    @per_image
    @con.skipif(_is_v1, "Does not apply to SIDD v1.0")
    def check_measurement_validdata(self, image_number, xml_tree) -> None:
        """Measurement/ValidData is a simple convex polygon with vertices in clockwise order"""
        xmlhelp = sksidd.XmlHelper(xml_tree)
        polygon = xmlhelp.load("./{*}Measurement/{*}ValidData")
        validdata_node = xml_tree.find("./{*}Measurement/{*}ValidData")
        vertex_nodes = sorted(
            list(validdata_node), key=lambda x: int(x.attrib["index"])
        )
        with self.need("ValidData indices are all present"):
            assert [int(x.attrib["index"]) for x in vertex_nodes] == list(
                range(1, len(vertex_nodes) + 1)
            )
        size = int(validdata_node.attrib["size"])
        with self.need("ValidData size attribute matches the number of vertices"):
            assert size == len(vertex_nodes)
        shg_polygon = shg.Polygon(polygon)
        with self.need("ValidData is simple"):
            assert shg_polygon.is_simple
        with self.need("ValidData is clockwise"):
            assert not shg_polygon.exterior.is_ccw
        with self.need("Vertex 1 determined by min row index & min col index"):
            assert np.lexsort((polygon[:, 1], polygon[:, 0]))[0] == 0
        with self.want("ValidData vertices contained within PixelFootprint"):
            nrows, ncols = xmlhelp.load("./{*}Measurement/{*}PixelFootprint")
            pad = 1
            assert np.all(polygon >= (-pad, -pad))
            assert np.all(polygon <= (nrows + pad, ncols + pad))

    @per_image
    def check_expfeatures_geometry(self, image_number, xml_tree) -> None:
        """ExploitationFeatures geometry is consistent with Measurement parameters"""

        def is_geom_param_equal(name, actual, expected):
            if name.endswith("Magnitude"):
                return actual == con.Approx(expected, atol=0.01, rtol=0.01)
            if name == "{*}Geometry/{*}Graze" and expected < 0:
                return True  # SIDD 2.0 spec is self-contradictory and does not state how to describe this geometry
            return (
                180 - abs(abs(actual - expected) - 180)
            ) <= 1.0  # compare angles (degrees)

        expfeatures_elem = xml_tree.find("./{*}ExploitationFeatures")
        with self.precondition():
            expected_geom = calc_expfeatures_geom(xml_tree)
            collection_geom = {
                k.removeprefix("{*}Collection/"): v
                for k, v in expected_geom.items()
                if k.startswith("{*}Collection")
            }
            any_collection_has_calculable_field = False
            any_collections_calculable_fields_match = False
            mismatched_collections = []
            for collection in expfeatures_elem.findall("./{*}Collection"):
                actual_geom = {
                    field: float(elem.text)
                    for field in collection_geom
                    if (elem := collection.find(f"./{field}")) is not None
                }
                if not actual_geom:
                    continue
                any_collection_has_calculable_field = True
                these_mismatches = {
                    name: {"actual": actual, "expected": collection_geom[name]}
                    for name, actual in actual_geom.items()
                    if not is_geom_param_equal(name, actual, collection_geom[name])
                }
                any_collections_calculable_fields_match |= not these_mismatches
                if any_collections_calculable_fields_match:
                    break
                else:
                    mismatched_collections.append(these_mismatches)
            if any_collection_has_calculable_field:
                with self.want(
                    "At least one ExploitationFeatures/Collection consistent with Measurement parameters"
                ):
                    assert any_collections_calculable_fields_match, pprint.pformat(
                        mismatched_collections, width=120
                    )

            north_elems = expfeatures_elem.findall("./{*}Product/{*}North")
            if north_elems:
                actual_norths = [float(elem.text) for elem in north_elems]
                expected_north = expected_geom["{*}Product/{*}North"]
                with self.want(
                    "At least one ExploitationFeatures/Product/North consistent with Measurement"
                ):
                    assert any(
                        is_geom_param_equal(
                            "{*}Product/{*}North", actual_ang, expected_north
                        )
                        for actual_ang in actual_norths
                    ), f"{expected_north=}\n{actual_norths=}"

    @per_image
    def check_geodata_image_corners(self, image_number, xml_tree) -> None:
        """Image Corners are consistent with Measurement element."""
        xmlhelp = sksidd.XmlHelper(xml_tree)
        n_rows, n_cols = xmlhelp.load("./{*}Measurement/{*}PixelFootprint")
        corners = [
            (-0.5, -0.5),
            (-0.5, n_cols - 0.5),
            (n_rows - 0.5, n_cols - 0.5),
            (n_rows - 0.5, -0.5),
        ]
        icp_from_measurement = sksidd.calculations.pixel_to_ecef(xml_tree, corners)

        icp_ll = _get_corners(xmlhelp)
        ref_pt = xmlhelp.load("./{*}Measurement//{*}ReferencePoint/{*}ECEF")
        _, _, ref_height = wgs84.cartesian_to_geodetic(ref_pt)
        icp_ecef = wgs84.geodetic_to_cartesian(
            np.concatenate((icp_ll, np.full((len(icp_ll), 1), ref_height)), axis=1)
        )
        chords = itertools.combinations(icp_ecef, 2)
        max_chord_length = max(np.linalg.norm(chord[1] - chord[0]) for chord in chords)
        for index, (icp_reported, icp_predicted) in enumerate(
            zip(icp_ecef, icp_from_measurement)
        ):
            icp_dist = np.linalg.norm(icp_predicted - icp_reported)
            with self.want(
                f"Distance between reported and predicted ICP{index + 1} "
                "< 0.05 * maximum chord length"
            ):
                assert icp_dist < 0.05 * max_chord_length


def calc_expfeatures_geom(sidd_xml):
    """ExploitationFeatures Geometry (SIDD2.0, Sec. 7.1-7.5)"""
    sidd_xh = sksidd.XmlHelper(sidd_xml)
    tcoa_poly = sidd_xh.load("./{*}Measurement/*[1]/{*}TimeCOAPoly")
    assert tcoa_poly is not None
    scp = sidd_xh.load("./{*}Measurement/*[1]/{*}ReferencePoint/{*}ECEF")
    scp_coa_time = tcoa_poly[0][0]
    arp_poly = sidd_xh.load("./{*}Measurement/{*}ARPPoly")

    p_a = npp.polyval(scp_coa_time, arp_poly)
    v_a = npp.polyval(scp_coa_time, npp.polyder(arp_poly))

    scp_px = sksidd.ecef_to_pixel(sidd_xml, scp)
    r, c = sksidd.pixel_to_ecef(sidd_xml, scp_px + [[1, 0], [0, 1]]) - scp
    r_hat = r / np.linalg.norm(r)
    c_hat = c / np.linalg.norm(c)

    sidd_versions = list(sksidd.VERSION_INFO)
    this_ns = lxml.etree.QName(sidd_xml.getroot()).namespace
    pre_3 = sidd_versions.index(this_ns) < sidd_versions.index("urn:SIDD:3.0.0")
    angles = sksidd.compute_angles(
        scp,
        p_a,
        v_a,
        r_hat,
        c_hat,
        convention="2.0" if pre_3 else "3.0",
    )
    return {
        "{*}Collection/{*}Geometry/{*}Azimuth": angles.Azimuth,
        "{*}Collection/{*}Geometry/{*}Slope": angles.Slope,
        "{*}Collection/{*}Geometry/{*}Squint": angles.Squint,
        "{*}Collection/{*}Geometry/{*}Graze": angles.Graze,
        "{*}Collection/{*}Geometry/{*}Tilt": angles.Tilt,
        "{*}Collection/{*}Geometry/{*}DopplerConeAngle": angles.DopplerCone,
        "{*}Collection/{*}Phenomenology/{*}Shadow/{*}Angle": angles.Shadow,
        "{*}Collection/{*}Phenomenology/{*}Shadow/{*}Magnitude": angles.ShadowMagnitude,
        "{*}Collection/{*}Phenomenology/{*}Layover/{*}Angle": angles.Layover,
        "{*}Collection/{*}Phenomenology/{*}Layover/{*}Magnitude": angles.LayoverMagnitude,
        "{*}Collection/{*}Phenomenology/{*}MultiPath": angles.MultiPath,
        "{*}Collection/{*}Phenomenology/{*}GroundTrack": angles.GroundTrack,
        "{*}Product/{*}North": angles.North,
    }


# Improve rendered docstring
con.modify_conchecker_docs(SiddConsistency)
