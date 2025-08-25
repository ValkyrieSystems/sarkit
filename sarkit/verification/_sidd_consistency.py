import datetime
import functools
import itertools
import logging
import os
import pprint
import re
from typing import Any, Optional

import jbpy
import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp

import sarkit.sidd as sksidd
import sarkit.verification._consistency as con
from sarkit import wgs84

logger = logging.getLogger(__name__)

try:
    import shapely.geometry as shg
except ImportError as ie:
    logger.warning("'shapely' package not found. Some features may not work correctly.")
    shg = con._ExceptionOnUse(ie)

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


def _unit(vec, axis=-1):
    return vec / np.linalg.norm(vec, axis=axis, keepdims=True)


def per_image(method):
    method.per_image = True
    return method


def _get_version(xml_tree):
    return lxml.etree.QName(xml_tree.getroot()).namespace.split(":")[-1]


class SiddConsistency(con.ConsistencyChecker):
    """Check SIDD file structure and metadata for internal consistency

    `SiddConsistency` objects should be instantiated using `from_file` or `from_parts`.

    Parameters
    ----------
    sidd_xml : lxml.etree.Element or lxml.etree.ElementTree
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
            item.getrootree() if hasattr(item, "getroottree") else item
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
                if not deseg["subheader"]["DESSHF"]["DESSHTN"].value.startswith(
                    "urn:SIDD"
                ):
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
                    if segment["subheader"]["DESSHF"]["DESSHTN"].value.startswith(
                        "urn:SIDD"
                    ):
                        deseg_type = "SIDD"
                    if segment["subheader"]["DESSHF"]["DESSHTN"].value.startswith(
                        "urn:SICD"
                    ):
                        deseg_type = "SICD"
                    else:
                        deseg_type = "SUPPORT"
                return deseg_type

            deseg_types = [
                _de_segment_type(deseg) for deseg in self.ntf["DataExtensionSegments"]
            ]
            with self.need("DE segments are in SIDD->SICD->OTHER order"):
                expected = sorted(
                    deseg_types, key=["SIDD", "SUPPORT", "SICD", "OTHER"].index
                )
                assert deseg_types == expected

            for des_idx, xml_tree in enumerate(self.xml_trees):
                subhdr = self.ntf["DataExtensionSegments"][des_idx]["subheader"]
                with self.need("SIDD XML is in a XML_DATA_CONTENT DES"):
                    assert subhdr["DESID"].value == "XML_DATA_CONTENT"
                with self.need("DESVER is 1"):
                    assert subhdr["DESVER"].value == 1
                with self.need("DESCRC is not used"):
                    assert subhdr["DESSHF"]["DESCRC"].value == 99999
                with self.need("DESSHFT is XML"):
                    assert subhdr["DESSHF"]["DESSHFT"].value == "XML"
                with self.need("DESSHSI specifies SIDD standard"):
                    expected = (
                        "SIDD Volume 1 Design & Implementation Description Document"
                    )
                    assert subhdr["DESSHF"]["DESSHSI"].value == expected

                actual = {
                    name: subhdr["DESSHF"][name].value
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

                desshlpg = subhdr["DESSHF"]["DESSHLPG"].value
                actual = [
                    _parse_ll_pair(desshlpg[pair * 25 : (pair + 1) * 25])
                    for pair in range(5)
                ]

                xmlhelp = sksidd.XmlHelper(xml_tree)
                icp_ll = xmlhelp.load("./{*}GeoData/{*}ImageCorners")

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
                    assert subhdr["DESSHF"]["DESSHLPT"].value == ""

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

            # not clear if all legends all go after the last SAR, or if each legend goes after the SAR it's attached to
            imseg_types = [
                self._im_segment_type(seg) for seg in self.ntf["ImageSegments"]
            ]
            with self.need("IM segments start with SIDD"):
                required_order = ["SIDD_SAR", "SIDD_LEGEND", "SIDD_DED", "OTHER"]
                expected = sorted(imseg_types, key=required_order.index)
                assert imseg_types == expected

    def check_nitf_iid1_order(self) -> None:
        """Images stored sequentially"""
        with self.precondition():
            assert self.ntf is not None

            sar_segments = [
                imseg
                for imseg in self.ntf["ImageSegments"]
                if self._im_segment_type(imseg) == "SIDD_SAR"
            ]
            iid1s = [imseg["subheader"]["IID1"].value for imseg in sar_segments]
            with self.need("SIDD image segments must be in order"):
                assert iid1s == sorted(iid1s)

    def check_nitf_segmentation(self) -> None:
        """NITF Image Subheaders match SIDD segmentation"""
        with self.precondition():
            assert self.ntf is not None

            segmentation_imhdrs = self._image_segmentation()
            for idx, expected in enumerate(segmentation_imhdrs):
                with self.need(f"Image Segment {idx} must be SAR"):
                    assert (
                        self.ntf["ImageSegments"][idx]["subheader"]["ICAT"].value
                        == "SAR"
                    )

                with self.need(f"Image Segment {idx} has expected IDLVL"):
                    assert (
                        self.ntf["ImageSegments"][idx]["subheader"]["IDLVL"].value
                        == expected["idlvl"]
                    )
                with self.need(f"Image Segment {idx} has expected IALVL"):
                    assert (
                        self.ntf["ImageSegments"][idx]["subheader"]["IALVL"].value
                        == expected["ialvl"]
                    )
                with self.need(f"Image Segment {idx} has expected ILOC"):
                    assert (
                        self.ntf["ImageSegments"][idx]["subheader"]["ILOC"].value
                        == expected["iloc"]
                    )
                with self.need(f"Image Segment {idx} has expected IID1"):
                    assert (
                        self.ntf["ImageSegments"][idx]["subheader"]["IID1"].value
                        == expected["iid1"]
                    )
                with self.need(f"Image Segment {idx} has expected NROWS"):
                    assert (
                        self.ntf["ImageSegments"][idx]["subheader"]["NROWS"].value
                        == expected["nrows"]
                    )
                with self.need(f"Image Segment {idx} has expected NCOLS"):
                    assert (
                        self.ntf["ImageSegments"][idx]["subheader"]["NCOLS"].value
                        == expected["ncols"]
                    )

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

                with self.need(f"Image Segment {idx} has expected IGEOLO"):
                    np.testing.assert_allclose(
                        igeolo_ll, expected["igeolo_dd"], atol=1.0 / 60 / 60, rtol=0
                    )

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
                        assert imseg["subheader"][f"ISUBCAT{idx + 1:05d}"].value == ""
                    with self.need('IFC is "N"'):
                        assert imseg["subheader"][f"IFC{idx + 1:05d}"].value == "N"
                    with self.need("IREPBAND is space-filled"):
                        assert imseg["subheader"][f"IMFLT{idx + 1:05d}"].value == ""
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

    def _image_segmentation(self):
        """Attempts to implement the SIDD segmentation algorithm (sans obvious document errors)"""
        z = 0
        fhdr_numi = 0
        num_rows_limit = []
        imhdr = []
        num_seg_per_image = []

        # Not defined by document, but necessary to make coordinate calculation work with multple xml_trees
        starting_seg_per_image = []

        num_rows = []
        num_cols = []
        for k, xml_tree in enumerate(self.xml_trees):
            # k is 1 based in document, 0 based here
            num_rows.append(
                int(xml_tree.findtext("./{*}Measurement/{*}PixelFootprint/{*}Row"))
            )
            num_cols.append(
                int(xml_tree.findtext("./{*}Measurement/{*}PixelFootprint/{*}Col"))
            )

        for k, xml_tree in enumerate(self.xml_trees):
            # k is 1 based in document, 0 based here
            pixel_info = _PIXEL_INFO[xml_tree.findtext("./{*}Display/{*}PixelType")]

            bytes_per_pixel = pixel_info["NBANDS"]
            bytes_per_row = (
                bytes_per_pixel * num_cols[k]
            )  # document says NumRows(m), which can't possibly be right
            num_rows_limit.append(
                min(np.floor(sksidd.LI_MAX / bytes_per_row), sksidd.ILOC_MAX)
            )
            product_size = bytes_per_pixel * num_rows[k] * num_cols[k]

            starting_seg_per_image.append(z)
            if product_size <= sksidd.LI_MAX:
                num_seg_per_image.append(1)
                z += 1
                fhdr_numi += 1
                imhdr.append(
                    {
                        "idlvl": z,
                        "ialvl": 0,
                        "iloc": (0, 0),
                        "iid1": f"SIDD{k + 1:03d}001",  # document say to use m instead of k, but that doesn't make sense
                        "nrows": num_rows[k],  # document says to use num_rows_limit[k],
                        "ncols": num_cols[k],
                    }
                )
            else:
                num_seg_per_image.append(
                    int(np.ceil(num_rows[k] / num_rows_limit[k]))
                )  # document omits ceil
                z += 1
                fhdr_numi += num_seg_per_image[k]
                imhdr.append(
                    {
                        "idlvl": z,
                        "ialvl": 0,
                        "iloc": (0, 0),
                        "iid1": f"SIDD{k + 1:03d}001",  # document say to use m instead of k, but that doesn't make sense
                        "nrows": num_rows_limit[k],  # document doesn't set nrows here
                        "ncols": num_cols[k],
                    }
                )
                for n in range(2, num_seg_per_image[k]):
                    z += 1
                    imhdr.append(
                        {
                            "idlvl": z,
                            "ialvl": z - 1,
                            "iloc": (num_rows_limit[k], 0),
                            "iid1": f"SIDD{k + 1:03d}{n:03d}",
                            "nrows": num_rows_limit[k],
                            "ncols": num_cols[k],
                        }
                    )
                z += 1
                last_seg_rows = (
                    num_rows[k] - (num_seg_per_image[k] - 1) * num_rows_limit[k]
                )
                imhdr.append(
                    {
                        "idlvl": z,
                        "ialvl": z - 1,
                        "iloc": (
                            num_rows_limit[k],
                            0,
                        ),  # document says to use last_seg_rows
                        "iid1": f"SIDD{k + 1:03d}{num_seg_per_image[k]:03d}",  # document says to use the undefiend 'n'
                        "nrows": last_seg_rows,
                        "ncols": num_cols[k],
                    }
                )

        # Image Segment Corner Coordinate Parameters
        for k, xml_tree in enumerate(self.xml_trees):
            # k is 1 based in document, 0 based here

            xmlhelp = sksidd.XmlHelper(xml_tree)
            icp_ll = xmlhelp.load("./{*}GeoData/{*}ImageCorners")
            pcc_lla = np.concatenate((icp_ll, np.full((len(icp_ll), 1), 0)), axis=1)
            pcc_ecef = wgs84.geodetic_to_cartesian(pcc_lla)

            for z in range(num_seg_per_image[k]):
                iscc_ecef = np.zeros((4, 3))
                imhdr_idx = starting_seg_per_image[k] + z
                # imhdr_idx is 'z' in document, but offset to actually work with multiple product images

                wgt1 = (z * num_rows_limit[k]) / num_rows[k]
                wgt2 = 1 - wgt1
                wgt3 = (z * num_rows_limit[k] + imhdr[imhdr_idx]["nrows"]) / num_rows[k]
                wgt4 = 1 - wgt3

                iscc_ecef[1 - 1] = wgt2 * pcc_ecef[1 - 1] + wgt1 * pcc_ecef[4 - 1]
                iscc_ecef[2 - 1] = wgt2 * pcc_ecef[2 - 1] + wgt1 * pcc_ecef[3 - 1]
                iscc_ecef[3 - 1] = wgt4 * pcc_ecef[2 - 1] + wgt3 * pcc_ecef[3 - 1]
                iscc_ecef[4 - 1] = wgt4 * pcc_ecef[1 - 1] + wgt3 * pcc_ecef[4 - 1]

                iscc = wgs84.cartesian_to_geodetic(iscc_ecef)[:, :2]
                imhdr[imhdr_idx]["igeolo_dd"] = [
                    [iscc[0, 0], iscc[0, 1]],
                    [iscc[1, 0], iscc[1, 1]],
                    [iscc[2, 0], iscc[2, 1]],
                    [iscc[3, 0], iscc[3, 1]],
                ]

        return imhdr

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
    def check_display_dra_bandstatssource(self, image_number, xml_tree) -> None:
        """Display/InteractiveProcessing/DynamicRangeAdjustment/BandStatsSource is a valid band index."""
        num_bands = int(xml_tree.findtext("./{*}Display/{*}NumBands"))
        path = "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment/{*}BandStatsSource"
        with self.need(f"{path} ∈ [1, {num_bands}]"):
            assert 1 <= int(xml_tree.findtext(path)) <= num_bands

    @per_image
    def check_display_auto_dra_parameters(self, image_number, xml_tree) -> None:
        """Display/InteractiveProcessing/DynamicRangeAdjustment/DRAParameters included if AlgorithmType = AUTO."""
        dra_path = "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment"
        for elem in xml_tree.findall(f'{dra_path}/{{*}}AlgorithmType[.="AUTO"]'):
            with self.need(
                f"{dra_path}/DRAParameters included if AlgorithmType = AUTO"
            ):
                assert elem.find("../{*}DRAParameters") is not None

    @per_image
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
    def check_display_none_dra_overrides(self, image_number, xml_tree) -> None:
        """Display/InteractiveProcessing/DynamicRangeAdjustment/DRAOverrides excluded if AlgorithmType = NONE."""
        dra_path = "./{*}Display/{*}InteractiveProcessing/{*}DynamicRangeAdjustment"
        for elem in xml_tree.findall(f'{dra_path}/{{*}}AlgorithmType[.="NONE"]'):
            with self.need(f"{dra_path}/DRAOverrides excluded if AlgorithmType = NONE"):
                assert elem.find("../{*}DRAOverrides") is None

    @per_image
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
            expected_geom = calc_expfeatures_geom(xml_tree, _get_version(xml_tree))
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
        """GeoData/ImageCorners are consistent with Measurement element."""
        with self.precondition():
            icp_from_measurement = calc_geodata_imagecorners(xml_tree)
            xmlhelp = sksidd.XmlHelper(xml_tree)
            icp_ll = xmlhelp.load("./{*}GeoData/{*}ImageCorners")
            scp = xmlhelp.load("./{*}Measurement//{*}ReferencePoint/{*}ECEF")
            _, _, scp_height = wgs84.cartesian_to_geodetic(scp)
            icp_ecef = wgs84.geodetic_to_cartesian(
                np.concatenate((icp_ll, np.full((len(icp_ll), 1), scp_height)), axis=1)
            )
            for index, (icp_reported, icp_predicted) in enumerate(
                zip(icp_ecef, icp_from_measurement)
            ):
                icp_dist = np.linalg.norm(icp_predicted - icp_reported)
                scp_dist = np.linalg.norm(icp_reported - scp)
                with self.need(
                    f"Distance between reported and predicted ICP{index + 1} "
                    "< 0.1 * (distance between reported ICP and SCP)"
                ):
                    assert icp_dist < 0.1 * scp_dist


def calc_expfeatures_geom(sidd_xml, sidd_version="2.0.0"):
    """ExploitationFeatures Geometry (SIDD2.0, Sec. 7.1-7.5)"""
    meas_proj_elem = sidd_xml.find("./{*}Measurement/*[1]")
    scp = sksidd.XyzType().parse_elem(
        meas_proj_elem.find("./{*}ReferencePoint/{*}ECEF")
    )
    scp_llh = wgs84.cartesian_to_geodetic(scp)
    ueast = wgs84.east(scp_llh)
    unor = wgs84.north(scp_llh)
    uup = wgs84.up(scp_llh)
    scp_coa_time = sksidd.PolyCoef2dType().parse_elem(
        meas_proj_elem.find("./{*}TimeCOAPoly")
    )[0][0]
    arp_poly = sksidd.XyzPolyType().parse_elem(
        sidd_xml.find("./{*}Measurement/{*}ARPPoly")
    )

    localname = lxml.etree.QName(meas_proj_elem).localname
    if localname == "PlaneProjection":
        r_hat = sksidd.XyzType().parse_elem(
            meas_proj_elem.find("./{*}ProductPlane/{*}RowUnitVector")
        )
        c_hat = sksidd.XyzType().parse_elem(
            meas_proj_elem.find("./{*}ProductPlane/{*}ColUnitVector")
        )
    elif localname == "GeographicProjection":
        r_hat = -unor
        c_hat = ueast
    else:
        raise NotImplementedError(f"{localname} not supported.")

    p_a = npp.polyval(scp_coa_time, arp_poly)
    v_a = npp.polyval(scp_coa_time, npp.polyder(arp_poly))
    va_hat = _unit(v_a)
    p_0 = scp
    zg_hat = uup

    # 7.2 - Slant Plane Definition
    xs_hat = _unit(p_a - p_0)
    n_hat = _unit(np.cross(xs_hat, v_a))
    zs_hat = np.sign(np.dot(p_0, n_hat)) * n_hat
    ys_hat = np.cross(zs_hat, xs_hat)

    # 7.2.1 - Image Plane Definition
    z_hat = np.cross(r_hat, c_hat)

    # 7.5.1 - Azimuth Angle
    azim_ang = np.arctan2(np.dot(ueast, xs_hat), np.dot(unor, xs_hat))

    # 7.5.2 - Slope Angle
    slope_ang = np.arccos(np.dot(zs_hat, zg_hat))

    # 7.5.3 - Doppler Cone Angle
    doppler_cone_ang = np.arccos(np.dot(-xs_hat, va_hat))

    # 7.5.4 - Squint
    # look direction seems to be omitted from the calculation in the spec but is referenced in the table
    uleft = _unit(np.cross(p_a, v_a))
    look = +1 if np.dot(uleft, xs_hat) < 0 else -1

    zp_hat = _unit(p_a)
    xs_prime = xs_hat - np.dot(xs_hat, zp_hat) * zp_hat
    va_prime = va_hat - np.dot(va_hat, zp_hat) * zp_hat
    squint_ang = look * np.arccos(-np.dot(_unit(xs_prime), _unit(va_prime)))

    # 7.5.5 - Grazing Angle
    graze_ang = np.arcsin(np.dot(xs_hat, zg_hat))

    # 7.5.6 - Tilt Angle
    tilt_ang = np.arctan(np.dot(zg_hat, ys_hat) / np.dot(zg_hat, zs_hat))

    # 7.6 Phenomenology
    # 7.6.1 - Shadow
    s = zg_hat - xs_hat / np.dot(xs_hat, zg_hat)
    s_prime = s - np.dot(s, z_hat) / np.dot(zs_hat, z_hat) * zs_hat
    shadow_ang2 = np.arctan2(np.dot(c_hat, s_prime), np.dot(r_hat, s_prime))
    shadow_ang3 = np.arctan2(np.dot(r_hat, s_prime), np.dot(c_hat, s_prime))
    shadow_mag = np.sqrt(np.dot(s_prime, s_prime))

    # 7.6.2 - Layover
    L = z_hat - zs_hat / np.dot(zs_hat, z_hat)  # noqa N806
    layover_ang2 = np.arctan2(np.dot(c_hat, L), np.dot(r_hat, L))
    layover_ang3 = np.arctan2(np.dot(r_hat, L), np.dot(c_hat, L))
    layover_mag = np.sqrt(np.dot(L, L))

    # 7.6.3 - North Direction
    n_prime = unor - np.dot(unor, z_hat) / np.dot(zs_hat, z_hat) * zs_hat
    north_ang2 = np.arctan2(np.dot(c_hat, n_prime), np.dot(r_hat, n_prime))
    north_ang3 = np.arctan2(np.dot(r_hat, n_prime), np.dot(c_hat, n_prime))

    # 7.6.5 - Multi-Path
    m = xs_hat - np.dot(xs_hat, z_hat) / np.dot(zs_hat, z_hat) * zs_hat
    multipath_ang2 = np.arctan2(np.dot(c_hat, m), np.dot(r_hat, m))
    multipath_ang3 = np.arctan2(np.dot(r_hat, m), np.dot(c_hat, m))

    # 7.6.6 - Ground Track (Image Track) Angle
    t = v_a - np.dot(v_a, z_hat) * z_hat
    groundtrack_ang2 = np.arctan2(np.dot(c_hat, t), np.dot(r_hat, t))
    groundtrack_ang3 = np.arctan2(np.dot(r_hat, t), np.dot(c_hat, t))

    if sidd_version == "2.0.0":
        exp_feat = {
            "{*}Collection/{*}Geometry/{*}Azimuth": np.degrees(azim_ang) % 360,
            "{*}Collection/{*}Geometry/{*}Slope": np.degrees(slope_ang),
            "{*}Collection/{*}Geometry/{*}Squint": np.degrees(squint_ang),
            "{*}Collection/{*}Geometry/{*}Graze": np.degrees(graze_ang),
            "{*}Collection/{*}Geometry/{*}Tilt": np.degrees(tilt_ang),
            "{*}Collection/{*}Geometry/{*}DopplerConeAngle": np.degrees(
                doppler_cone_ang
            ),
            "{*}Collection/{*}Phenomenology/{*}Shadow/{*}Angle": np.degrees(
                shadow_ang2
            ),
            "{*}Collection/{*}Phenomenology/{*}Shadow/{*}Magnitude": shadow_mag,
            "{*}Collection/{*}Phenomenology/{*}Layover/{*}Angle": np.degrees(
                layover_ang2
            ),
            "{*}Collection/{*}Phenomenology/{*}Layover/{*}Magnitude": layover_mag,
            "{*}Collection/{*}Phenomenology/{*}MultiPath": np.degrees(multipath_ang2),
            "{*}Collection/{*}Phenomenology/{*}GroundTrack": np.degrees(
                groundtrack_ang2
            ),
            "{*}Product/{*}North": np.degrees(north_ang2),
        }

    elif sidd_version == "3.0.0":
        exp_feat = {
            "{*}Collection/{*}Geometry/{*}Azimuth": np.degrees(azim_ang) % 360,
            "{*}Collection/{*}Geometry/{*}Slope": np.degrees(slope_ang),
            "{*}Collection/{*}Geometry/{*}Squint": np.degrees(squint_ang),
            "{*}Collection/{*}Geometry/{*}Graze": np.degrees(graze_ang),
            "{*}Collection/{*}Geometry/{*}Tilt": np.degrees(tilt_ang),
            "{*}Collection/{*}Geometry/{*}DopplerConeAngle": np.degrees(
                doppler_cone_ang
            ),
            "{*}Collection/{*}Phenomenology/{*}Shadow/{*}Angle": np.degrees(shadow_ang3)
            % 360,
            "{*}Collection/{*}Phenomenology/{*}Shadow/{*}Magnitude": shadow_mag,
            "{*}Collection/{*}Phenomenology/{*}Layover/{*}Angle": np.degrees(
                layover_ang3
            )
            % 360,
            "{*}Collection/{*}Phenomenology/{*}Layover/{*}Magnitude": layover_mag,
            "{*}Collection/{*}Phenomenology/{*}MultiPath": np.degrees(multipath_ang3)
            % 360,
            "{*}Collection/{*}Phenomenology/{*}GroundTrack": np.degrees(
                groundtrack_ang3
            )
            % 360,
            "{*}Product/{*}North": np.degrees(north_ang3) % 360,
        }
    else:
        raise ValueError(f"SIDD version {sidd_version} is not supported.")

    return exp_feat


def calc_geodata_imagecorners(sidd_xml):
    """Return approximate image corners (in ECEF) derived from a SIDD Measurement element (SIDD2.0, Sec. 3)"""
    meas_proj_elem = sidd_xml.find("./{*}Measurement/*[1]")
    ref_pt = sksidd.XyzType().parse_elem(
        meas_proj_elem.find("./{*}ReferencePoint/{*}ECEF")
    )
    r_0, c_0 = sksidd.RowColDblType().parse_elem(
        meas_proj_elem.find("./{*}ReferencePoint/{*}Point")
    )
    n_rows, n_cols = sksidd.RowColIntType().parse_elem(
        sidd_xml.find("./{*}Measurement/{*}PixelFootprint")
    )

    def passthrough(*args):
        return args

    rc_to_sensor_coord = passthrough

    if (ss_elem := meas_proj_elem.find("./{*}SampleSpacing")) is not None:
        delta_r, delta_c = sksidd.RowColDblType().parse_elem(ss_elem)

        def rc_to_sensor_grid_distance(r, c):
            return delta_r * (r - r_0), delta_c * (c - c_0)

        rc_to_sensor_coord = rc_to_sensor_grid_distance

    sensor_coord_to_ecef = None
    localname = lxml.etree.QName(meas_proj_elem).localname
    if localname == "PlaneProjection":  # 3.2 - PGD Pixel to ECEF
        p_pgd = ref_pt
        r_pgd = sksidd.XyzType().parse_elem(
            meas_proj_elem.find("./{*}ProductPlane/{*}RowUnitVector")
        )
        c_pgd = sksidd.XyzType().parse_elem(
            meas_proj_elem.find("./{*}ProductPlane/{*}ColUnitVector")
        )

        def sensor_coord_to_ecef_pgd(d_r, d_c):
            reshape_vec = np.append(np.ones(d_r.ndim, dtype=int), -1)
            return (
                p_pgd.reshape(reshape_vec)
                + np.multiply.outer(d_r, r_pgd)
                + np.multiply.outer(d_c, c_pgd)
            )

        sensor_coord_to_ecef = sensor_coord_to_ecef_pgd

    elif localname == "GeographicProjection":  # 3.4 GGD Pixel to Geodetic
        lon_0, lat_0, h_0 = wgs84.cartesian_to_geodetic(ref_pt)

        def sensor_coord_to_ecef_ggd(d_r, d_c):
            d_r_, d_c_, h_0_ = np.broadcast_arrays(d_r, d_c, h_0)
            pt_llh = np.stack(
                [
                    np.radians(lon_0 + d_c_ / 3600),
                    np.radians(lat_0 - d_r_ / 3600),
                    h_0_,
                ],
                -1,
            )
            return wgs84.geodetic_to_cartesian(pt_llh)

        sensor_coord_to_ecef = sensor_coord_to_ecef_ggd
    else:
        raise NotImplementedError(f"{localname} not supported.")

    # use pixel-as-area
    r_corner = np.array([-0.5, -0.5, n_rows - 0.5, n_rows - 0.5])
    c_corner = np.array([-0.5, n_cols - 0.5, n_cols - 0.5, -0.5])
    return sensor_coord_to_ecef(*rc_to_sensor_coord(r_corner, c_corner))
