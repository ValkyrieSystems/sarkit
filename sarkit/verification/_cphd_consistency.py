"""
Functionality for verifying CPHD files for internal consistency.
"""

import collections
import collections.abc
import copy
import functools
import itertools
import logging
import numbers
import os
import re
import struct
from typing import Any, Optional

import numpy as np
import numpy.polynomial.polynomial as npp
from lxml import etree

import sarkit.cphd as skcphd
import sarkit.verification._consistency as con
from sarkit import _constants

logger = logging.getLogger(__name__)

try:
    import shapely.geometry as shg
except ImportError as ie:
    logger.warning("'shapely' package not found. Some features may not work correctly.")
    shg = con._ExceptionOnUse(ie)

INVALID_CHAR_REGEX = re.compile(r"\W")


def per_channel(method):
    """Decorator to mark check methods as being applicable to each CPHD channel

    Parameters
    ----------
    method : Callable
        Method to mark

    Returns
    -------
    Callable
        Marked input `method`
    """

    method.per_channel = True
    return method


def get_by_id(xml, path, id_val):
    """Matches the first element that has a child named Identifier whose text is id_val.

    Parameters
    ----------
    xml : etree.Element
        Root node of XPath expression
    path : str
        XPath expression relative to xml
    id_val : str
        Value of child Identifier node

    Returns
    -------
    None|etree.Element
        node found by path with an Identifier node with value of id_val or None if a match is not found
    """

    return xml.find(f'{path}[{{*}}Identifier="{id_val}"]')


class CphdConsistency(con.ConsistencyChecker):
    """Check CPHD file structure and metadata for internal consistency

    `CphdConsistency` objects should be instantiated using `from_file` or `from_parts`.

    Parameters
    ----------
    cphd_xml : lxml.etree.Element or lxml.etree.ElementTree
        CPHD XML
    file_type_header : str, optional
        File type header from the first line of the file
    kvp_list : dict of {str : str}, optional
        Key-Value pair list of header fields
    pvps : dict of {str : ndarray}, optional
        CPHD Per-Vector-Parameters keyed by channel identifier
    schema_override : `path-like object`, optional
        Path to XML Schema. If None, tries to find a version-specific schema
    file : `file object`, optional
        CPHD file; when specified, portions of the file not specified in other parameters may be read
    """

    def __init__(
        self,
        cphd_xml,
        *,
        file_type_header=None,
        kvp_list=None,
        pvps=None,
        schema_override=None,
        file=None,
    ):
        super().__init__()
        # handle element or tree -> element
        try:
            self.cphdroot = cphd_xml.getroot()
        except AttributeError:
            self.cphdroot = cphd_xml.getroottree().getroot()
        self.xmlhelp = skcphd.XmlHelper(self.cphdroot.getroottree())

        self.file_type_header = file_type_header
        self.kvp_list = kvp_list
        self.pvps = pvps

        ns = etree.QName(self.cphdroot).namespace
        self.schema = schema_override or skcphd.VERSION_INFO.get(ns, {}).get("schema")

        self.file = file

        channel_ids = [
            x.text for x in self.cphdroot.findall("./{*}Data/{*}Channel/{*}Identifier")
        ]

        # process decorated methods to generate per-channel tests
        # reverse the enumerated list so that we don't disturb indices on later iterations as we insert into the list
        for index, func in reversed(list(enumerate(self.funcs))):
            if getattr(func, "per_channel", False):
                subfuncs = []
                for channel_id in channel_ids:
                    channel_node = get_by_id(
                        self.cphdroot, "./{*}Channel/{*}Parameters", channel_id
                    )
                    subfunc = functools.partial(func, channel_id, channel_node)
                    this_doc = func.__doc__.strip()
                    if this_doc.endswith("."):
                        this_doc = this_doc[:-1]
                    subfunc.__doc__ = f"{this_doc} for channel {channel_id}."
                    modified_channel_id = re.sub(INVALID_CHAR_REGEX, "_", channel_id)
                    subfunc.__name__ = f"{func.__name__}_{modified_channel_id}"
                    subfuncs.append(subfunc)
                self.funcs[index : index + 1] = subfuncs

    @staticmethod
    def from_file(
        file,
        schema: Optional[str] = None,
        thorough: bool = False,
    ) -> "CphdConsistency":
        """Create a CphdConsistency object from a file

        Parameters
        ----------
        file : `file object`
            CPHD or CPHD XML file to check
        schema : str, optional
            Path to XML Schema. If None, tries to find a version-specific schema
        thorough : bool, optional
            Run checks that may seek/read through large portions of the file.
            file must stay open to run checks. Ignored if file is CPHD XML.

        Returns
        -------
        CphdConsistency
            The initialized consistency checker object

        See Also
        --------
        from_parts

        Examples
        --------
        Use `from_file` to check an XML file:

        .. testsetup::

            import sarkit.cphd as skcphd
            import lxml.etree
            meta = skcphd.Metadata(
                xmltree=lxml.etree.parse("data/example-cphd-1.0.1.xml"),
            )
            file = tmppath / "example.cphd"
            with file.open("wb") as f, skcphd.Writer(f, meta) as w:
                f.seek(
                    w._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"]
                    + w._file_header_kvp["SIGNAL_BLOCK_SIZE"]
                    - 1
                )
                f.write(b"0")

        .. doctest::

            >>> import sarkit.verification as skver

            >>> with open("data/example-cphd-1.0.1.xml", "r") as f:
            ...     con = skver.CphdConsistency.from_file(f)
            >>> con.check()
            >>> bool(con.passes())
            True
            >>> bool(con.failures())
            False

        Use `from_file` to check a CPHD file, with and without thorough checks:

        .. doctest::

            >>> with file.open("rb") as f:
            ...     con_thorough = skver.CphdConsistency.from_file(f, thorough=True)
            ...     con = skver.CphdConsistency.from_file(f)
            ...     con_thorough.check()  # thorough checks require open file
            >>> con.check()  # without thorough, open file only used for construction
            >>> print(len(con.skips()) > len(con_thorough.skips()))
            True
        """
        kwargs: dict[str, Any] = {"schema_override": schema}
        try:
            cphd_xmltree = etree.parse(file)
        except etree.XMLSyntaxError:
            file.seek(0, os.SEEK_SET)
            reader = skcphd.Reader(file)
            cphd_xmltree = reader.metadata.xmltree
            file.seek(0, os.SEEK_SET)
            file_type_header, kvp_list = skcphd.read_file_header(file)
            pvps = {}
            for channel_node in cphd_xmltree.findall("./{*}Data/{*}Channel"):
                channel_id = channel_node.findtext("./{*}Identifier")
                pvps[channel_id] = reader.read_pvps(channel_id)
            kwargs.update(
                {
                    "file_type_header": file_type_header,
                    "kvp_list": kvp_list,
                    "pvps": pvps,
                }
            )
            if thorough:
                kwargs["file"] = file

        return CphdConsistency(
            cphd_xmltree,
            **kwargs,
        )

    @staticmethod
    def from_parts(
        cphd_xml: "etree.Element | etree.ElementTree",
        file_type_header: Optional[str] = None,
        kvp_list: Optional[dict[str, str]] = None,
        pvps: Optional[dict[str, np.ndarray]] = None,
        schema: Optional[str] = None,
    ) -> "CphdConsistency":
        """Create a CphdConsistency object from assorted parts

        Parameters
        ----------
        cphd_xml : lxml.etree.Element or lxml.etree.ElementTree
            CPHD XML
        file_type_header : str, optional
            File type header from the first line of the file
        kvp_list : dict of {str : str}, optional
            Key-Value pair list of header fields
        pvps : dict of {str : ndarray], optional
            CPHD Per-Vector-Parameters keyed by channel identifier
        schema : str, optional
            Path to XML Schema. If None, tries to find a version-specific schema

        Returns
        -------
        CphdConsistency
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
            >>> cphd_xmltree = lxml.etree.parse("data/example-cphd-1.0.1.xml")
            >>> con = skver.CphdConsistency.from_parts(cphd_xmltree)
            >>> con.check()
            >>> bool(con.passes())
            True
            >>> bool(con.failures())
            False

        Use `from_parts` to check a parsed XML element tree and an invalid file type header:

        .. doctest::

            >>> con = skver.CphdConsistency.from_parts(cphd_xmltree, file_type_header="CPHD/INVALID\\n")
            >>> con.check()
            >>> bool(con.failures())
            True
        """
        return CphdConsistency(
            cphd_xml=cphd_xml,
            file_type_header=file_type_header,
            kvp_list=kvp_list,
            pvps=pvps,
            schema_override=schema,
        )

    def _get_channel_pvps(self, channel_id):
        """
        Returns the PVPs associated with the channel keyed by `channel_id` or raises an AssertionError.
        """
        assert self.pvps is not None
        assert channel_id in self.pvps
        return self.pvps[channel_id]

    def get_polygon(self, polygon_node, check=False, reverse=False):
        """Returns the polygon from the specified node, with basic polygon verification."""
        vertex_nodes = sorted(list(polygon_node), key=lambda x: int(x.attrib["index"]))
        polygon = np.asarray(
            [self.xmlhelp.load_elem(vertex) for vertex in vertex_nodes]
        )
        if check:
            with self.need("Polygon indices are all present"):
                assert [int(x.attrib["index"]) for x in vertex_nodes] == list(
                    range(1, len(vertex_nodes) + 1)
                )
            if "size" in polygon_node.attrib:
                size = int(polygon_node.attrib["size"])
                with self.need("Polygon size attribute matches the number of vertices"):
                    assert size == len(vertex_nodes)
            shg_polygon = shg.Polygon(polygon)
            with self.need("Polygon is simple"):
                assert shg_polygon.is_simple
            with self.need("Polygon is clockwise"):
                assert not shg_polygon.exterior.is_ccw
        return polygon

    def check_file_type_header(self):
        """File type header is consistent with the XML."""
        with self.precondition():
            assert self.file_type_header is not None
            version = skcphd.VERSION_INFO.get(
                etree.QName(self.cphdroot).namespace, {}
            ).get("version")
            assert version is not None
            with self.need("File type header is consistent with the XML"):
                assert self.file_type_header == f"CPHD/{version}\n"

    def check_header_kvp_list(self):
        """Asserts that the required keys are in the header KVP list."""
        with self.precondition():
            assert self.kvp_list is not None
            required_fields = set(
                [
                    "XML_BLOCK_SIZE",
                    "XML_BLOCK_BYTE_OFFSET",
                    "PVP_BLOCK_SIZE",
                    "PVP_BLOCK_BYTE_OFFSET",
                    "SIGNAL_BLOCK_SIZE",
                    "SIGNAL_BLOCK_BYTE_OFFSET",
                    "CLASSIFICATION",
                    "RELEASE_INFO",
                ]
            )
            for name in required_fields:
                with self.need(f"Required KVP field: {name} is in KVP list"):
                    assert name in self.kvp_list

            has_sb_size = "SUPPORT_BLOCK_SIZE" in self.kvp_list.keys()
            has_sb_offset = "SUPPORT_BLOCK_BYTE_OFFSET" in self.kvp_list.keys()
            positive_sb_size = (
                has_sb_size and int(self.kvp_list["SUPPORT_BLOCK_SIZE"]) > 0
            )
            with self.need("SUPPORT_BLOCK fields go together"):
                assert not (has_sb_size or has_sb_offset) or (
                    positive_sb_size and has_sb_offset
                )

    def check_classification_and_release_info(self):
        """Asserts that the Classification and ReleaseInfo fields are the same in header KVP list and the xml."""
        with self.precondition():
            assert self.kvp_list is not None
            with self.need("KVP list CLASSIFICATION matches XML Classification"):
                assert self.kvp_list["CLASSIFICATION"] == self.xmlhelp.load(
                    "./{*}CollectionID/{*}Classification"
                )
            with self.need("KVP list RELEASE_INFO matches XML ReleaseInfo"):
                assert self.kvp_list["RELEASE_INFO"] == self.xmlhelp.load(
                    "./{*}CollectionID/{*}ReleaseInfo"
                )

    def check_against_schema(self):
        """The XML matches the schema."""
        with self.need(
            f"Schema available for checking xml whose root tag = {self.cphdroot.tag}"
        ):
            assert self.schema is not None
            schema = etree.XMLSchema(file=str(self.schema))
            with self.need("XML passes schema"):
                assert schema.validate(self.cphdroot), schema.error_log

    @per_channel
    def check_channel_dwell_polys(self, channel_id, channel_node):
        """/Dwell/CODTime/CODTimePoly and /Dwell/DwellTime/DwellTimePoly are consistent with other metadata."""
        cod_id = channel_node.findtext("./{*}DwellTimes/{*}CODId")
        cod_node = get_by_id(self.cphdroot, "./{*}Dwell/{*}CODTime", cod_id)
        dwell_id = channel_node.findtext("./{*}DwellTimes/{*}DwellId")
        dwell_node = get_by_id(self.cphdroot, "./{*}Dwell/{*}DwellTime", dwell_id)
        with self.need(
            f"/Dwell/DwellTime with Identifier={dwell_id} exists for DwellTime in channel={channel_id}"
        ):
            assert dwell_node is not None

        codtime_poly = self.xmlhelp.load_elem(cod_node.find("./{*}CODTimePoly"))
        dwelltime_poly = self.xmlhelp.load_elem(dwell_node.find("./{*}DwellTimePoly"))

        def _get_image_area_polygon(image_area_elem):
            if image_area_elem.find("./{*}Polygon") is not None:
                return shg.Polygon(
                    self.xmlhelp.load_elem(image_area_elem.find("./{*}Polygon"))
                )
            x1, y1 = self.xmlhelp.load_elem(image_area_elem.find("./{*}X1Y1"))
            x2, y2 = self.xmlhelp.load_elem(image_area_elem.find("./{*}X2Y2"))
            return shg.box(x1, y1, x2, y2)

        image_area_elem = channel_node.find("./{*}ImageArea")
        if image_area_elem is None:
            image_area_elem = self.cphdroot.find("./{*}SceneCoordinates/{*}ImageArea")
        image_area_polygon = _get_image_area_polygon(image_area_elem)

        def _get_points_in_polygon(polygon, grid_size=25):
            bounds = np.asarray(polygon.bounds).reshape(
                2, 2
            )  # [[xmin, ymin], [xmax, ymax]]
            mesh = np.stack(
                np.meshgrid(
                    np.linspace(bounds[0, 0], bounds[1, 0], grid_size),
                    np.linspace(bounds[0, 1], bounds[1, 1], grid_size),
                ),
                axis=-1,
            )
            coords = shg.MultiPoint(
                np.concatenate(
                    [mesh.reshape(-1, 2), np.asarray(polygon.exterior.coords)[:-1, :]],
                    axis=0,
                )
            )
            return np.asarray([pt.coords for pt in polygon.intersection(coords).geoms])

        sampled_iacs = _get_points_in_polygon(image_area_polygon).T
        sampled_cods = npp.polyval2d(*sampled_iacs, codtime_poly)
        sampled_dwells = npp.polyval2d(*sampled_iacs, dwelltime_poly)
        with self.need("/Dwell/DwellTime/DwellTimePoly is nonnegative in image area"):
            assert sampled_dwells.min() >= 0.0

        sampled_tref1 = sampled_cods - 0.5 * sampled_dwells
        sampled_tref2 = sampled_cods + 0.5 * sampled_dwells
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            mask = np.isfinite(pvp["TxTime"])

            def calc_tref(v):
                r_xmt = np.linalg.norm(v["TxPos"] - v["SRPPos"])
                r_rcv = np.linalg.norm(v["RcvPos"] - v["SRPPos"])
                return v["TxTime"] + r_xmt / (r_xmt + r_rcv) * (
                    v["RcvTime"] - v["TxTime"]
                )

            pvps_tref1 = calc_tref(pvp[mask][0])
            pvps_tref2 = calc_tref(pvp[mask][-1])

            with self.need(
                "/Dwell/CODTime/CODTimePoly and /Dwell/DwellTime/DwellTimePoly supported by PVPs"
            ):
                assert sampled_tref1.min() >= con.Approx(pvps_tref1, atol=100e-6)
                assert sampled_tref2.max() <= con.Approx(pvps_tref2, atol=100e-6)

    def check_antenna(self):
        """Check that antenna node is consistent."""
        with self.precondition():
            antenna_node = self.cphdroot.find("./{*}Antenna")
            assert antenna_node is not None

            expected_num_acfs = self.xmlhelp.load("./{*}Antenna/{*}NumACFs")
            actual_num_acfs = len(antenna_node.findall("./{*}AntCoordFrame"))
            with self.need("The NumACFs must be equal to the number of ACF nodes."):
                assert expected_num_acfs == actual_num_acfs

            expected_num_apcs = self.xmlhelp.load("./{*}Antenna/{*}NumAPCs")
            actual_num_apcs = len(antenna_node.findall("./{*}AntPhaseCenter"))
            with self.need("The NumAPCs must be equal to the number of APC nodes."):
                assert expected_num_apcs == actual_num_apcs

            expected_num_antpats = self.xmlhelp.load("./{*}Antenna/{*}NumAntPats")
            actual_num_antpats = len(antenna_node.findall("./{*}AntPattern"))
            with self.need(
                "The NumAntPats must be equal to the number of AntPattern nodes."
            ):
                assert expected_num_antpats == actual_num_antpats

            apc_acfids = antenna_node.findall("./{*}AntPhaseCenter/{*}ACFId")
            apc_acf_ids_text = {apc_acfid.text for apc_acfid in apc_acfids}
            acf_identifiers = antenna_node.findall("./{*}AntCoordFrame/{*}Identifier")
            acf_identifiers_text = {
                acf_identifier.text for acf_identifier in acf_identifiers
            }
            with self.need(
                "./AntPhaseCenter/ACFId references an identifier in AntCoordFrame."
            ):
                assert apc_acf_ids_text <= acf_identifiers_text

    @per_channel
    def check_channel_antenna_exist(self, channel_id, channel_node):
        """The antenna patterns and phase centers exist if declared."""
        with self.precondition():
            assert channel_node.find("./{*}Antenna") is not None
            for side in "Tx", "Rcv":
                apc_id = self.xmlhelp.load_elem(
                    channel_node.find(f"./{{*}}Antenna/{{*}}{side}APCId")
                )
                with self.need(
                    f"AntPhaseCenter node exists with name {apc_id} (for {side})"
                ):
                    assert (
                        get_by_id(
                            self.cphdroot, "./{*}Antenna/{*}AntPhaseCenter/", apc_id
                        )
                        is not None
                    )
                apat_id = self.xmlhelp.load_elem(
                    channel_node.find(f"./{{*}}Antenna/{{*}}{side}APATId")
                )
                with self.need(
                    f"AntPattern node exists with name {apat_id} (for {side})"
                ):
                    assert (
                        get_by_id(self.cphdroot, "./{*}Antenna/{*}AntPattern/", apat_id)
                        is not None
                    )

    @per_channel
    def check_channel_txrcv_exist(self, channel_id, channel_node):
        """The declared TxRcv nodes exist."""
        with self.precondition():
            assert channel_node.find("./{*}TxRcv") is not None
            for tx_wf_id in channel_node.findall("./{*}TxRcv/{*}TxWFId"):
                with self.need(f"TxWFParameters node exists with id {tx_wf_id.text}"):
                    assert (
                        get_by_id(
                            self.cphdroot, "./{*}TxRcv/{*}TxWFParameters", tx_wf_id.text
                        )
                        is not None
                    )
            for rcv_id in channel_node.findall("./{*}TxRcv/{*}RcvId"):
                with self.need(f"RcvParameters node exists with id {rcv_id.text}"):
                    assert (
                        get_by_id(
                            self.cphdroot, "./{*}TxRcv/{*}RcvParameters", rcv_id.text
                        )
                        is not None
                    )

    @per_channel
    def check_time_increasing(self, channel_id, channel_node):
        """PVP times increase."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            for side in "Tx", "Rcv":
                spvp = pvp[f"{side}Time"]
                mask = np.isfinite(spvp)
                with self.need(f"{side}Time strictly increasing"):
                    assert np.all(np.greater(np.diff(spvp[mask]), 0))

    @per_channel
    def check_rcv_after_tx(self, channel_id, channel_node):
        """RcvTime is after TxTime."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            tx_time = pvp["TxTime"]
            rcv_time = pvp["RcvTime"]
            mask = np.logical_and(
                np.isfinite(pvp["TxTime"]), np.isfinite(pvp["RcvTime"])
            )
            with self.need("Rcv after Tx"):
                assert np.all(np.greater(rcv_time[mask], tx_time[mask]))

    @per_channel
    def check_rcv_finite(self, channel_id, channel_node):
        """RcvTime and Pos are finite."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            rcv_time = pvp["RcvTime"]
            rcv_pos = pvp["RcvPos"]
            with self.need("RcvTime"):
                assert np.all(np.isfinite(rcv_time))
            with self.need("RcvPos"):
                assert np.all(np.isfinite(rcv_pos))

    @per_channel
    def check_channel_fxfixed(self, channel_id, channel_node):
        """PVP agrees with FXFixed."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            fx1_ptp = np.ptp([np.nanmin(pvp["FX1"]), np.nanmax(pvp["FX1"])])
            fx2_ptp = np.ptp([np.nanmin(pvp["FX2"]), np.nanmax(pvp["FX2"])])
            with self.precondition():
                assert self.xmlhelp.load_elem(channel_node.find(".{*}FXFixed"))
                with self.need("FX1 does not change"):
                    assert fx1_ptp == 0
                with self.need("FX2 does not change"):
                    assert fx2_ptp == 0

            with self.precondition():
                assert not self.xmlhelp.load_elem(channel_node.find("./{*}FXFixed"))
                with self.need("FX1 or FX2 does change"):
                    assert fx1_ptp != 0 or fx2_ptp != 0

    @per_channel
    def check_channel_toafixed(self, channel_id, channel_node):
        """PVP agrees with TOAFixed."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            toa1_ptp = np.ptp([np.nanmin(pvp["TOA1"]), np.nanmax(pvp["TOA1"])])
            toa2_ptp = np.ptp([np.nanmin(pvp["TOA2"]), np.nanmax(pvp["TOA2"])])
            with self.precondition():
                assert self.xmlhelp.load_elem(channel_node.find("./{*}TOAFixed"))
                with self.need("TOA1 does not change"):
                    assert toa1_ptp == 0
                with self.need("TOA2 does not change"):
                    assert toa2_ptp == 0

            with self.precondition():
                assert not self.xmlhelp.load_elem(channel_node.find("./{*}TOAFixed"))
                with self.need("TOA1 or TOA2 does change"):
                    assert toa1_ptp != 0 or toa2_ptp != 0

    @per_channel
    def check_channel_srpfixed(self, channel_id, channel_node):
        """PVP agrees with SRPFixed."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.precondition():
                assert self.xmlhelp.load_elem(channel_node.find("./{*}SRPFixed"))
                with self.need("SRPPos is fixed"):
                    assert np.all(pvp["SRPPos"] == pvp["SRPPos"][0])

            with self.precondition():
                assert not self.xmlhelp.load_elem(channel_node.find("./{*}SRPFixed"))
                with self.need("SRPPos is not exactly fixed"):
                    assert np.any(pvp["SRPPos"] != pvp["SRPPos"][0])

    def check_file_fxfixed(self):
        """The FXFixedCPHD element matches the rest of the file."""
        fxc_vals = np.array(
            [
                self.xmlhelp.load_elem(elem)
                for elem in self.cphdroot.findall("./{*}Channel/{*}Parameters/{*}FxC")
            ]
        )
        fxc_minmax = np.array([fxc_vals.min(), fxc_vals.max()])
        fxc_tol = con.Approx(fxc_vals.mean())
        fxbw_vals = np.array(
            [
                self.xmlhelp.load_elem(elem)
                for elem in self.cphdroot.findall("./{*}Channel/{*}Parameters/{*}FxBW")
            ]
        )
        fxbw_minmax = np.array([fxbw_vals.min(), fxbw_vals.max()])
        fxbw_tol = con.Approx(fxbw_vals.mean())

        chan_fxfixed = self.cphdroot.findall("./{*}Channel/{*}Parameters/{*}FXFixed")
        with self.precondition():
            assert self.xmlhelp.load("./{*}Channel/{*}FXFixedCPHD")
            with self.need("All channels have FXFixed"):
                assert all(self.xmlhelp.load_elem(elem) for elem in chan_fxfixed)
            with self.need("All channels have same FxC"):
                assert fxc_minmax == fxc_tol
            with self.need("All channels have same FxBW"):
                assert fxbw_minmax == fxbw_tol

        with self.precondition():
            assert not self.xmlhelp.load("./{*}Channel/{*}FXFixedCPHD")
            assert all(self.xmlhelp.load_elem(elem) for elem in chan_fxfixed)
            with self.need("Channels are not the same"):
                assert not (
                    fxc_vals.min() == fxc_vals.max()
                    and fxbw_vals.min() == fxbw_vals.max()
                )

        with self.precondition():
            assert self.pvps is not None
            pvp = np.concatenate(list(self.pvps.values()))
            fx1_ptp = np.ptp([np.nanmin(pvp["FX1"]), np.nanmax(pvp["FX1"])])
            fx2_ptp = np.ptp([np.nanmin(pvp["FX2"]), np.nanmax(pvp["FX2"])])
            with self.precondition():
                assert self.xmlhelp.load("./{*}Channel/{*}FXFixedCPHD")
                with self.need("FX1 does not change"):
                    assert fx1_ptp == 0
                with self.need("FX2 does not change"):
                    assert fx2_ptp == 0

            with self.precondition():
                assert not self.xmlhelp.load("./{*}Channel/{*}FXFixedCPHD")
                with self.need("FX1 or FX2 does change"):
                    assert fx1_ptp != 0 or fx2_ptp != 0

    def check_file_toafixed(self):
        """The TOAFixedCPHD element matches the rest of the file."""
        toa_saved_vals = np.array(
            [
                self.xmlhelp.load_elem(elem)
                for elem in self.cphdroot.findall(
                    "./{*}Channel/{*}Parameters/{*}TOASaved"
                )
            ]
        )
        toa_saved_minmax = np.array([toa_saved_vals.min(), toa_saved_vals.max()])
        toa_saved_tol = con.Approx(toa_saved_minmax.mean())

        chan_toafixed = self.cphdroot.findall("./{*}Channel/{*}Parameters/{*}TOAFixed")
        with self.precondition():
            assert self.xmlhelp.load("./{*}Channel/{*}TOAFixedCPHD")
            with self.need("All channels have TOAFixed"):
                assert all(self.xmlhelp.load_elem(elem) for elem in chan_toafixed)
            with self.need("All channels have same TOASaved"):
                assert toa_saved_minmax == toa_saved_tol

        with self.precondition():
            assert not self.xmlhelp.load("./{*}Channel/{*}TOAFixedCPHD")
            assert all(self.xmlhelp.load_elem(elem) for elem in chan_toafixed)
            with self.need("Channels are not the same"):
                assert not (toa_saved_vals.min() == toa_saved_vals.max())

        with self.precondition():
            assert self.pvps is not None
            pvp = np.concatenate(list(self.pvps.values()))
            toa1_ptp = np.ptp([np.nanmin(pvp["TOA1"]), np.nanmax(pvp["TOA1"])])
            toa2_ptp = np.ptp([np.nanmin(pvp["TOA2"]), np.nanmax(pvp["TOA2"])])
            with self.precondition():
                assert self.xmlhelp.load("./{*}Channel/{*}TOAFixedCPHD")
                with self.need("TOA1 does not change"):
                    assert toa1_ptp == 0
                with self.need("TOA2 does not change"):
                    assert toa2_ptp == 0

            with self.precondition():
                assert not self.xmlhelp.load("./{*}Channel/{*}TOAFixedCPHD")
                with self.need("TOA1 or TOA2 does change"):
                    assert toa1_ptp != 0 or toa2_ptp != 0

    def check_file_srpfixed(self):
        """The SRPFixedCPHD element matches the rest of the file."""
        with self.precondition():
            assert self.xmlhelp.load("./{*}Channel/{*}SRPFixedCPHD")
            with self.need("All channels have SRPFixed"):
                assert all(
                    self.xmlhelp.load_elem(elem)
                    for elem in self.cphdroot.findall(
                        "./{*}Channel/{*}Parameters/{*}SRPFixed"
                    )
                )

        with self.precondition():
            assert self.pvps is not None
            pvp = np.concatenate(list(self.pvps.values()))
            with self.precondition():
                assert self.xmlhelp.load("./{*}Channel/{*}SRPFixedCPHD")
                with self.need("SRPPos is fixed"):
                    assert np.all(pvp["SRPPos"] == pvp["SRPPos"][0])

            with self.precondition():
                assert not self.xmlhelp.load("./{*}Channel/{*}SRPFixedCPHD")
                with self.need("SRPPos is not exactly fixed"):
                    assert np.any(pvp["SRPPos"] != pvp["SRPPos"][0])

    @per_channel
    def check_channel_signalnormal(self, channel_id, channel_node):
        """PVP agrees with SignalNormal."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            assert channel_node.find("./{*}SignalNormal") is not None
            with self.need("SIGNAL PVP present"):
                assert "SIGNAL" in pvp.dtype.names
                with self.need("SignalNormal matches SIGNAL PVPs"):
                    assert np.all(pvp["SIGNAL"] == 1) == self.xmlhelp.load_elem(
                        channel_node.find("./{*}SignalNormal")
                    )

    @per_channel
    def check_channel_fxc(self, channel_id, channel_node):
        """PVP agrees with FxC."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("FxC is (max(fx2) + min(fx1)) / 2"):
                assert (
                    con.Approx(self.xmlhelp.load_elem(channel_node.find("./{*}FxC")))
                    == (np.nanmax(pvp["FX2"]) + np.nanmin(pvp["FX1"])) / 2
                )

    @per_channel
    def check_channel_fxbw(self, channel_id, channel_node):
        """PVP agrees with FxBW."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("FxBW is max(fx2) - min(fx1)"):
                assert con.Approx(
                    self.xmlhelp.load_elem(channel_node.find("./{*}FxBW"))
                ) == np.nanmax(pvp["FX2"]) - np.nanmin(pvp["FX1"])

    @per_channel
    def check_channel_fxbwnoise(self, channel_id, channel_node):
        """PVP agrees with FxBWNoise."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            assert channel_node.find("./{*}FxBWNoise") is not None
            with self.need("Domain is FX when FxBWNoise is provided"):
                assert self.xmlhelp.load("./{*}Global/{*}DomainType") == "FX"
            with self.need("FxBWNoise is max(FXN2) - min(FXN1)"):
                assert con.Approx(
                    self.xmlhelp.load_elem(channel_node.find("./{*}FxBWNoise"))
                ) == np.nanmax(pvp["FXN2"]) - np.nanmin(pvp["FXN1"])

    @per_channel
    def check_channel_toasaved(self, channel_id, channel_node):
        """PVP agrees with TOASaved."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("TOASaved is max(TOA2) - min(TOA1)"):
                assert con.Approx(
                    self.xmlhelp.load_elem(channel_node.find("./{*}TOASaved"))
                ) == np.nanmax(pvp["TOA2"]) - np.nanmin(pvp["TOA1"])

    @per_channel
    def check_channel_toaextsaved(self, channel_id, channel_node):
        """PVP agrees with TOAExtSaved."""
        toa_ext_saved = channel_node.find("./{*}TOAExtended/{*}TOAExtSaved")
        has_toa_ext_saved = toa_ext_saved is not None
        has_toae1 = self.xmlhelp.load("./{*}PVP/{*}TOAE1") is not None
        has_toae2 = self.xmlhelp.load("./{*}PVP/{*}TOAE2") is not None
        with self.want("TOA extended swath parameters are specified together"):
            assert has_toa_ext_saved == has_toae1 == has_toae2
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            assert has_toa_ext_saved
            assert {"TOAE1", "TOAE2"}.issubset(pvp.dtype.fields)
            with self.need("TOAExtSaved is max(TOAE2) - min(TOAE1)"):
                assert con.Approx(self.xmlhelp.load_elem(toa_ext_saved)) == np.nanmax(
                    pvp["TOAE2"]
                ) - np.nanmin(pvp["TOAE1"])

    @per_channel
    def check_channel_fx_osr(self, channel_id, channel_node):
        """FX domain vectors are sufficiently sampled"""
        with self.precondition():
            assert self.xmlhelp.load("./{*}Global/{*}DomainType") == "FX"
            pvp = self._get_channel_pvps(channel_id)
            if {"TOAE1", "TOAE2"}.issubset(pvp.dtype.fields):
                toa_xtnt = pvp["TOAE2"] - pvp["TOAE1"]
            else:
                toa_xtnt = pvp["TOA2"] - pvp["TOA1"]
            fx_osr = 1 / (pvp["SCSS"] * toa_xtnt)
            with self.need("FX_OSR is at least 1.1"):
                assert np.nanmin(fx_osr) >= 1.1
            with self.want("FX_OSR is at least 1.2"):
                assert np.nanmin(fx_osr) >= 1.2

    @per_channel
    def check_channel_toa_osr(self, channel_id, channel_node):
        """TOA domain vectors are sufficiently sampled"""
        with self.precondition():
            assert self.xmlhelp.load("./{*}Global/{*}DomainType") == "TOA"
            pvp = self._get_channel_pvps(channel_id)
            fx_bw = pvp["FX2"] - pvp["FX1"]
            toa_osr = 1 / (pvp["SCSS"] * fx_bw)
            with self.need("TOA_OSR is at least 1.1"):
                assert np.nanmin(toa_osr) >= 1.1
            with self.want("TOA_OSR is at least 1.2"):
                assert np.nanmin(toa_osr) >= 1.2

    @per_channel
    def check_channel_global_txtime(self, channel_id, channel_node):
        """PVP within global TxTime1 and TxTime2."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("TxTime is greater than TxTime1"):
                assert np.nanmin(pvp["TxTime"]) >= con.Approx(
                    float(self.xmlhelp.load("./{*}Global/{*}Timeline/{*}TxTime1"))
                )
            with self.need("TxTime is less than TxTime2"):
                assert np.nanmax(pvp["TxTime"]) <= con.Approx(
                    float(self.xmlhelp.load("./{*}Global/{*}Timeline/{*}TxTime2"))
                )

    @per_channel
    def check_channel_global_fxminmax(self, channel_id, channel_node):
        """PVP within global FxMin and FxMax."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("FX1 is greater than FxMin"):
                assert np.nanmin(pvp["FX1"]) >= con.Approx(
                    float(self.xmlhelp.load("./{*}Global/{*}FxBand/{*}FxMin"))
                )
            with self.need("FX2 is less than FxMax"):
                assert np.nanmax(pvp["FX2"]) <= con.Approx(
                    float(self.xmlhelp.load("./{*}Global/{*}FxBand/{*}FxMax"))
                )

    @per_channel
    def check_channel_global_toaswath(self, channel_id, channel_node):
        """PVP within global TOASwath."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("TOA1 is greater than TOAMin"):
                assert np.nanmin(pvp["TOA1"]) >= con.Approx(
                    float(self.xmlhelp.load("./{*}Global/{*}TOASwath/{*}TOAMin"))
                )
            with self.need("TOA2 is less than TOAMax"):
                assert np.nanmax(pvp["TOA2"]) <= con.Approx(
                    float(self.xmlhelp.load("./{*}Global/{*}TOASwath/{*}TOAMax"))
                )

    @per_channel
    def check_channel_afdop(self, channel_id, channel_node):
        """aFDOP PVP is consistent with other PVPs."""

        def calc_rdot(pos, vel, srp):
            return (vel * unit(pos - srp)).sum(axis=-1)

        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            rdot_xmt_srp = calc_rdot(pvp["TxPos"], pvp["TxVel"], pvp["SRPPos"])
            rdot_rcv_srp = calc_rdot(pvp["RcvPos"], pvp["RcvVel"], pvp["SRPPos"])
            rdot_avg_srp = 0.5 * (rdot_xmt_srp + rdot_rcv_srp)
            afdop_expected = rdot_avg_srp * (-2 / _constants.c)
            mask = np.logical_and(
                np.isfinite(afdop_expected), np.isfinite(pvp["aFDOP"])
            )
            assert mask.any()
            assert np.count_nonzero(
                pvp["aFDOP"]
            )  # CPHD advises these "may be set equal to zero for all vectors"
            with self.want("aFDOP consistent with other PVPs"):
                assert afdop_expected[mask] == con.Approx(pvp["aFDOP"][mask], atol=1e-9)

    @per_channel
    def check_channel_afrr1_afrr2_relative(self, channel_id, channel_node):
        """aFRR1 & aFRR2 PVPs are related by fx_C."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            fx_c = 0.5 * (pvp["FX1"] + pvp["FX2"])
            # CPHD advises these "may be set equal to zero for all vectors"
            afrr1_afrr2_zero = (pvp["aFRR1"] == 0) & (pvp["aFRR2"] == 0)
            mask = (
                np.isfinite(fx_c)
                & np.isfinite(pvp["aFRR1"])
                & np.isfinite(pvp["aFRR2"])
                & ~afrr1_afrr2_zero
            )
            assert mask.any()
            with self.want("aFRR1 == (FX1 + FX2) * aFRR2 / 2"):
                assert pvp["aFRR1"][mask] / (
                    fx_c[mask] * pvp["aFRR2"][mask]
                ) == con.Approx(1)

    def _get_channel_tx_lfmrates(self, channel_node):
        tx_lfmrates = set()
        for txwdid_node in channel_node.findall("./{*}TxRcv/{*}TxWFId"):
            this_lfmrate = self.xmlhelp.load(
                f"./{{*}}TxRcv/{{*}}TxWFParameters[{{*}}Identifier='{txwdid_node.text}']/{{*}}LFMRate"
            )
            if this_lfmrate is not None:
                tx_lfmrates.add(float(this_lfmrate))
        assert tx_lfmrates
        return np.fromiter(tx_lfmrates, float)

    @per_channel
    def check_channel_afrr1(self, channel_id, channel_node):
        """aFRR1 is consistent with /TxRcv/TxWFParameters/LFMRate."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            fx_c = 0.5 * (pvp["FX1"] + pvp["FX2"])
            tx_lfmrates = self._get_channel_tx_lfmrates(channel_node)
            with np.errstate(divide="ignore"):
                derived_fx_rate = fx_c * 2 / (_constants.c * pvp["aFRR1"])
            mask = np.isfinite(derived_fx_rate)
            assert mask.any()
            derived_fx_matches_tx_lfmrates = np.isclose(
                derived_fx_rate[mask, np.newaxis], tx_lfmrates[np.newaxis, :]
            ).any(axis=1)
            inconsistent_derived_lfmrates = derived_fx_rate[mask][
                ~derived_fx_matches_tx_lfmrates
            ].tolist()
            with self.want(
                f"aFRR1 is consistent with /TxRcv/TxWFParameters/LFMRate(s): {tx_lfmrates}"
            ):
                assert not inconsistent_derived_lfmrates

    @per_channel
    def check_channel_afrr2(self, channel_id, channel_node):
        """aFRR2 is consistent with /TxRcv/TxWFParameters/LFMRate(s)."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            tx_lfmrates = self._get_channel_tx_lfmrates(channel_node)
            with np.errstate(divide="ignore"):
                derived_fx_rate = 2 / (_constants.c * pvp["aFRR2"])
            mask = np.isfinite(derived_fx_rate)
            assert mask.any()
            derived_fx_matches_tx_lfmrates = np.isclose(
                derived_fx_rate[mask, np.newaxis], tx_lfmrates[np.newaxis, :]
            ).any(axis=1)
            inconsistent_derived_lfmrates = derived_fx_rate[mask][
                ~derived_fx_matches_tx_lfmrates
            ].tolist()
            with self.want(
                f"aFRR2 is consistent with /TxRcv/TxWFParameters/LFMRate(s): {tx_lfmrates}"
            ):
                assert not inconsistent_derived_lfmrates

    @per_channel
    def check_channel_imagearea_polygon(self, channel_id, channel_node):
        """Image area polygon is simple and consistent with X1Y1 and X2Y2."""

        chan_ia_poly_node = channel_node.find("./{*}ImageArea/{*}Polygon")
        with self.precondition():
            assert chan_ia_poly_node is not None
            polygon = self.get_polygon(chan_ia_poly_node, check=True)
            x1y1 = self.xmlhelp.load_elem(channel_node.find("./{*}ImageArea/{*}X1Y1"))
            x2y2 = self.xmlhelp.load_elem(channel_node.find("./{*}ImageArea/{*}X2Y2"))
            with self.need("Polygon works with X1Y1"):
                assert polygon.min(axis=0) == con.Approx(x1y1, atol=1e-3)
            with self.need("Polygon works with X2Y2"):
                assert polygon.max(axis=0) == con.Approx(x2y2, atol=1e-3)

    @per_channel
    def check_channel_identifier_uniqueness(self, channel_id, channel_node):
        """Identifier nodes within /Channel/Parameters are unique."""
        identifier_sets = (
            {"./{*}TxRcv/{*}TxWFId"},
            {"./{*}TxRcv/{*}RcvId"},
        )
        for identifier_set in identifier_sets:
            these_identifiers = []
            for path in identifier_set:
                these_identifiers.extend(x.text for x in channel_node.findall(path))
            repeated_identifiers = _get_repeated_elements(these_identifiers)
            with self.want(f"Identifiers {identifier_set} are unique"):
                assert not repeated_identifiers

    @per_channel
    def check_channel_rcv_sample_rate(self, channel_id, channel_node):
        """/TxRcv/RcvParameters/SampleRate sufficient to support saved TOA swath."""
        toa_swath = float(
            channel_node.findtext("./{*}TOAExtended/{*}TOAExtSaved", np.nan)
        )
        if np.isnan(toa_swath):
            toa_swath = self.xmlhelp.load_elem(channel_node.find("./{*}TOASaved"))
        txwf_ids = {x.text for x in channel_node.findall("./{*}TxRcv/{*}TxWFId")}
        rcv_ids = {x.text for x in channel_node.findall("./{*}TxRcv/{*}RcvId")}
        with self.precondition():
            assert len(txwf_ids) == 1 and len(rcv_ids) == 1
            txwf_params = get_by_id(
                self.cphdroot, "./{*}TxRcv/{*}TxWFParameters", next(iter(txwf_ids))
            )
            rcv_params = get_by_id(
                self.cphdroot, "./{*}TxRcv/{*}RcvParameters", next(iter(rcv_ids))
            )
            tx_lfm_rate = float(txwf_params.findtext("./{*}LFMRate", np.nan))
            rcv_lfm_rate = float(rcv_params.findtext("./{*}LFMRate", np.nan))
            assert np.isfinite([tx_lfm_rate, rcv_lfm_rate]).all()
            tx_pulse_length = self.xmlhelp.load_elem(
                txwf_params.find("./{*}PulseLength")
            )
            rcv_sample_rate = self.xmlhelp.load_elem(rcv_params.find("./{*}SampleRate"))
            claimed_bw = abs(tx_lfm_rate - rcv_lfm_rate) * tx_pulse_length + abs(
                toa_swath * rcv_lfm_rate
            )
            with self.need(
                "/TxRcv/RcvParameters/SampleRate sufficient to support saved TOA swath"
            ):
                assert claimed_bw <= con.Approx(rcv_sample_rate)

    def check_imagearea_polygon(self):
        """Scene Image area polygon is simple and consistent with X1Y1 and X2Y2."""
        poly_node = self.cphdroot.find("./{*}SceneCoordinates/{*}ImageArea/{*}Polygon")
        with self.precondition():
            assert poly_node is not None
            polygon = self.get_polygon(poly_node, check=True)
            x1y1 = self.xmlhelp.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1")
            x2y2 = self.xmlhelp.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2")
            with self.need("Polygon works with X1Y1"):
                assert polygon.min(axis=0) == con.Approx(x1y1, atol=1e-3)
            with self.need("Polygon works with X2Y2"):
                assert polygon.max(axis=0) == con.Approx(x2y2, atol=1e-3)

    def check_geoinfo_polygons(self):
        """GeoInfo polygons are simple polygons in clockwise order."""
        geo_polygons = self.cphdroot.findall(".//{*}GeoInfo/{*}Polygon")
        with self.precondition():
            assert geo_polygons
            for geo_polygon in geo_polygons:
                with self.need(etree.ElementTree(self.cphdroot).getpath(geo_polygon)):
                    self.get_polygon(geo_polygon, check=True)

    def check_image_area_corner_points(self):
        """The corner points represent a simple quadrilateral in clockwise order."""
        iacp_node = self.cphdroot.find("./{*}SceneCoordinates/{*}ImageAreaCornerPoints")
        vertex_nodes = sorted(list(iacp_node), key=lambda x: int(x.attrib["index"]))
        polygon = np.asarray(
            [self.xmlhelp.load_elem(vertex)[::-1] for vertex in vertex_nodes]
        )
        with self.need("4 corner points"):
            assert len(polygon) == 4
        with self.need("Polygon indices are all present"):
            assert [int(x.attrib["index"]) for x in vertex_nodes] == list(
                range(1, len(vertex_nodes) + 1)
            )
            shg_polygon = shg.Polygon(polygon)
            with self.need("Polygon is simple"):
                assert shg_polygon.is_simple
            with self.need("Polygon is clockwise"):
                assert not shg_polygon.exterior.is_ccw

    def check_extended_imagearea_polygon(self):
        """Scene extended area polygon is simple and consistent with X1Y1 and X2Y2."""
        scene_coords_node = self.cphdroot.find("./{*}SceneCoordinates")
        with self.precondition():
            assert scene_coords_node.find("./{*}ExtendedArea") is not None
            extended_area_node = scene_coords_node.find("./{*}ExtendedArea")
            with self.precondition():
                assert extended_area_node.find("./{*}Polygon") is not None
                extended_area_polygon_node = extended_area_node.find("./{*}Polygon")
                extended_area_polygon = self.get_polygon(
                    extended_area_polygon_node, check=True
                )
                extended_x1y1 = self.xmlhelp.load_elem(
                    extended_area_node.find("./{*}X1Y1")
                )
                extended_x2y2 = self.xmlhelp.load_elem(
                    extended_area_node.find("./{*}X2Y2")
                )

                with self.need("Polygon works with X1Y1"):
                    assert extended_area_polygon.min(axis=0) == con.Approx(
                        extended_x1y1, atol=1e-3
                    )
                with self.need("Polygon works with X2Y2"):
                    assert extended_area_polygon.max(axis=0) == con.Approx(
                        extended_x2y2, atol=1e-3
                    )
                polygon_node = scene_coords_node.find("./{*}ImageArea/{*}Polygon")
                with self.precondition():
                    assert polygon_node is not None
                    polygon = self.get_polygon(polygon_node)
                    shg_extended = shg.Polygon(extended_area_polygon)
                    shg_polygon = shg.Polygon(polygon)
                    with self.need("Extended area polygon covers image area polygon"):
                        assert shg.Polygon(shg_extended).covers(shg_polygon)

    @per_channel
    def check_channel_imagearea_x1y1(self, channel_id, channel_node):
        """Channel/Parameters/ImageArea X1Y1 and X2Y2 consistent with SceneCoordinates/ImageArea X1Y1 and X2Y2."""
        with self.precondition():
            assert channel_node.find("./{*}ImageArea") is not None
            x1y1 = self.xmlhelp.load_elem(channel_node.find("./{*}ImageArea/{*}X1Y1"))
            x2y2 = self.xmlhelp.load_elem(channel_node.find("./{*}ImageArea/{*}X2Y2"))
            with self.need(
                "Channel/Parameters/ImageArea/X1Y1 < Channel/Parameters/ImageArea/X2Y2"
            ):
                assert x1y1[0] < x2y2[0]
                assert x1y1[1] < x2y2[1]
            ia_x1y1 = self.xmlhelp.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1")
            ia_x2y2 = self.xmlhelp.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2")
            with self.need(
                "Channel/Parameters/ImageArea/X1Y1 bounded by SceneCoordinates/ImageArea/X1Y1"
            ):
                assert x1y1 >= con.Approx(ia_x1y1)
            with self.need(
                "Channel/Parameters/ImageArea/X2Y2 bounded by SceneCoordinates/ImageArea/X2Y2"
            ):
                assert x2y2 <= con.Approx(ia_x2y2)

    def check_imagearea_x1y1_x2y2(self):
        """SceneCoordinates/ImageArea is self-consistent."""

        x1, y1 = self.xmlhelp.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1")
        x2, y2 = self.xmlhelp.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2")
        with self.need(
            "SceneCoordinates/ImageArea/X1Y1 < SceneCoordinates/ImageArea/X2Y2"
        ):
            assert x1 < x2
            assert y1 < y2

    def check_extended_imagearea_x1y1_x2y2(self):
        """Extended image area contains the image area."""
        with self.precondition():
            assert (
                self.cphdroot.find("./{*}SceneCoordinates/{*}ExtendedArea") is not None
            )
            extended_x1y1 = self.xmlhelp.load(
                "./{*}SceneCoordinates/{*}ExtendedArea/{*}X1Y1"
            )
            extended_x2y2 = self.xmlhelp.load(
                "./{*}SceneCoordinates/{*}ExtendedArea/{*}X2Y2"
            )
            with self.need(
                "SceneCoordinates/ExtendedArea/X1Y1 < SceneCoordinates/ExtendedArea/X2Y2"
            ):
                assert extended_x1y1[0] < extended_x2y2[0]
                assert extended_x1y1[1] < extended_x2y2[1]
            global_x1y1 = self.xmlhelp.load(
                "./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1"
            )
            global_x2y2 = self.xmlhelp.load(
                "./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2"
            )
            with self.need("Extended X1Y1 less than image area X1Y1"):
                assert extended_x1y1 <= con.Approx(global_x1y1)
            with self.need("Extended X2Y2 greater than image area X2Y2"):
                assert extended_x2y2 >= con.Approx(global_x2y2)

    @per_channel
    def check_channel_signal_data(self, channel_id, channel_node):
        """Sample data is all finite."""
        with self.precondition():
            assert self.kvp_list is not None
            assert self.cphdroot.find("./{*}Data/{*}SignalCompressionID") is None
            format_string = self.xmlhelp.load("./{*}Data/{*}SignalArrayFormat")
            signal_dtype = skcphd.binary_format_string_to_dtype(format_string)

            channel_data_node = get_by_id(
                self.cphdroot, "./{*}Data/{*}Channel", channel_id
            )
            signal_offset = self.xmlhelp.load_elem(
                channel_data_node.find("./{*}SignalArrayByteOffset")
            )
            num_vectors = self.xmlhelp.load_elem(
                channel_data_node.find("./{*}NumVectors")
            )
            num_samples = self.xmlhelp.load_elem(
                channel_data_node.find("./{*}NumSamples")
            )
            signal_end = (
                signal_offset + num_vectors * num_samples * signal_dtype.itemsize
            )
            signal_file_offset = (
                int(self.kvp_list["SIGNAL_BLOCK_BYTE_OFFSET"]) + signal_offset
            )

            with self.need("Channel signal fits in signal block"):
                assert int(self.kvp_list["SIGNAL_BLOCK_SIZE"]) >= signal_end
            with self.precondition():
                assert self.file is not None
                assert format_string == "CF8"
                self.file.seek(signal_file_offset, os.SEEK_SET)
                samples_remaining = num_vectors * num_samples
                max_read_bytes = 2**20
                max_read_samples = max_read_bytes // signal_dtype.itemsize
                while samples_remaining:
                    data = self.file.read(
                        signal_dtype.itemsize * min(max_read_samples, samples_remaining)
                    )
                    with self.need("Channel signal fits within file"):
                        assert data
                    signal = np.frombuffer(data, signal_dtype.newbyteorder(">"))
                    with self.need("All signal samples are finite and not NaN"):
                        assert np.all(np.isfinite(signal))
                    samples_remaining -= len(signal)

    @per_channel
    def check_channel_normal_signal_pvp(self, channel_id, channel_node):
        """SIGNAL PVP = 1 for at least half of the vectors."""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            assert "SIGNAL" in pvp.dtype.fields
            num_normal = np.count_nonzero(pvp["SIGNAL"] == 1)
            with self.want("SIGNAL PVP = 1 for at least half of the vectors"):
                assert num_normal / pvp.size >= 0.5

    def check_image_grid_exists(self):
        """Verify that the ImageGrid is defined"""
        with self.want(
            "It is recommended to populate SceneCoordinates.ImageGrid for processing purposes"
        ):
            assert self.cphdroot.find("./{*}SceneCoordinates/{*}ImageGrid") is not None

    def check_pad_header_xml(self):
        """The pad between the header and XML is 0."""
        with self.precondition():
            assert self.kvp_list is not None
            with self.want("XML appears early in the file"):
                assert int(self.kvp_list["XML_BLOCK_BYTE_OFFSET"]) < 2**28
            assert self.file is not None
            self.file.seek(0, os.SEEK_SET)
            before_xml = self.file.read(int(self.kvp_list["XML_BLOCK_BYTE_OFFSET"]))
            first_form_feed = before_xml.find("\f\n".encode("utf-8"))
            with self.need("header section terminator exists before XML"):
                assert b"\f\n" in before_xml
            with self.want("Pad is 0"):
                assert np.all(
                    np.frombuffer(before_xml[first_form_feed + 2 :], dtype=np.uint8)
                    == 0
                )

    def check_pad_after_xml(self):
        """The pad after XML is 0."""
        with self.precondition():
            assert self.kvp_list is not None
            xml_end = int(self.kvp_list["XML_BLOCK_BYTE_OFFSET"]) + int(
                self.kvp_list["XML_BLOCK_SIZE"]
            )
            if "SUPPORT_BLOCK_BYTE_OFFSET" in self.kvp_list:
                num_bytes_after_xml = (
                    int(self.kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"]) - xml_end
                )
                next_block = "Support"
            else:
                num_bytes_after_xml = (
                    int(self.kvp_list["PVP_BLOCK_BYTE_OFFSET"]) - xml_end
                )
                next_block = "PVP"
            with self.need(f"{next_block} comes after XML"):
                assert num_bytes_after_xml - 2 >= 0

            assert self.file is not None
            self.file.seek(xml_end, os.SEEK_SET)
            bytes_after_xml = self.file.read(num_bytes_after_xml)
            terminator = bytes_after_xml[:2]
            pad = np.array(
                struct.unpack(f"{num_bytes_after_xml - 2}B", bytes_after_xml[2:])
            )
            with self.need("Section terminator exists"):
                assert terminator == b"\f\n"
            with self.want("Pad is 0"):
                assert np.all(pad == 0)

    def check_pad_after_support(self):
        """The pad after support arrays is 0."""
        with self.precondition():
            assert self.kvp_list is not None
            assert "SUPPORT_BLOCK_BYTE_OFFSET" in self.kvp_list
            support_end = int(self.kvp_list["SUPPORT_BLOCK_BYTE_OFFSET"]) + int(
                self.kvp_list["SUPPORT_BLOCK_SIZE"]
            )
            num_bytes_after_support = (
                int(self.kvp_list["PVP_BLOCK_BYTE_OFFSET"]) - support_end
            )
            with self.need("PVP comes after Support"):
                assert num_bytes_after_support >= 0

            assert self.file is not None
            self.file.seek(support_end, os.SEEK_SET)
            bytes_after_support = self.file.read(num_bytes_after_support)
            bytes_after_support = np.array(
                struct.unpack(f"{num_bytes_after_support}B", bytes_after_support)
            )
            with self.want("Pad is 0"):
                assert np.all(bytes_after_support == 0)

    def check_pad_after_pvp(self):
        """The pad after PVPs is 0."""
        with self.precondition():
            assert self.kvp_list is not None
            pvp_end = int(self.kvp_list["PVP_BLOCK_BYTE_OFFSET"]) + int(
                self.kvp_list["PVP_BLOCK_SIZE"]
            )
            num_bytes_after_pvp = (
                int(self.kvp_list["SIGNAL_BLOCK_BYTE_OFFSET"]) - pvp_end
            )
            with self.need("Signal comes after PVP"):
                assert num_bytes_after_pvp >= 0

            assert self.file is not None
            self.file.seek(pvp_end, os.SEEK_SET)
            bytes_after_pvp = self.file.read(num_bytes_after_pvp)
            bytes_after_pvp = np.array(
                struct.unpack(f"{num_bytes_after_pvp}B", bytes_after_pvp)
            )
            with self.want("Pad is 0"):
                assert np.all(bytes_after_pvp == 0)

    def check_signal_block_size_and_packing(self):
        """Signal block is correctly sized and packed"""
        has_compression_id = (
            self.cphdroot.find("{*}Data/{*}SignalCompressionID") is not None
        )
        signal_array_offset_size = {}

        signal_dtype_str = self.cphdroot.findtext("{*}Data/{*}SignalArrayFormat")
        signal_dtype = skcphd.binary_format_string_to_dtype(signal_dtype_str)
        num_bytes_samp = signal_dtype.itemsize
        for channel_node in self.cphdroot.findall("{*}Data/{*}Channel"):
            array_id = channel_node.findtext("{*}Identifier")
            compressed_signal_size_elem = channel_node.find("{*}CompressedSignalSize")
            if has_compression_id:
                with self.need(
                    f"CompressedSignalSize in Data/Channel for SIGNAL array {array_id}"
                ):
                    assert compressed_signal_size_elem is not None
                array_size = int(compressed_signal_size_elem.text)
            else:
                with self.need(
                    f"CompressedSignalSize not in Data/Channel for SIGNAL array {array_id}"
                ):
                    assert compressed_signal_size_elem is None
                array_size = (
                    int(channel_node.findtext("{*}NumVectors"))
                    * int(channel_node.findtext("{*}NumSamples"))
                    * num_bytes_samp
                )

            signal_array_offset_size[array_id] = (
                int(channel_node.findtext("{*}SignalArrayByteOffset")),
                array_size,
            )
        prev_end = 0
        sorted_arrays = sorted(signal_array_offset_size.items(), key=lambda x: x[1])
        for array_id, (offset, size) in sorted_arrays:
            with self.need(f"SIGNAL array {array_id} starts at offset {prev_end}"):
                assert offset == prev_end
            prev_end = offset + size
        with self.precondition():
            assert self.kvp_list is not None
            with self.need(
                f"SIGNAL_BLOCK_SIZE matches the end of the last SIGNAL array {sorted_arrays[-1][0]}"
            ):
                assert prev_end == int(self.kvp_list["SIGNAL_BLOCK_SIZE"])

    def check_signal_at_end_of_file(self):
        """Signal is at the end of the file."""
        with self.precondition():
            assert self.kvp_list is not None
            assert self.file is not None
            with self.need("Signal is at the end of the file"):
                self.file.seek(0, os.SEEK_END)
                file_size = self.file.tell()
                assert file_size == int(
                    self.kvp_list["SIGNAL_BLOCK_BYTE_OFFSET"]
                ) + int(self.kvp_list["SIGNAL_BLOCK_SIZE"])

    def check_scene_plane_axis_vectors(self):
        """Scene plane axis vectors are orthonormal."""
        planar_node = self.cphdroot.find(
            "./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar"
        )
        with self.precondition():
            assert planar_node is not None
            uiax = self.xmlhelp.load_elem(planar_node.find("./{*}uIAX"))
            uiay = self.xmlhelp.load_elem(planar_node.find("./{*}uIAY"))
            with self.need("uIAX is unit"):
                assert np.linalg.norm(uiax) == con.Approx(1)
            with self.need("uIAY is unit"):
                assert np.linalg.norm(uiay) == con.Approx(1)
            with self.need("uIAX and uIAY are orthogonal (dot is zero)"):
                assert np.dot(uiax, uiay) == con.Approx(0, atol=1e-6)

    def check_global_txtime_limits(self):
        """The Global TxTime1 and TxTime2 match the PVPs."""
        with self.precondition():
            assert self.pvps is not None
            txtime1_chan = min(np.nanmin(x["TxTime"]) for x in self.pvps.values())
            txtime2_chan = max(np.nanmax(x["TxTime"]) for x in self.pvps.values())
            with self.need("Timeline TxTime1 matches PVP"):
                assert txtime1_chan == con.Approx(
                    self.xmlhelp.load("./{*}Global/{*}Timeline/{*}TxTime1")
                )
            with self.need("Timeline TxTime2 matches PVP"):
                assert txtime2_chan == con.Approx(
                    self.xmlhelp.load("./{*}Global/{*}Timeline/{*}TxTime2")
                )

    def check_global_fx_band(self):
        """The Global FXBand matches the PVPs."""
        with self.precondition():
            assert self.pvps is not None
            fx1min_chan = min(np.nanmin(x["FX1"]) for x in self.pvps.values())
            fx2max_chan = max(np.nanmax(x["FX2"]) for x in self.pvps.values())
            with self.need("FxMin matches PVP"):
                assert fx1min_chan == con.Approx(
                    self.xmlhelp.load("./{*}Global/{*}FxBand/{*}FxMin")
                )
            with self.need("FxMax match PVP"):
                assert fx2max_chan == con.Approx(
                    self.xmlhelp.load("./{*}Global/{*}FxBand/{*}FxMax")
                )

    def check_global_toaswath(self):
        """The Global TOASwath matches the PVPs."""
        with self.precondition():
            assert self.pvps is not None
            toa1min_chan = min(np.nanmin(x["TOA1"]) for x in self.pvps.values())
            toa2max_chan = max(np.nanmax(x["TOA2"]) for x in self.pvps.values())
            with self.need("TOAMin matches PVP"):
                assert toa1min_chan == con.Approx(
                    self.xmlhelp.load("./{*}Global/{*}TOASwath/{*}TOAMin")
                )
            with self.need("TOAMax matches PVP"):
                assert toa2max_chan == con.Approx(
                    self.xmlhelp.load("./{*}Global/{*}TOASwath/{*}TOAMax")
                )

    def _check_ids_in_channel_for_optional_branch(self, branch_name):
        with self.precondition():
            assert self.cphdroot.find(f"./{{*}}{branch_name}") is not None
            with self.want(f"{branch_name} present in /Channel/Parameters"):
                assert (
                    self.cphdroot.find(
                        f"./{{*}}Channel/{{*}}Parameters/{{*}}{branch_name}"
                    )
                    is not None
                )

    def check_antenna_ids_in_channel(self):
        """If the Antenna branch exists, then Antenna is also present in /Channel/Parameters"""
        self._check_ids_in_channel_for_optional_branch("Antenna")

    def check_txrcv_ids_in_channel(self):
        """If the TxRcv branch exists, then TxRcv is also present in /Channel/Parameters"""
        self._check_ids_in_channel_for_optional_branch("TxRcv")

    def check_refgeom(self):
        """The ReferenceGeometry parameters are consistent with the other metadata"""
        with self.precondition():
            assert self.pvps is not None
            newroot = skcphd.ElementWrapper(copy.deepcopy(self.cphdroot))
            newroot["ReferenceGeometry"] = skcphd.compute_reference_geometry(
                newroot.elem.getroottree(), self.pvps[newroot["Channel"]["RefChId"]]
            )

            def _compare_children(actual_parent, expected_parent, parent_key):
                with self.need(f"{parent_key} contains only expected elements"):
                    actual_names = list(actual_parent)
                    expected_names = list(expected_parent)
                    assert actual_names == expected_names
                    for key in actual_names:
                        actual_val = actual_parent[key]
                        expected_val = expected_parent[key]
                        if isinstance(expected_val, collections.abc.Mapping):
                            _compare_children(actual_val, expected_val, key)
                            continue

                        approx_args = {}
                        if "Angle" in key:
                            approx_args["atol"] = 1
                        elif key.endswith("Time"):
                            approx_args["atol"] = 1e-6
                        elif key.endswith("Pos"):
                            approx_args["atol"] = 1e-2
                        elif key.endswith("Vel"):
                            approx_args["atol"] = 1e-3

                        if issubclass(
                            np.asarray(expected_val).dtype.type, numbers.Number
                        ):
                            actual_val = con.Approx(actual_val, **approx_args)
                        with self.need(f"{key} matches defined PVP/calculation"):
                            assert np.all(expected_val == actual_val)

            _compare_children(
                skcphd.ElementWrapper(self.cphdroot)["ReferenceGeometry"],
                newroot["ReferenceGeometry"],
                "ReferenceGeometry",
            )

    def check_identifier_uniqueness(self):
        """Identifier nodes are unique."""
        identifier_sets = (
            {"./{*}Antenna/{*}AntCoordFrame/{*}Identifier"},
            {"./{*}Antenna/{*}AntPattern/{*}Identifier"},
            {"./{*}Antenna/{*}AntPhaseCenter/{*}Identifier"},
            {"./{*}Channel/{*}Parameters/{*}Identifier"},
            {"./{*}Data/{*}Channel/{*}Identifier"},
            {"./{*}Data/{*}SupportArray/{*}Identifier"},
            {"./{*}Dwell/{*}CODTime/{*}Identifier"},
            {"./{*}Dwell/{*}DwellTime/{*}Identifier"},
            {
                "./{*}SceneCoordinates/{*}ImageGrid/{*}SegmentList/{*}Segment/{*}Identifier"
            },
            {"./{*}TxRcv/{*}RcvParameters/{*}Identifier"},
            {"./{*}TxRcv/{*}TxWFParameters/{*}Identifier"},
            {
                f"./{{*}}SupportArray/{{*}}{sa_type}/{{*}}Identifier"
                for sa_type in (
                    "IAZArray",
                    "AntGainPhase",
                    "DwellTimeArray",
                    "AddedSupportArray",
                )
            },
        )
        for identifier_set in identifier_sets:
            these_identifiers = []
            for path in identifier_set:
                these_identifiers.extend(x.text for x in self.cphdroot.findall(path))
            repeated_identifiers = _get_repeated_elements(these_identifiers)
            with self.need(f"Identifiers {identifier_set} are unique"):
                assert not repeated_identifiers

    def check_polynomials(self):
        """Polynomial types are correctly specified."""

        def check_poly(poly_elem):
            path = poly_elem.getroottree().getpath(poly_elem)
            order_by_dim = {
                dim: int(poly_elem.get(f"order{dim}"))
                for dim in (1, 2)
                if poly_elem.get(f"order{dim}") is not None
            }
            coef_exponents = [
                tuple(int(coef.get(f"exponent{dim}")) for dim in order_by_dim)
                for coef in poly_elem.findall("./{*}Coef")
            ]
            repeated_coef_exponents = _get_repeated_elements(coef_exponents)
            with self.need(f"{path} is correctly specified"):
                for index, order in enumerate(order_by_dim.values()):
                    dim_coefs_above_order = [
                        coef_exp[index]
                        for coef_exp in coef_exponents
                        if coef_exp[index] > order
                    ]
                    assert not dim_coefs_above_order
                assert not repeated_coef_exponents

        poly_paths = itertools.chain(
            [
                f"./{{*}}Antenna/{{*}}AntPattern/{{*}}{j}/{{*}}{k}Poly"
                for j, k in itertools.product(("Array", "Element"), ("Gain", "Phase"))
            ],
            [
                f"./{{*}}Antenna/{{*}}AntCoordFrame/{{*}}{axis}AxisPoly/{{*}}{comp}"
                for axis, comp in itertools.product("XY", "XYZ")
            ],
            ["./{*}Antenna/{*}AntPattern/{*}GainBSPoly"],
            [f"./{{*}}Antenna/{{*}}AntPattern/{{*}}EB/{{*}}DC{ax}Poly" for ax in "XY"],
            [f"./{{*}}Dwell/{{*}}{x}Time/{{*}}{x}TimePoly" for x in ("COD", "Dwell")],
        )
        for element_path in poly_paths:
            for poly in self.cphdroot.findall(element_path):
                check_poly(poly)

    def check_optional_pvps_fx(self):
        """FXN1 & FXN2 PVPs are included appropriately."""
        is_fx_domain = self.xmlhelp.load("./{*}Global/{*}DomainType") == "FX"
        has_fxn1 = self.xmlhelp.load("./{*}PVP/{*}FXN1") is not None
        has_fxn2 = self.xmlhelp.load("./{*}PVP/{*}FXN2") is not None
        with self.need(
            "FXN1/FXN2 only allowed when /Global/DomainType = FX and must be included together"
        ):
            assert not (has_fxn1 or has_fxn2) or (
                is_fx_domain and has_fxn1 and has_fxn2
            )

    def check_optional_pvps_toa(self):
        """TOAE1 & TOAE2 PVPs are included appropriately."""
        has_toae1 = self.xmlhelp.load("./{*}PVP/{*}TOAE1") is not None
        has_toae2 = self.xmlhelp.load("./{*}PVP/{*}TOAE2") is not None
        with self.need("TOAE1/TOAE2 must be included together"):
            assert has_toae1 == has_toae2


def _get_repeated_elements(items):
    return [x for x, count in collections.Counter(items).items() if count > 1]


def unit(vec, axis=-1):
    return vec / np.linalg.norm(vec, axis=axis, keepdims=True)
