"""
Functionality for verifying CRSD files for internal consistency.
"""

import collections.abc
import copy
import functools
import logging
import numbers
import os
import re
from typing import Any, Optional

import numpy as np
from lxml import etree

import sarkit.crsd as skcrsd
import sarkit.verification._consistency as con
import sarkit.wgs84

logger = logging.getLogger(__name__)

try:
    import shapely.geometry as shg
except ImportError as ie:
    logger.warning("'shapely' package not found. Some features may not work correctly.")
    shg = con._ExceptionOnUse(ie)

INVALID_CHAR_REGEX = re.compile(r"\W")


BINARY_BLOCK_ORDER = ["SUPPORT", "PPP", "PVP", "SIGNAL"]


def dot(x, y):
    return np.sum(x * y, axis=-1)


def per_channel(method):
    """Decorator to mark check methods as being applicable to each CRSD channel

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


def per_sequence(method):
    """Decorator to mark check methods as being applicable to each CRSD TxSequence

    Parameters
    ----------
    method : Callable
        Method to mark

    Returns
    -------
    Callable
        Marked input `method`
    """

    method.per_sequence = True
    return method


class CrsdConsistency(con.ConsistencyChecker):
    """Check CRSD file structure and metadata for internal consistency

    `CrsdConsistency` objects should be instantiated using `from_file` or `from_parts`.

    Parameters
    ----------
    crsd_xml : lxml.etree.Element or lxml.etree.ElementTree
        CRSD XML
    file_type_header : str, optional
        File type header from the first line of the file
    kvp_list : dict of {str : str}, optional
        Key-Value pair list of header fields
    ppps : dict of {str : ndarray}, optional
        Per-Pulse-Parameters keyed by transmit sequence identifier
    pvps : dict of {str : ndarray}, optional
        Per-Vector-Parameters keyed by channel identifier
    support_arrays : dict of {str : ndarray}, optional
        Support arrays keyed by Support Array ID
    schema_override : `path-like object`, optional
        Path to XML Schema. If None, tries to find a version-specific schema
    file : `file object`, optional
        CRSD file; when specified, portions of the file not specified in other parameters may be read
    """

    def __init__(
        self,
        crsd_xml,
        *,
        file_type_header=None,
        kvp_list=None,
        ppps=None,
        pvps=None,
        support_arrays=None,
        schema_override=None,
        file=None,
    ):
        super().__init__()
        # handle element or tree -> element
        try:
            self.crsdroot = crsd_xml.getroot()
        except AttributeError:
            self.crsdroot = crsd_xml.getroottree().getroot()
        self.xmlhelp = skcrsd.XmlHelper(self.crsdroot.getroottree())

        self.file_type_header = file_type_header
        self.kvp_list = kvp_list
        self.ppps = ppps
        self.pvps = pvps
        self.support_arrays = support_arrays
        self.crsd_type = etree.QName(self.crsdroot).localname
        ns = etree.QName(self.crsdroot).namespace
        self.schema = schema_override or skcrsd.VERSION_INFO.get(ns, {}).get("schema")

        self.file = file

        sequence_ids = [
            x.text
            for x in self.crsdroot.findall("{*}Data/{*}Transmit/{*}TxSequence/{*}TxId")
        ]
        channel_ids = [
            x.text
            for x in self.crsdroot.findall("{*}Data/{*}Receive/{*}Channel/{*}ChId")
        ]
        # process decorated methods to generate per-sequence and per-channel tests
        # reverse the enumerated list so that we don't disturb indices on later iterations as we insert into the list
        for index, func in reversed(list(enumerate(self.funcs))):
            if getattr(func, "per_channel", False):
                subfuncs = []
                for channel_id in channel_ids:
                    channel_node = self.crsdroot.find(
                        f"{{*}}Channel/{{*}}Parameters[{{*}}Identifier='{channel_id}']"
                    )
                    subfunc = functools.partial(func, channel_id, channel_node)
                    this_doc = func.__doc__.strip().removesuffix(".")
                    subfunc.__doc__ = f"{this_doc} for channel '{channel_id}'."
                    modified_channel_id = re.sub(INVALID_CHAR_REGEX, "_", channel_id)
                    subfunc.__name__ = f"{func.__name__}_{modified_channel_id}"
                    subfuncs.append(subfunc)
                self.funcs[index : index + 1] = subfuncs
            elif getattr(func, "per_sequence", False):
                subfuncs = []
                for sequence_id in sequence_ids:
                    sequence_node = self.crsdroot.find(
                        f"{{*}}TxSequence/{{*}}Parameters[{{*}}Identifier='{sequence_id}']"
                    )
                    subfunc = functools.partial(func, sequence_id, sequence_node)
                    this_doc = func.__doc__.strip().removesuffix(".")
                    subfunc.__doc__ = f"{this_doc} for TxSequence '{sequence_id}'."
                    modified_sequence_id = re.sub(INVALID_CHAR_REGEX, "_", sequence_id)
                    subfunc.__name__ = f"{func.__name__}_{modified_sequence_id}"
                    subfuncs.append(subfunc)
                self.funcs[index : index + 1] = subfuncs

    @staticmethod
    def from_file(
        file,
        schema: Optional[str] = None,
        thorough: bool = False,
    ) -> "CrsdConsistency":
        """Create a CrsdConsistency object from a file

        Parameters
        ----------
        file : `file object`
            CRSD or CRSD XML file to check
        schema : str, optional
            Path to XML Schema. If None, tries to find a version-specific schema
        thorough : bool, optional
            Run checks that may seek/read through large portions of the file.
            file must stay open to run checks. Ignored if file is CRSD XML.

        Returns
        -------
        CrsdConsistency
            The initialized consistency checker object

        See Also
        --------
        from_parts

        Examples
        --------
        Use `from_file` to check an XML file:

        .. doctest::

            >>> import sarkit.verification as skver

            >>> with open("data/example-crsd-1.0.xml", "r") as f:
            ...     con = skver.CrsdConsistency.from_file(f)
            >>> con.check()
            >>> bool(con.passes())
            True
            >>> bool(con.failures())
            False

        Use `from_file` to check a CRSD file, with and without thorough checks:

        .. testsetup::

            import sarkit.crsd as skcrsd
            import lxml.etree
            meta = skcrsd.Metadata(
                xmltree=lxml.etree.parse("data/example-crsd-1.0.xml"),
            )
            file = tmppath / "example.crsd"
            with file.open("wb") as f, skcrsd.Writer(f, meta) as w:
                f.seek(
                    w._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"]
                    + w._file_header_kvp["SIGNAL_BLOCK_SIZE"]
                    - 1
                )
                f.write(b"0")

        .. doctest::

            >>> with file.open("rb") as f:
            ...     con_thorough = skver.CrsdConsistency.from_file(f, thorough=True)
            ...     con = skver.CrsdConsistency.from_file(f)
            ...     con_thorough.check()  # thorough checks require open file
            >>> con.check()  # without thorough, open file only used for construction
            >>> print(len(con.skips()) > len(con_thorough.skips()))
            True
        """
        kwargs: dict[str, Any] = {"schema_override": schema}
        try:
            crsd_xmltree = etree.parse(file)
            kvp_list = None
            pvps = None
            ppps = None
            support_arrays = None
        except etree.XMLSyntaxError:
            file.seek(0, os.SEEK_SET)
            reader = skcrsd.Reader(file)
            crsd_xmltree = reader.metadata.xmltree
            file.seek(0, os.SEEK_SET)
            file_type_header, kvp_list = skcrsd.read_file_header(file)
            pvps = {}
            for channel_node in crsd_xmltree.findall("./{*}Data/{*}Receive/{*}Channel"):
                channel_id = channel_node.findtext("./{*}ChId")
                pvps[channel_id] = reader.read_pvps(channel_id)
            ppps = {}
            for sequence_node in crsd_xmltree.findall(
                "./{*}Data/{*}Transmit/{*}TxSequence"
            ):
                sequence_id = sequence_node.findtext("./{*}TxId")
                ppps[sequence_id] = reader.read_ppps(sequence_id)

            support_arrays = {
                sa_id.text: reader.read_support_array(sa_id.text)
                for sa_id in crsd_xmltree.findall(
                    "{*}SupportArray/{*}GainPhaseArray/{*}Identifier"
                )
                + crsd_xmltree.findall(
                    "{*}SupportArray/{*}DwellTimeArray/{*}Identifier"
                )
            }
            kwargs.update(
                {
                    "file_type_header": file_type_header,
                    "kvp_list": kvp_list,
                    "ppps": ppps,
                    "pvps": pvps,
                    "support_arrays": support_arrays,
                }
            )
            if thorough:
                kwargs["file"] = file

        return CrsdConsistency(
            crsd_xmltree,
            **kwargs,
        )

    @staticmethod
    def from_parts(
        crsd_xml: "etree.Element | etree.ElementTree",
        file_type_header: Optional[str] = None,
        kvp_list: Optional[dict[str, str]] = None,
        ppps: Optional[dict[str, np.ndarray]] = None,
        pvps: Optional[dict[str, np.ndarray]] = None,
        support_arrays: Optional[dict[str, np.ndarray]] = None,
        schema: Optional[str] = None,
    ) -> "CrsdConsistency":
        """Create a CrsdConsistency object from assorted parts

        Parameters
        ----------
        crsd_xml : lxml.etree.Element or lxml.etree.ElementTree
            CRSD XML
        file_type_header : str, optional
            File type header from the first line of the file
        kvp_list : dict of {str : str}, optional
            Key-Value pair list of header fields
        ppps : dict of {str : ndarray}, optional
            Per-Pulse-Parameters keyed by transmit sequence identifier
        pvps : dict of {str : ndarray], optional
            Per-Vector-Parameters keyed by channel identifier
        support_arrays : dict of {str : ndarray}, optional
            Support arrays keyed by Support Array ID
        schema : str, optional
            Path to XML Schema. If None, tries to find a version-specific schema

        Returns
        -------
        CrsdConsistency
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
            >>> crsd_xmltree = lxml.etree.parse("data/example-crsd-1.0.xml")
            >>> con = skver.CrsdConsistency.from_parts(crsd_xmltree)
            >>> con.check()
            >>> bool(con.passes())
            True
            >>> bool(con.failures())
            False

        Use `from_parts` to check a parsed XML element tree and an invalid file type header:

        .. doctest::

            >>> con = skver.CrsdConsistency.from_parts(crsd_xmltree, file_type_header="CRSDsar/INVALID\\n")
            >>> con.check()
            >>> bool(con.failures())
            True
        """
        return CrsdConsistency(
            crsd_xml=crsd_xml,
            file_type_header=file_type_header,
            kvp_list=kvp_list,
            ppps=ppps,
            pvps=pvps,
            support_arrays=support_arrays,
            schema_override=schema,
        )

    def _get_support_array(self, sa_id):
        """
        Returns the support array keyed by `sa_id` or raises an AssertionError.
        """
        assert self.support_arrays is not None
        assert sa_id in self.support_arrays
        return self.support_arrays[sa_id]

    def _get_channel_pvps(self, channel_id):
        """
        Returns the PVPs associated with the channel keyed by `channel_id` or raises an AssertionError.
        """
        assert self.pvps is not None
        assert channel_id in self.pvps
        return self.pvps[channel_id]

    def _get_sequence_ppps(self, sequence_id):
        """
        Returns the PPPs associated with the TxSequence keyed by `sequence_id` or raises an AssertionError.
        """
        assert self.ppps is not None
        assert sequence_id in self.ppps
        return self.ppps[sequence_id]

    def _check_num(self, num_match, elem_match):
        """Need the number of elements found by elem_match to equal the value of the element found by num_match"""
        num_elem = self.crsdroot.find(num_match)
        num = int(num_elem.text)
        num_of_elem = len(self.crsdroot.findall(elem_match))
        with self.need(f"{etree.QName(num_elem).localname} is correct"):
            assert num == num_of_elem

    def check_file_type_header(self):
        """File type header is consistent with the XML."""
        with self.precondition():
            assert self.file_type_header is not None
            version = skcrsd.VERSION_INFO.get(
                etree.QName(self.crsdroot).namespace, {}
            ).get("version")
            assert version is not None
            with self.need("File type header is consistent with the XML"):
                assert self.file_type_header == f"{self.crsd_type}/{version}\n"

    def check_header_kvp_list(self):
        """Asserts that the required keys are in the header KVP list."""
        with self.precondition():
            assert self.kvp_list is not None
            required_fields = set(
                [
                    "XML_BLOCK_SIZE",
                    "XML_BLOCK_BYTE_OFFSET",
                    "SUPPORT_BLOCK_SIZE",
                    "SUPPORT_BLOCK_BYTE_OFFSET",
                    "CLASSIFICATION",
                    "RELEASE_INFO",
                ]
            )
            forbidden_fields = set()
            if self.crsd_type == "CRSDtx":
                required_fields.update(["PPP_BLOCK_SIZE", "PPP_BLOCK_BYTE_OFFSET"])
                forbidden_fields.update(["PVP_BLOCK_SIZE", "PVP_BLOCK_BYTE_OFFSET"])
                forbidden_fields.update(
                    ["SIGNAL_BLOCK_SIZE", "SIGNAL_BLOCK_BYTE_OFFSET"]
                )
            elif self.crsd_type == "CRSDrcv":
                required_fields.update(["PVP_BLOCK_SIZE", "PVP_BLOCK_BYTE_OFFSET"])
                required_fields.update(
                    ["SIGNAL_BLOCK_SIZE", "SIGNAL_BLOCK_BYTE_OFFSET"]
                )
                forbidden_fields.update(["PPP_BLOCK_SIZE", "PPP_BLOCK_BYTE_OFFSET"])
            elif self.crsd_type == "CRSDsar":
                required_fields.update(["PPP_BLOCK_SIZE", "PPP_BLOCK_BYTE_OFFSET"])
                required_fields.update(["PVP_BLOCK_SIZE", "PVP_BLOCK_BYTE_OFFSET"])
                required_fields.update(
                    ["SIGNAL_BLOCK_SIZE", "SIGNAL_BLOCK_BYTE_OFFSET"]
                )
            else:
                raise ValueError(f"Invalid crsd_type {self.crsd_type}")
            for name in required_fields:
                with self.need(f"Required KVP field: {name} is in KVP list"):
                    assert name in self.kvp_list
            for name in forbidden_fields:
                with self.need(f"Forbidden KVP field: {name} is not in KVP list"):
                    assert name not in self.kvp_list

            for block_type in ("PVP", "PPP", "SIGNAL"):
                has_size = f"{block_type}_BLOCK_SIZE" in self.kvp_list
                has_offset = f"{block_type}_BLOCK_BYTE_OFFSET" in self.kvp_list
                with self.need(f"{block_type}_BLOCK fields go together"):
                    assert has_size == has_offset

    def check_classification_and_release_info(self):
        """Asserts that the Classification and ReleaseInfo fields are the same in header KVP list and the xml."""
        with self.precondition():
            assert self.kvp_list is not None
            with self.need("KVP list CLASSIFICATION matches XML Classification"):
                assert self.kvp_list["CLASSIFICATION"] == self.xmlhelp.load(
                    "./{*}ProductInfo/{*}Classification"
                )
            with self.need("KVP list RELEASE_INFO matches XML ReleaseInfo"):
                assert self.kvp_list["RELEASE_INFO"] == self.xmlhelp.load(
                    "./{*}ProductInfo/{*}ReleaseInfo"
                )

    def check_sensorname(self):
        """Transmit and receive sensor names are the same for monostatic SAR"""
        with self.precondition():
            assert self.crsdroot.findtext("{*}SARInfo/{*}CollectType") == "MONOSTATIC"
            with self.need("Transmit and receive sensor names are the same"):
                assert self.crsdroot.findtext(
                    "{*}TransmitInfo/{*}SensorName"
                ) == self.crsdroot.findtext("{*}ReceiveInfo/{*}SensorName")

    def check_eventname(self):
        """Transmit and receive event names are the same for monostatic SAR"""
        with self.precondition():
            assert self.crsdroot.findtext("{*}SARInfo/{*}CollectType") == "MONOSTATIC"
            with self.need("Transmit and receive event names are the same"):
                assert self.crsdroot.findtext(
                    "{*}TransmitInfo/{*}EventName"
                ) == self.crsdroot.findtext("{*}ReceiveInfo/{*}EventName")

    def check_global_txtime12(self):
        """Global TxTime1 and TxTime2 match TxSequences"""
        with self.precondition():
            assert self.crsdroot.find("{*}TxSequence") is not None
            txtime1s = [
                float(x.text)
                for x in self.crsdroot.findall("{*}TxSequence/{*}Parameters/{*}TxTime1")
            ]
            txtime2s = [
                float(x.text)
                for x in self.crsdroot.findall("{*}TxSequence/{*}Parameters/{*}TxTime2")
            ]
            with self.need("Global TxTime1 matches earliest TxSequence TxTime1"):
                assert float(
                    self.crsdroot.findtext("{*}Global/{*}Transmit/{*}TxTime1")
                ) == min(txtime1s)
            with self.need("Global TxTime2 matches latest TxSequence TxTime2"):
                assert float(
                    self.crsdroot.findtext("{*}Global/{*}Transmit/{*}TxTime2")
                ) == max(txtime2s)

    def check_global_fxminmax(self):
        """Global FxMin and FxMax match TxSequences"""
        with self.precondition():
            assert self.crsdroot.find("{*}TxSequence") is not None
            fxcs = np.array(
                [
                    float(x.text)
                    for x in self.crsdroot.findall("{*}TxSequence/{*}Parameters/{*}FxC")
                ]
            )
            fxbws = np.array(
                [
                    float(x.text)
                    for x in self.crsdroot.findall(
                        "{*}TxSequence/{*}Parameters/{*}FxBW"
                    )
                ]
            )
            with self.need("Global FxMin matches min of TxSequence FxC - FxBW/2"):
                assert float(
                    self.crsdroot.findtext("{*}Global/{*}Transmit/{*}FxMin")
                ) == con.Approx(min(fxcs - fxbws / 2))
            with self.need("Global FxMax matches max of TxSequence FxC + FxBW/2"):
                assert float(
                    self.crsdroot.findtext("{*}Global/{*}Transmit/{*}FxMax")
                ) == con.Approx(max(fxcs + fxbws / 2))

    def check_global_rcvstarttime12(self):
        """Global RcvStartTime1 and RcvStartTime2 match Channels"""
        with self.precondition():
            assert self.crsdroot.find("{*}Channel") is not None
            rcvstart1 = [
                float(x.text)
                for x in self.crsdroot.findall(
                    "{*}Channel/{*}Parameters/{*}RcvStartTime1"
                )
            ]
            rcvstart2 = [
                float(x.text)
                for x in self.crsdroot.findall(
                    "{*}Channel/{*}Parameters/{*}RcvStartTime2"
                )
            ]
            with self.need("Global RcvStartTime1 matches min of Channel RcvStartTime1"):
                assert float(
                    self.crsdroot.findtext("{*}Global/{*}Receive/{*}RcvStartTime1")
                ) == min(rcvstart1)
            with self.need("Global RcvStartTime2 matches max of Channel RcvStartTime2"):
                assert float(
                    self.crsdroot.findtext("{*}Global/{*}Receive/{*}RcvStartTime2")
                ) == max(rcvstart2)

    def check_global_frcvminmax(self):
        """Global FrcvMin and FrcvMax match Channels"""
        with self.precondition():
            assert self.crsdroot.find("{*}Channel") is not None
            frcvmins = [
                float(x.text)
                for x in self.crsdroot.findall("{*}Channel/{*}Parameters/{*}FrcvMin")
            ]
            frcvmaxs = [
                float(x.text)
                for x in self.crsdroot.findall("{*}Channel/{*}Parameters/{*}FrcvMax")
            ]
            with self.need("Global FrcvMin matches min of Channel FrcvMin"):
                assert float(
                    self.crsdroot.findtext("{*}Global/{*}Receive/{*}FrcvMin")
                ) == min(frcvmins)
            with self.need("Global FrcvMax matches max of Channel FrcvMax"):
                assert float(
                    self.crsdroot.findtext("{*}Global/{*}Receive/{*}FrcvMax")
                ) == max(frcvmaxs)

    def check_reftxid(self):
        """RefTxId equal to reference channel's transmit pulse sequence"""
        with self.precondition():
            assert self.crsd_type == "CRSDsar"
            ref_tx_id = self.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
            ref_chan_id = self.crsdroot.findtext("{*}Channel/{*}RefChId")
            ref_chan_tx_id = self.crsdroot.findtext(
                f'{{*}}Channel/{{*}}Parameters[{{*}}Identifier="{ref_chan_id}"]/{{*}}SARImage/{{*}}TxId'
            )
            with self.need("RefTxId is the TxId of the reference channel"):
                assert ref_chan_tx_id == ref_tx_id

    @per_sequence
    def check_xm_id(self, sequence_id, sequence_param_elem):
        """XMId is appropriate"""
        with self.precondition():
            assert self.crsdroot.findtext("{*}TxSequence/{*}TxWFType") == "LFM"
            with self.need("XMId is not present"):
                assert sequence_param_elem.find("{*}XMId") is None
        with self.precondition():
            assert self.crsdroot.findtext("{*}TxSequence/{*}TxWFType") != "LFM"
            with self.need("XMId is present"):
                assert sequence_param_elem.find("{*}XMId") is not None

    @per_sequence
    def check_ref_pulse_index(self, sequence_id, sequence_param_elem):
        """RefPulseIndex refers to an extant pulse"""
        ref_pulse_index = int(sequence_param_elem.findtext("{*}RefPulseIndex"))
        num_pulses = int(
            self.crsdroot.findtext(
                f'{{*}}Data/{{*}}Transmit/{{*}}TxSequence[{{*}}TxId="{sequence_id}"]/{{*}}NumPulses'
            )
        )
        with self.need("RefPulseIndex refers to an extant pulse"):
            # nonnegative is checked by the schema
            assert ref_pulse_index < num_pulses

    @per_sequence
    def check_txrefpoint(self, sequence_id, sequence_param_elem):
        """TxRefPoint IAC maps to ECF"""
        ref_pt_iac = self.xmlhelp.load_elem(
            sequence_param_elem.find("{*}TxRefPoint/{*}IAC")
        )
        ref_pt_ecf = self.xmlhelp.load_elem(
            sequence_param_elem.find("{*}TxRefPoint/{*}ECF")
        )
        with self.need("IAC maps to ECF"):
            self.assert_iac_matches_ecf(ref_pt_iac, ref_pt_ecf)

    @per_sequence
    def check_ref_rad_intensity(self, sequence_id, sequence_param_elem):
        """TxRefRadIntensity matches the PPP"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            ref_pulse_index = int(sequence_param_elem.findtext("{*}RefPulseIndex"))
            tx_rad_int_ppp = ppp[ref_pulse_index]["TxRadInt"]
            tx_rad_int_xml = float(sequence_param_elem.findtext("{*}TxRefRadIntensity"))
            with self.need("TxRefRadIntensity matches the PPP"):
                assert tx_rad_int_xml == con.Approx(tx_rad_int_ppp, atol=1e-5)

    @per_sequence
    def check_txtime12(self, sequence_id, sequence_param_elem):
        """TxTime1 and TxTime2 match the PPPs"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            tx_time_1_xml = float(sequence_param_elem.findtext("{*}TxTime1"))
            tx_time_2_xml = float(sequence_param_elem.findtext("{*}TxTime2"))
            with self.need("TxTime1 matches the first PPP"):
                assert tx_time_1_xml == con.Approx(
                    ppp["TxTime"][0]["Int"] + ppp["TxTime"][0]["Frac"], atol=1e-6
                )
            with self.need("TxTime2 matches the last PPP"):
                assert tx_time_2_xml == con.Approx(
                    ppp["TxTime"][-1]["Int"] + ppp["TxTime"][-1]["Frac"], atol=1e-6
                )

    @per_sequence
    def check_txmt_minmax(self, sequence_id, sequence_param_elem):
        """TXmtMin and TXmtMax match the PPPs"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            txmt_ppp = ppp["TXmt"]
            txmt_min = float(sequence_param_elem.findtext("{*}TXmtMin"))
            txmt_max = float(sequence_param_elem.findtext("{*}TXmtMax"))
            with self.need("TXmtMin matches minimum value in the PPPs"):
                assert txmt_ppp.min() == con.Approx(txmt_min, atol=1e-9)
            with self.need("TXmtMax matches maximum value in the PPPs"):
                assert txmt_ppp.max() == con.Approx(txmt_max, atol=1e-9)

    @per_sequence
    def check_tx_frequency_band(self, sequence_id, sequence_param_elem):
        """FxC, FxBW and FxBWFixed match the PPPs"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            fxc = float(sequence_param_elem.findtext("{*}FxC"))
            fxbw = float(sequence_param_elem.findtext("{*}FxBW"))
            fx1 = ppp["FX1"]
            fx2 = ppp["FX2"]
            with self.need("FxC matches the PPPs"):
                assert con.Approx(fxc) == (fx1.min() + fx2.max()) / 2
            with self.need("FxBW matches the PPPs"):
                assert con.Approx(fxbw) == (fx2.max() - fx1.min())
            with self.precondition():
                assert self.xmlhelp.load_elem(sequence_param_elem.find("{*}FxBWFixed"))
                with self.need("FX1 does not change"):
                    assert np.ptp(fx1) == 0
                with self.need("FX2 does not change"):
                    assert np.ptp(fx2) == 0
            with self.precondition():
                assert not self.xmlhelp.load_elem(
                    sequence_param_elem.find("{*}FxBWFixed")
                )
                with self.need("FX1 or FX2 does change"):
                    assert np.ptp(fx1) != 0 or np.ptp(fx1) != 0

    @per_sequence
    def check_txpolarization(self, sequence_id, sequence_param_elem):
        """Check the TxPolarization with respect to HV"""
        with self.need("AmpH and AmpV are normalized"):
            assert float(
                sequence_param_elem.findtext("{*}TxPolarization/{*}AmpH")
            ) ** 2 + float(
                sequence_param_elem.findtext("{*}TxPolarization/{*}AmpV")
            ) ** 2 == con.Approx(1.0)
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            apat_id = sequence_param_elem.findtext("{*}TxAPATId")
            ref_pulse_index = int(sequence_param_elem.findtext("{*}RefPulseIndex"))
            ref_point = self.xmlhelp.load_elem(
                sequence_param_elem.find("{*}TxRefPoint/{*}ECF")
            )
            ampx = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}AmpX"
                )
            )
            ampy = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}AmpY"
                )
            )
            phasex = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}PhaseX"
                )
            )
            phasey = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}PhaseY"
                )
            )
            acx = ppp["TxACX"][ref_pulse_index]
            acy = ppp["TxACY"][ref_pulse_index]
            apc_pos = ppp["TxPos"][ref_pulse_index]

            amph, ampv, phaseh, phasev = skcrsd.compute_h_v_pol_parameters(
                apc_pos, acx, acy, ref_point, 1, ampx, ampy, phasex, phasey
            )
            with self.need("H component is correct"):
                assert float(
                    sequence_param_elem.findtext("{*}TxPolarization/{*}AmpH")
                ) * np.exp(
                    2j
                    * np.pi
                    * float(sequence_param_elem.findtext("{*}TxPolarization/{*}PhaseH"))
                ) == con.Approx(amph * np.exp(2j * np.pi * phaseh), atol=1e-3)
            with self.need("V component is correct"):
                assert float(
                    sequence_param_elem.findtext("{*}TxPolarization/{*}AmpV")
                ) * np.exp(
                    2j
                    * np.pi
                    * float(sequence_param_elem.findtext("{*}TxPolarization/{*}PhaseV"))
                ) == con.Approx(ampv * np.exp(2j * np.pi * phasev), atol=1e-3)

    def check_xmindex_ppp_presence(self):
        """XMIndex PPP is present if and only if TxWFType is not LFM"""
        with self.precondition():
            assert self.crsdroot.find("{*}PPP/{*}XMIndex") is not None
            with self.need("TxWFType is not LFM"):
                assert self.crsdroot.findtext("{*}TxSequence/{*}TxWFType") != "LFM"
        with self.precondition():
            assert self.crsdroot.find("{*}PPP") is not None
            assert self.crsdroot.find("{*}PPP/{*}XMIndex") is None
            with self.need("TxWFType is LFM"):
                assert self.crsdroot.findtext("{*}TxSequence/{*}TxWFType") == "LFM"

    @per_sequence
    def check_xmindex_value(self, sequence_id, sequence_param_elem):
        """XMIndex PPP refers to an extant row in the XMArray"""
        with self.precondition():
            assert self.crsdroot.find("{*}PPP/{*}XMIndex") is not None
            ppp = self._get_sequence_ppps(sequence_id)
            xm_id = sequence_param_elem.findtext("{*}XMId")
            num_xm = int(
                self.crsdroot.findtext(
                    f'{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId="{xm_id}"]/{{*}}NumRows'
                )
            )
            with self.need("XMIndex >= 0"):
                assert np.all(ppp["XMIndex"] >= 0)
            with self.need("XMIndex < NumXM"):
                assert np.all(ppp["XMIndex"] < num_xm)

    @per_sequence
    def check_fx12_ppp(self, sequence_id, sequence_param_elem):
        """FX1 less than FX2"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            with self.need("FX1 < FX2"):
                assert np.all(ppp["FX1"] < ppp["FX2"])

    @per_sequence
    def check_txposveltime(self, sequence_id, sequence_param_elem):
        """TxPos, TxVel, TxTime seem normal"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            txtime_float = ppp["TxTime"]["Int"] + ppp["TxTime"]["Frac"]
            with self.want("TxPos, TxVel, TxTime are a matched set"):
                assert np.diff(ppp["TxPos"], axis=0) == con.Approx(
                    np.diff(txtime_float)[:, np.newaxis]
                    * (ppp["TxVel"][:-1] + ppp["TxVel"][1:])
                    / 2,
                    rtol=0.2,
                )
            with self.want("TxPos is near Earth"):
                assert np.linalg.norm(ppp["TxPos"], axis=-1) == con.Approx(
                    sarkit.wgs84.SEMI_MAJOR_AXIS, rtol=0.5
                )

    @per_sequence
    def check_txtime(self, sequence_id, sequence_param_elem):
        """TxTime PPP is increasing and properly formatted as int-frac"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            with self.need("TxTime is strictly increasing"):
                assert np.all(
                    np.diff(ppp["TxTime"]["Int"]) + np.diff(ppp["TxTime"]["Frac"]) > 0
                )
            with self.need("TxTime.Frac is non-negative"):
                assert np.all(ppp["TxTime"]["Frac"] >= 0)
            with self.need("TxTime.Frac is less than 1"):
                assert np.all(ppp["TxTime"]["Frac"] < 1)

    @per_sequence
    def check_txoverlap(self, sequence_id, sequence_param_elem):
        """TxTime and TXmt do not lead to overlap"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            tx_time_float = ppp["TxTime"]["Int"] + ppp["TxTime"]["Frac"]
            with self.want("TxTime and TXmt do not lead to overlap"):
                assert np.all(
                    np.diff(tx_time_float) >= (ppp["TXmt"][:-1] + ppp["TXmt"][1:]) / 2
                )

    @per_sequence
    def check_phix0_intfrac(self, sequence_id, sequence_param_elem):
        """PhiX0 PPP is properly formatted as int-frac"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            with self.need("PhiX0.Frac is non-negative"):
                assert np.all(ppp["PhiX0"]["Frac"] >= 0)
            with self.need("PhiX0.Frac is less than 1"):
                assert np.all(ppp["PhiX0"]["Frac"] < 1)

    @per_sequence
    def check_tx_acxy(self, sequence_id, sequence_param_elem):
        """TxACX and TxACY PPPs are orthonormal"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            with self.need("TxACX is unit length"):
                assert np.linalg.norm(ppp["TxACX"], axis=-1) == con.Approx(1.0)
            with self.need("TxACY is unit length"):
                assert np.linalg.norm(ppp["TxACY"], axis=-1) == con.Approx(1.0)
            with self.need("TxACX and TxACY are orthogonal"):
                assert dot(ppp["TxACX"], ppp["TxACY"]) == con.Approx(0.0, atol=1e-6)

    @per_sequence
    def check_txeb(self, sequence_id, sequence_param_elem):
        """TxEB composed of valid direction cosines"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            with self.need("TxEB is less than unit length"):
                assert np.linalg.norm(ppp["TxEB"], axis=-1) <= con.Approx(1.0)

    @per_sequence
    def check_fxresponseindex(self, sequence_id, sequence_param_elem):
        """FxResponseIndex PPP refers to an extant row of the FxResponseArray"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            fx_id = sequence_param_elem.findtext("{*}FxResponseId")
            num_fxr = int(
                self.crsdroot.findtext(
                    f'{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId="{fx_id}"]/{{*}}NumRows'
                )
            )
            with self.need("FxResponseIndex is non-negative"):
                assert np.all(ppp["FxResponseIndex"] >= 0)
            with self.need("FxResponseIndex is less than NumFXR"):
                assert np.all(ppp["FxResponseIndex"] < num_fxr)

    @per_sequence
    def check_fxrate(self, sequence_id, sequence_param_elem):
        """FxRate PPP is zero when TxWFType is XM"""
        with self.precondition():
            assert self.crsdroot.findtext("{*}TxSequence/{*}TxWFType") == "XM"
            ppp = self._get_sequence_ppps(sequence_id)
            with self.need("FxRate is zero"):
                assert np.all(ppp["FxRate"] == 0)

    @per_sequence
    def check_fxfreq0(self, sequence_id, sequence_param_elem):
        """FxFreq0 PPP is positive"""
        with self.precondition():
            ppp = self._get_sequence_ppps(sequence_id)
            with self.need("FxFreq0 is positive"):
                assert np.all(ppp["FxFreq0"] > 0)

    @per_sequence
    def check_xm_pulse_length(self, sequence_id, sequence_param_elem):
        """Each TXmt covers enough XM samples"""
        with self.precondition():
            xm_id = sequence_param_elem.findtext("{*}XMId")
            assert xm_id is not None
            txmtmin = float(sequence_param_elem.findtext("{*}TXmtMin"))
            txmtmax = float(sequence_param_elem.findtext("{*}TXmtMax"))
            ts_xma = float(
                self.crsdroot.findtext(
                    f'{{*}}SupportArray/{{*}}XMArray[{{*}}Identifier="{xm_id}"]/{{*}}TsXMA'
                )
            )
            ns_xma = int(
                self.crsdroot.findtext(
                    f'{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId="{xm_id}"]/{{*}}NumCols'
                )
            )
            with self.need("half of TXmt is at least (close to) 500 XM samples"):
                assert np.round(txmtmin / 2 / ts_xma) >= 500
            with self.need("pulse fits appropriately in the XM Array"):
                assert np.round(txmtmax / 2 / ts_xma) <= (ns_xma - 1) // 2

    @per_channel
    def check_refvectorindex(self, channel_id, channel_param_elem):
        """RefVectorIndex refers to an appropriate extant vector"""
        ref_vector_index = int(channel_param_elem.findtext("{*}RefVectorIndex"))
        num_vectors = int(
            self.crsdroot.findtext(
                f"{{*}}Data/{{*}}Receive/{{*}}Channel[{{*}}ChId='{channel_id}']/{{*}}NumVectors"
            )
        )
        with self.need("RefVectorIndex refers to an extant vector"):
            # nonnegative is checked by the schema
            assert ref_vector_index < num_vectors
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("RefVectorIndex refers to a vector with normal signal"):
                assert pvp["SIGNAL"][ref_vector_index] == 1
            with self.precondition():
                assert self.crsd_type == "CRSDsar"
                with self.need(
                    "RefVectorIndex refers to a vector with a transmit pulse"
                ):
                    assert pvp["TxPulseIndex"][ref_vector_index] != -1

    @per_channel
    def check_reffreqfixed(self, channel_id, channel_param_elem):
        """RefFreqFixed boolean matches PVP"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.precondition():
                assert self.xmlhelp.load_elem(
                    channel_param_elem.find("{*}RefFreqFixed")
                )
                with self.need("RefFreq PVP is fixed"):
                    assert np.ptp(pvp["RefFreq"]) == 0
            with self.precondition():
                assert not self.xmlhelp.load_elem(
                    channel_param_elem.find("{*}RefFreqFixed")
                )
                with self.need("RefFreq PVP is not fixed"):
                    assert np.ptp(pvp["RefFreq"]) != 0

    @per_channel
    def check_frcvfixed(self, channel_id, channel_param_elem):
        """FrcvFixed boolean matches PVPs"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.precondition():
                assert self.xmlhelp.load_elem(channel_param_elem.find("{*}FrcvFixed"))
                with self.need("FRCV1 PVP is fixed"):
                    assert np.ptp(pvp["FRCV1"]) == 0
                with self.need("FRCV2 PVP is fixed"):
                    assert np.ptp(pvp["FRCV2"]) == 0
            with self.precondition():
                assert not self.xmlhelp.load_elem(
                    channel_param_elem.find("{*}FrcvFixed")
                )
                with self.need("Either FRCV1 or FRCV2 is not fixed"):
                    assert np.ptp(pvp["FRCV1"]) != 0 or np.ptp(pvp["FRCV2"]) != 0

    @per_channel
    def check_signalnormal(self, channel_id, channel_param_elem):
        """SignalNormal boolean matches PVP"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.precondition():
                assert self.xmlhelp.load_elem(
                    channel_param_elem.find("{*}SignalNormal")
                )
                with self.need("SIGNAL PVP is always 1"):
                    assert np.all(pvp["SIGNAL"] == 1)
            with self.precondition():
                assert not self.xmlhelp.load_elem(
                    channel_param_elem.find("{*}SignalNormal")
                )
                with self.need("SIGNAL PVP is not always 1"):
                    assert not np.all(pvp["SIGNAL"] == 1)

    @per_channel
    def check_f0ref(self, channel_id, channel_param_elem):
        """F0Ref matches reference vector RefFreq PVP"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            ref_vector_index = int(channel_param_elem.findtext("{*}RefVectorIndex"))
            with self.need("F0Ref matches reference vector RefFreq PVP"):
                assert pvp["RefFreq"][ref_vector_index] == con.Approx(
                    float(channel_param_elem.findtext("{*}F0Ref"))
                )

    @per_channel
    def check_inst_osr(self, channel_id, channel_param_elem):
        """Receive oversample of instantaneous bandwidth"""
        fs = float(channel_param_elem.findtext("{*}Fs"))
        bw_inst = float(channel_param_elem.findtext("{*}BWInst"))
        with self.need("sufficient oversample of instantaneous bandwidth"):
            assert fs >= con.Approx(1.1 * bw_inst)

    @per_channel
    def check_rcvstart12(self, channel_id, channel_param_elem):
        """RcvStartTime1 and RcvStartTime2 are sane and match the PVPs"""
        rcv_start1 = float(channel_param_elem.findtext("{*}RcvStartTime1"))
        rcv_start2 = float(channel_param_elem.findtext("{*}RcvStartTime2"))
        with self.need("RcvStartTime1 not greater than RcvStartTime2"):
            assert rcv_start1 <= rcv_start2
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("RcvStartTime1 matches first PVP"):
                assert rcv_start1 == con.Approx(
                    pvp["RcvStart"][0]["Int"] + pvp["RcvStart"][0]["Frac"]
                )
            with self.need("RcvStartTime2 matches last PVP"):
                assert rcv_start2 == con.Approx(
                    pvp["RcvStart"][-1]["Int"] + pvp["RcvStart"][-1]["Frac"]
                )

    @per_channel
    def check_frcvminmax(self, channel_id, channel_param_elem):
        """FrcvMin and FrcvMax are sane and match the PVPs"""
        frcvmin = float(channel_param_elem.findtext("{*}FrcvMin"))
        frcvmax = float(channel_param_elem.findtext("{*}FrcvMax"))
        with self.need("FrcvMin less than FrcvMax"):
            assert frcvmin < frcvmax
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("FrcvMin matches FRCV1"):
                assert frcvmin == con.Approx(pvp["FRCV1"].min())
            with self.need("FrcvMax matches FRCV2"):
                assert frcvmax == con.Approx(pvp["FRCV2"].max())

    @per_channel
    def check_rcvrefpoint(self, channel_id, channel_param_elem):
        """RcvRefPoint IAC maps to ECF"""
        ref_pt_iac = self.xmlhelp.load_elem(
            channel_param_elem.find("{*}RcvRefPoint/{*}IAC")
        )
        ref_pt_ecf = self.xmlhelp.load_elem(
            channel_param_elem.find("{*}RcvRefPoint/{*}ECF")
        )
        with self.need("IAC maps to ECF"):
            self.assert_iac_matches_ecf(ref_pt_iac, ref_pt_ecf)

    @per_channel
    def check_rcvpolarization(self, channel_id, channel_param_elem):
        """Check the RcvPolarization with respect to HV"""
        with self.need("AmpH and AmpV are normalized"):
            assert float(
                channel_param_elem.findtext("{*}RcvPolarization/{*}AmpH")
            ) ** 2 + float(
                channel_param_elem.findtext("{*}RcvPolarization/{*}AmpV")
            ) ** 2 == con.Approx(1.0)
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            apat_id = channel_param_elem.findtext("{*}RcvAPATId")
            ref_vec_index = int(channel_param_elem.findtext("{*}RefVectorIndex"))
            ref_point = self.xmlhelp.load_elem(
                channel_param_elem.find("{*}RcvRefPoint/{*}ECF")
            )
            ampx = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}AmpX"
                )
            )
            ampy = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}AmpY"
                )
            )
            phasex = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}PhaseX"
                )
            )
            phasey = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}PhaseY"
                )
            )
            acx = pvp["RcvACX"][ref_vec_index]
            acy = pvp["RcvACY"][ref_vec_index]
            apc_pos = pvp["RcvPos"][ref_vec_index]

            amph, ampv, phaseh, phasev = skcrsd.compute_h_v_pol_parameters(
                apc_pos, acx, acy, ref_point, -1, ampx, ampy, phasex, phasey
            )
            with self.need("H component is correct"):
                assert float(
                    channel_param_elem.findtext("{*}RcvPolarization/{*}AmpH")
                ) * np.exp(
                    2j
                    * np.pi
                    * float(channel_param_elem.findtext("{*}RcvPolarization/{*}PhaseH"))
                ) == con.Approx(amph * np.exp(2j * np.pi * phaseh), atol=1e-3)
            with self.need("V component is correct"):
                assert float(
                    channel_param_elem.findtext("{*}RcvPolarization/{*}AmpV")
                ) * np.exp(
                    2j
                    * np.pi
                    * float(channel_param_elem.findtext("{*}RcvPolarization/{*}PhaseV"))
                ) == con.Approx(ampv * np.exp(2j * np.pi * phasev), atol=1e-3)

    @per_channel
    def check_sartxpolarization(self, channel_id, channel_param_elem):
        """Check the SARImage TxPolarization with respect to HV"""
        with self.precondition():
            sequence_id = channel_param_elem.findtext("{*}SARImage/{*}TxId")
            assert sequence_id is not None
            with self.need("AmpH and AmpV are normalized"):
                assert float(
                    channel_param_elem.findtext("{*}SARImage/{*}TxPolarization/{*}AmpH")
                ) ** 2 + float(
                    channel_param_elem.findtext("{*}SARImage/{*}TxPolarization/{*}AmpV")
                ) ** 2 == con.Approx(1.0)
            sequence_param_elem = self.crsdroot.find(
                f"{{*}}TxSequence/{{*}}Parameters[{{*}}Identifier='{sequence_id}']"
            )
            ppp = self._get_sequence_ppps(sequence_id)
            apat_id = sequence_param_elem.findtext("{*}TxAPATId")
            pulse_index = int(
                channel_param_elem.findtext("{*}SARImage/{*}RefVectorPulseIndex")
            )
            ref_point = self.xmlhelp.load_elem(
                channel_param_elem.find("{*}RcvRefPoint/{*}ECF")
            )
            ampx = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}AmpX"
                )
            )
            ampy = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}AmpY"
                )
            )
            phasex = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}PhaseX"
                )
            )
            phasey = float(
                self.crsdroot.findtext(
                    f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']/{{*}}AntPolRef/{{*}}PhaseY"
                )
            )
            acx = ppp["TxACX"][pulse_index]
            acy = ppp["TxACY"][pulse_index]
            apc_pos = ppp["TxPos"][pulse_index]

            amph, ampv, phaseh, phasev = skcrsd.compute_h_v_pol_parameters(
                apc_pos, acx, acy, ref_point, 1, ampx, ampy, phasex, phasey
            )
            with self.need("H component is correct"):
                assert float(
                    channel_param_elem.findtext("{*}SARImage/{*}TxPolarization/{*}AmpH")
                ) * np.exp(
                    2j
                    * np.pi
                    * float(
                        channel_param_elem.findtext(
                            "{*}SARImage/{*}TxPolarization/{*}PhaseH"
                        )
                    )
                ) == con.Approx(amph * np.exp(2j * np.pi * phaseh), atol=1e-3)
            with self.need("V component is correct"):
                assert float(
                    channel_param_elem.findtext("{*}SARImage/{*}TxPolarization/{*}AmpV")
                ) * np.exp(
                    2j
                    * np.pi
                    * float(
                        channel_param_elem.findtext(
                            "{*}SARImage/{*}TxPolarization/{*}PhaseV"
                        )
                    )
                ) == con.Approx(ampv * np.exp(2j * np.pi * phasev), atol=1e-3)

    @per_channel
    def check_refvectorpulseindex(self, channel_id, channel_param_elem):
        """RefVectorPulseIndex is sane and matches PPP"""
        with self.precondition():
            assert self.crsd_type == "CRSDsar"
            ref_vector_pulse_index = int(
                channel_param_elem.findtext("{*}SARImage/{*}RefVectorPulseIndex")
            )
            txid = channel_param_elem.findtext("{*}SARImage/{*}TxId")
            num_pulses = int(
                self.crsdroot.findtext(
                    f'{{*}}Data/{{*}}Transmit/{{*}}TxSequence[{{*}}TxId="{txid}"]/{{*}}NumPulses'
                )
            )
            with self.need("RefVectorPulseIndex references an extant pulse"):
                assert ref_vector_pulse_index < num_pulses
            ref_vector_index = int(channel_param_elem.findtext("{*}RefVectorIndex"))
            with self.precondition():
                pvp = self._get_channel_pvps(channel_id)
                with self.need("RefVectorPulseIndex matches PVP"):
                    assert (
                        pvp["TxPulseIndex"][ref_vector_index] == ref_vector_pulse_index
                    )

    @per_channel
    def check_rcvstart(self, channel_id, channel_param_elem):
        """RcvStart PVP is increasing and properly formatted as int-frac"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("RcvStart is strictly increasing"):
                assert np.all(
                    np.diff(pvp["RcvStart"]["Int"]) + np.diff(pvp["RcvStart"]["Frac"])
                    > 0
                )
            with self.need("RcvStart.Frac is non-negative"):
                assert np.all(pvp["RcvStart"]["Frac"] >= 0)
            with self.need("RcvStart.Frac is less than 1"):
                assert np.all(pvp["RcvStart"]["Frac"] < 1)

    @per_channel
    def check_rcvstart_sample(self, channel_id, channel_param_elem):
        """RcvStart is an integer multiple of sample periods after the first in the channel"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            rel_rcv_start = (pvp["RcvStart"]["Int"] - pvp["RcvStart"][0]["Int"]) + (
                pvp["RcvStart"]["Frac"] - pvp["RcvStart"][0]["Frac"]
            )
            sample_frequency = float(channel_param_elem.findtext("{*}Fs"))
            with self.need(
                "RcvStart is an integer multiple of sample periods after the first in the channel"
            ):
                assert rel_rcv_start * sample_frequency == con.Approx(
                    np.round(rel_rcv_start * sample_frequency), atol=1e-3, rtol=0
                )

    @per_channel
    def check_rcvstart_overlap(self, channel_id, channel_param_elem):
        """RcvStart, Fs, and NumSamples do not describe overlapping receive windows"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            delta_rcv_start = np.diff(pvp["RcvStart"]["Int"]) + np.diff(
                pvp["RcvStart"]["Frac"]
            )
            sample_frequency = float(channel_param_elem.findtext("{*}Fs"))
            ns = int(
                self.crsdroot.findtext(
                    f'{{*}}Data/{{*}}Receive/{{*}}Channel[{{*}}ChId="{channel_id}"]/{{*}}NumSamples'
                )
            )
            rcv_duration = ns / sample_frequency

            with self.need("RcvStart, Fs and NumSamples do not describe overlap"):
                assert delta_rcv_start.min() >= con.Approx(rcv_duration)

    @per_channel
    def check_rcvposveltime(self, channel_id, channel_param_elem):
        """RcvPos, RcvVel, RcvStart seem normal"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            rcvtime_float = pvp["RcvStart"]["Int"] + pvp["RcvStart"]["Frac"]
            with self.want("RcvPos, RcvVel, RcvStart are a matched set"):
                assert np.diff(pvp["RcvPos"], axis=0) == con.Approx(
                    np.diff(rcvtime_float)[:, np.newaxis]
                    * (pvp["RcvVel"][:-1] + pvp["RcvVel"][1:])
                    / 2,
                    rtol=0.2,
                )
            with self.want("RcvPos is near Earth"):
                assert np.linalg.norm(pvp["RcvPos"], axis=-1) == con.Approx(
                    sarkit.wgs84.SEMI_MAJOR_AXIS, rtol=0.5
                )

    @per_channel
    def check_refphi0_intfrac(self, channel_id, channel_param_elem):
        """RefPhi0 PVP is properly formatted as int-frac"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("RefPhi0.Frac is non-negative"):
                assert np.all(pvp["RefPhi0"]["Frac"] >= 0)
            with self.need("RefPhi0.Frac is less than 1"):
                assert np.all(pvp["RefPhi0"]["Frac"] < 1)

    @per_channel
    def check_signal_pvp(self, channel_id, channel_param_elem):
        """SIGNAL PVP is in range"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("SIGNAL PVP is in range"):
                assert np.all(pvp["SIGNAL"] >= 0)
                assert np.all(pvp["SIGNAL"] <= 3)

    @per_channel
    def check_ampsf_pvp(self, channel_id, channel_param_elem):
        """AmpSF PVP is positive"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("AmpSF PVP is positive"):
                assert np.all(pvp["AmpSF"] > 0)

    @per_channel
    def check_txpulseindex_pvp(self, channel_id, channel_param_elem):
        """TxPulseIndex refers to an extant pulse or the 'no pulse' value"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            assert "TxPulseIndex" in pvp.dtype.fields
            tx_id = channel_param_elem.findtext("{*}SARImage/{*}TxId")
            num_pulses = int(
                self.crsdroot.findtext(
                    f'{{*}}Data/{{*}}Transmit/{{*}}TxSequence[{{*}}TxId="{tx_id}"]/{{*}}NumPulses'
                )
            )
            with self.need("TxPulseIndex is either -1 or a valid index"):
                assert np.all(
                    np.logical_and(
                        pvp["TxPulseIndex"] >= -1, pvp["TxPulseIndex"] < num_pulses
                    )
                )
            with self.need("TxPulseIndex is -1 when SIGNAL is 2"):
                assert np.all(pvp["TxPulseIndex"][pvp["SIGNAL"] == 2] == -1)
            with self.need("TxPulseIndex is not -1 when SIGNAL is 1"):
                assert np.all(pvp["TxPulseIndex"][pvp["SIGNAL"] == 1] != -1)

    @per_channel
    def check_rcv_acxy(self, channel_id, channel_param_elem):
        """RcvACX and RcvACY PVPs are orthonormal"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("RcvACX is unit length"):
                assert np.linalg.norm(pvp["RcvACX"], axis=-1) == con.Approx(1.0)
            with self.need("RcvACY is unit length"):
                assert np.linalg.norm(pvp["RcvACY"], axis=-1) == con.Approx(1.0)
            with self.need("RcvACX and RcvACY are orthogonal"):
                assert dot(pvp["RcvACX"], pvp["RcvACY"]) == con.Approx(0.0, atol=1e-6)

    @per_channel
    def check_rcveb(self, channel_id, channel_param_elem):
        """RcvEB composed of valid direction cosines"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("RcvEB is less than unit length"):
                assert np.linalg.norm(pvp["RcvEB"], axis=-1) <= con.Approx(1.0)

    @per_channel
    def check_dfic0_ficrate(self, channel_id, channel_param_elem):
        """DFIC0 and FICRate match"""
        with self.precondition():
            pvp = self._get_channel_pvps(channel_id)
            with self.need("DFIC0 = 0 when FICRate = 0"):
                assert np.all(pvp["DFIC0"][pvp["FICRate"] == 0] == 0)
            ns = int(
                self.crsdroot.findtext(
                    f'{{*}}Data/{{*}}Receive/{{*}}Channel[{{*}}ChId="{channel_id}"]/{{*}}NumSamples'
                )
            )
            fs = float(channel_param_elem.findtext("{*}Fs"))
            rcv_duration = ns * fs
            with self.need(
                "DFIC0 + FICRate * t = 0 for some point in the receive window"
            ):
                assert np.all(
                    pvp["DFIC0"] * (pvp["DFIC0"] + pvp["FICRate"] * rcv_duration) <= 0
                )

    @per_channel
    def check_dwell_array_coverage(self, channel_id, channel_param_elem):
        """DwellTimeArrays cover SARImage ImageArea"""
        with self.precondition():
            dta_id = channel_param_elem.findtext(
                "{*}SARImage/{*}DwellTimes/{*}Array/{*}DTAId"
            )
            assert dta_id is not None
            dta_node = self.crsdroot.find(
                f'{{*}}SupportArray/{{*}}DwellTimeArray[{{*}}Identifier="{dta_id}"]'
            )
            x0 = float(dta_node.findtext("{*}X0"))
            y0 = float(dta_node.findtext("{*}Y0"))
            xss = float(dta_node.findtext("{*}XSS"))
            yss = float(dta_node.findtext("{*}YSS"))
            dta_data_node = self.crsdroot.find(
                f'{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId="{dta_id}"]'
            )
            num_rows = int(dta_data_node.findtext("{*}NumRows"))
            num_cols = int(dta_data_node.findtext("{*}NumCols"))
            x1y1 = self.xmlhelp.load_elem(
                channel_param_elem.find("{*}SARImage/{*}ImageArea/{*}X1Y1")
            )
            x2y2 = self.xmlhelp.load_elem(
                channel_param_elem.find("{*}SARImage/{*}ImageArea/{*}X2Y2")
            )
            with self.need("Dwell array covers X1Y1"):
                assert np.all((x0 - 0.5 * xss, y0 - 0.5 * yss) <= x1y1)
            with self.need("Dwell array covers X2Y2"):
                assert np.all(
                    (x0 + xss * (num_rows - 0.5), y0 + yss * (num_cols - 0.5)) >= x2y2
                )

    def check_numcodtimes(self):
        """CODTime polynomials are properly counted"""
        with self.precondition():
            assert self.crsdroot.find("{*}DwellPolynomials") is not None
            self._check_num(
                "{*}DwellPolynomials/{*}NumCODTimes", "{*}DwellPolynomials/{*}CODTime"
            )

    def check_numdwelltimes(self):
        """DwellTime polynomials are properly counted"""
        with self.precondition():
            assert self.crsdroot.find("{*}DwellPolynomials") is not None
            self._check_num(
                "{*}DwellPolynomials/{*}NumDwellTimes",
                "{*}DwellPolynomials/{*}DwellTime",
            )

    def check_scene_iarp(self):
        """IARP is consistent and near Earth's surface"""
        iarp_ecf = self.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}ECF")
        iarp_llh = self.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}LLH")
        iarp_llh_ecf = sarkit.wgs84.geodetic_to_cartesian(iarp_llh)
        with self.need("IARP ECF matches LLH"):
            assert np.linalg.norm(iarp_ecf - iarp_llh_ecf) < 1
        with self.want("IARP is near Earth's surface"):
            assert np.abs(iarp_llh[2]) < 100e3

    def check_scene_planar_axes(self):
        """Planar uIAX and uIAY are orthonormal and IAZ is upward"""
        with self.precondition():
            assert (
                self.crsdroot.find("{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar")
                is not None
            )
            uiax = self.xmlhelp.load(
                "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"
            )
            uiay = self.xmlhelp.load(
                "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"
            )
            iarp = self.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}ECF")
            with self.need("uIAX is unit length"):
                assert np.linalg.norm(uiax) == con.Approx(1.0)
            with self.need("uIAY is unit length"):
                assert np.linalg.norm(uiay) == con.Approx(1.0)
            with self.need("uIAX and uIAY are orthogonal"):
                assert abs(dot(uiax, uiay)) < 1e-6
            with self.want("uIAZ is upward"):
                assert dot(iarp, np.cross(uiax, uiay)) > 0

    def check_scene_hae_axes(self):
        """HAE uIAX and uIAY are orthonormal in ECF and IAZ is upward"""
        with self.precondition():
            assert (
                self.crsdroot.find("{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE")
                is not None
            )
            uiax = self.xmlhelp.load(
                "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAXLL"
            )
            uiay = self.xmlhelp.load(
                "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAYLL"
            )
            iarp_ecf = self.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}ECF")
            iarp_llh = self.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}LLH")
            uiax_pt = iarp_llh.copy()
            uiax_pt[:2] += np.rad2deg(uiax)
            uiax_ecf = sarkit.wgs84.geodetic_to_cartesian(uiax_pt) - iarp_ecf
            uiay_pt = iarp_llh.copy()
            uiay_pt[:2] += np.rad2deg(uiay)
            uiay_ecf = sarkit.wgs84.geodetic_to_cartesian(uiay_pt) - iarp_ecf

            # looser tols on HAE unit vectors because the transformation depends on scale
            with self.need("uIAXLL is unit length in ECF"):
                assert np.linalg.norm(uiax_ecf) == con.Approx(1.0, atol=1e-3)
            with self.need("uIAYLL is unit length in ECF"):
                assert np.linalg.norm(uiay_ecf) == con.Approx(1.0, atol=1e-3)
            with self.need("uIAXLL and uIAYLL are orthogonal in ECF"):
                assert abs(dot(uiax_ecf, uiay_ecf)) < 1e-5
            with self.want("uIAZ is upward"):
                assert dot(iarp_ecf, np.cross(uiax_ecf, uiay_ecf)) > 0

    def check_iacp(self):
        """IACPs are reasonable"""
        x1y1 = self.xmlhelp.load("{*}SceneCoordinates/{*}ImageArea/{*}X1Y1")
        x2y2 = self.xmlhelp.load("{*}SceneCoordinates/{*}ImageArea/{*}X2Y2")
        iacps = self.xmlhelp.load("{*}SceneCoordinates/{*}ImageAreaCornerPoints")
        # lay out the XY corners in clockwise order
        xy_lim_corners = [x1y1, [x1y1[0], x2y2[1]], x2y2, [x2y2[0], x1y1[1]]]
        xy_lim_ecf = [self.iac_to_ecf(xy) for xy in xy_lim_corners]
        xy_lim_llh = sarkit.wgs84.cartesian_to_geodetic(xy_lim_ecf)
        # find which IACP most closely corresponds to the first XY corner
        num_roll = np.argmin(np.linalg.norm(xy_lim_llh[0, :2] - iacps))
        # make first xy_lim_llh correspond with first IACP
        xy_lim_llh = np.roll(xy_lim_llh, num_roll)
        with self.need("IACP are clockwise and match X1Y1, X2Y2"):
            assert (
                np.max(np.linalg.norm(xy_lim_llh[:, :2] - iacps)) < 1e-4
            )  # roughly 11m at equator

    def verify_imagearea_polygon(self, polygon_node):
        """Verify that an image area polygon is reasonable"""
        polygon = self.xmlhelp.load_elem(polygon_node)
        x1y1 = self.xmlhelp.load_elem(polygon_node.getparent().find("{*}X1Y1"))
        x2y2 = self.xmlhelp.load_elem(polygon_node.getparent().find("{*}X2Y2"))
        with self.need("Polygon is bounded by X1Y1 X2Y2"):
            assert polygon.min(axis=0) == con.Approx(x1y1)
            assert polygon.max(axis=0) == con.Approx(x2y2)
        shg_polygon = shg.Polygon(polygon)
        with self.need("Polygon is simple"):
            assert shg_polygon.is_simple
        with self.need("Polygon is clockwise"):
            assert not shg_polygon.exterior.is_ccw

    def check_scene_imagearea_polygon(self):
        """Check SceneCoordinates/ImageArea/Polygon"""
        self.verify_imagearea_polygon(
            self.crsdroot.find("{*}SceneCoordinates/{*}ImageArea/{*}Polygon")
        )

    def check_scene_extendedarea_polygon(self):
        """Check SceneCoordinates/ExtendedArea/Polygon"""
        with self.precondition():
            assert self.crsdroot.find("{*}SceneCoordinates/{*}ExtendedArea") is not None
            self.verify_imagearea_polygon(
                self.crsdroot.find("{*}SceneCoordinates/{*}ExtendedArea/{*}Polygon")
            )

    @per_channel
    def check_channel_imagearea_polygon(self, channel_id, channel_param_elem):
        """Check channel ImageArea Polygon"""
        with self.precondition():
            assert channel_param_elem.find("{*}SARImage") is not None
            self.verify_imagearea_polygon(
                channel_param_elem.find("{*}SARImage/{*}ImageArea/{*}Polygon")
            )

    def check_segment_list(self):
        """Segments are within the grid"""
        with self.precondition():
            assert (
                self.crsdroot.find("{*}SceneCoordinates/{*}ImageGrid/{*}SegmentList")
                is not None
            )
            segment_nodes = self.crsdroot.findall(
                "{*}SceneCoordinates/{*}ImageGrid/{*}SegmentList/{*}Segment"
            )
            with self.need("NumSegments is correct"):
                assert int(
                    self.crsdroot.findtext(
                        "{*}SceneCoordinates/{*}ImageGrid/{*}SegmentList/{*}NumSegments"
                    )
                ) == len(segment_nodes)

            first_line = int(
                self.crsdroot.findtext(
                    "{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}FirstLine"
                )
            )
            num_lines = int(
                self.crsdroot.findtext(
                    "{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}NumLines"
                )
            )
            first_sample = int(
                self.crsdroot.findtext(
                    "{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}FirstSample"
                )
            )
            num_samples = int(
                self.crsdroot.findtext(
                    "{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}NumSamples"
                )
            )
            for segment_node in segment_nodes:
                identifier = segment_node.findtext("{*}Identifier")
                start_line = int(segment_node.findtext("{*}StartLine"))
                start_sample = int(segment_node.findtext("{*}StartSample"))
                end_line = int(segment_node.findtext("{*}EndLine"))
                end_sample = int(segment_node.findtext("{*}EndSample"))
                with self.need(f"{identifier}: StartLine is within grid"):
                    assert first_line <= start_line <= first_line + num_lines - 1
                with self.need(f"{identifier}: StartSample is within grid"):
                    assert (
                        first_sample <= start_sample <= first_sample + num_samples - 1
                    )
                with self.need(f"{identifier}: StartLine is less than EndLine"):
                    assert start_line < end_line
                with self.need(f"{identifier}: StartSample is less than EndSample"):
                    assert start_sample < end_sample
                with self.need(f"{identifier}: EndLine is within grid"):
                    assert first_line <= end_line <= first_line + num_lines - 1
                with self.need(f"{identifier}: EndSample is within grid"):
                    assert first_sample <= end_sample <= first_sample + num_samples - 1
                polygon = self.xmlhelp.load_elem(segment_node.find("{*}SegmentPolygon"))
                with self.need(f"{identifier}: Polygon is bounded by segment limits"):
                    assert np.all(
                        polygon.min(axis=0) >= [start_line - 0.5, start_sample - 0.5]
                    )
                    assert np.all(
                        polygon.max(axis=0) <= [end_line + 0.5, end_sample + 0.5]
                    )
                shg_polygon = shg.Polygon(polygon)
                with self.need(f"{identifier}: Polygon is simple"):
                    assert shg_polygon.is_simple
                with self.need(f"{identifier}: Polygon is clockwise"):
                    assert not shg_polygon.exterior.is_ccw

    def check_numsupportarrays(self):
        """SupportArrays are properly counted"""
        self._check_num(
            "{*}Data/{*}Support/{*}NumSupportArrays",
            "{*}Data/{*}Support/{*}SupportArray",
        )

    def check_support_array_bytes_per_element(self):
        """Support array bytes per element matches the element type"""
        support_array_ids = [
            x.text
            for x in self.crsdroot.findall("{*}Data/{*}Support/{*}SupportArray/{*}SAId")
        ]
        for support_array_id in support_array_ids:
            element_format = self.crsdroot.findtext(
                f'{{*}}SupportArray//*[{{*}}Identifier="{support_array_id}"]/{{*}}ElementFormat'
            )
            dtype = skcrsd.binary_format_string_to_dtype(element_format)
            bytes_per_element = int(
                self.crsdroot.findtext(
                    f'{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId="{support_array_id}"]/{{*}}BytesPerElement'
                )
            )
            with self.need(f"{support_array_id} BytesPerElement matches ElementFormat"):
                assert bytes_per_element == dtype.itemsize

    def check_numtxsequences(self):
        """Transmit sequences are properly counted"""
        with self.precondition():
            assert self.crsdroot.find("{*}Data/{*}Transmit") is not None
            self._check_num(
                "{*}Data/{*}Transmit/{*}NumTxSequences",
                "{*}Data/{*}Transmit/{*}TxSequence",
            )

    def check_numbytesppp(self):
        """NumBytesPPP matches PPP data structure"""
        with self.precondition():
            assert self.crsdroot.find("{*}Data/{*}Transmit") is not None
            num_bytes_ppp = int(
                self.crsdroot.findtext("{*}Data/{*}Transmit/{*}NumBytesPPP")
            )
            with self.need("NumBytesPPP matches PPP structure"):
                assert num_bytes_ppp == skcrsd.get_ppp_dtype(self.crsdroot).itemsize

    def check_ppp_min_offset(self):
        """Minimum offset of any PPP is zero"""
        with self.precondition():
            assert self.crsdroot.find("{*}PPP") is not None
            offsets = [int(x.text) for x in self.crsdroot.findall("{*}PPP//{*}Offset")]
            with self.need("First PPP has offset zero"):
                assert min(offsets) == 0

    def check_ppp_unique_names(self):
        """Names of PPPs are all unique"""
        with self.precondition():
            assert self.crsdroot.find("{*}PPP") is not None
            node_names = [
                etree.QName(x).localname for x in self.crsdroot.findall("{*}PPP/*")
            ]
            not_added_names = [name for name in node_names if name != "AddedPPP"]
            added_names = [
                x.text for x in self.crsdroot.findall("{*}PPP/{*}AddedPPP/{*}Name")
            ]
            names = not_added_names + added_names
            with self.need("PPP names are unique"):
                assert len(set(names)) == len(names)
            # The only non-required defined PPP is XMIndex
            # An AddedPPP with name XMIndex may be missed in the previous check
            with self.need("XMIndex is not an AddedPPP name"):
                assert "XMIndex" not in added_names

    def check_numcrsdchannels(self):
        """Channels are properly counted"""
        with self.precondition():
            assert self.crsdroot.find("{*}Data/{*}Receive") is not None
            self._check_num(
                "{*}Data/{*}Receive/{*}NumCRSDChannels", "{*}Data/{*}Receive/{*}Channel"
            )

    def check_numbytespvp(self):
        """NumBytesPVP matches PVP data structure"""
        with self.precondition():
            assert self.crsdroot.find("{*}Data/{*}Receive") is not None
            num_bytes_pvp = int(
                self.crsdroot.findtext("{*}Data/{*}Receive/{*}NumBytesPVP")
            )
            with self.need("NumBytesPVP matches PVP structure"):
                assert num_bytes_pvp == skcrsd.get_pvp_dtype(self.crsdroot).itemsize

    def check_pvp_min_offset(self):
        """Minimum offset of any PVP is zero"""
        with self.precondition():
            assert self.crsdroot.find("{*}PVP") is not None
            offsets = [int(x.text) for x in self.crsdroot.findall("{*}PVP//{*}Offset")]
            with self.need("First PVP has offset zero"):
                assert min(offsets) == 0

    def check_pvp_unique_names(self):
        """Names of PVPs are all unique"""
        with self.precondition():
            assert self.crsdroot.find("{*}PVP") is not None
            node_names = [
                etree.QName(x).localname for x in self.crsdroot.findall("{*}PVP/*")
            ]
            not_added_names = [name for name in node_names if name != "AddedPVP"]
            added_names = [
                x.text for x in self.crsdroot.findall("{*}PVP/{*}AddedPVP/{*}Name")
            ]
            names = not_added_names + added_names
            with self.need("PVP names are unique"):
                assert len(set(names)) == len(names)
            # The only non-required defined PVP is TxPulseIndex
            # An AddedPVP with name TxPulseIndex may be missed in the previous check
            with self.need("TxPulseIndex is not an AddedPVP name"):
                assert "TxPulseIndex" not in added_names

    def check_support_array_nodata(self):
        """NODATA entries are the correct length for the ElementFormat"""
        for sa_element in self.crsdroot.findall("{*}SupportArray/*[{*}NODATA]"):
            identifier = sa_element.findtext("{*}Identifier")
            nodata_value = sa_element.findtext("{*}NODATA")
            num_bytes = int(
                self.crsdroot.findtext(
                    f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{identifier}']/{{*}}BytesPerElement"
                )
            )
            with self.need(f"SupportArray: {identifier} NODATA is the right size"):
                assert num_bytes * 2 == len(nodata_value)

    def check_ns_fxr(self):
        """Check that each FxResponseArray has at least 3 columns"""
        fx_ids = self.crsdroot.findall(
            "{*}SupportArray/{*}FxResponseArray/{*}Identifier"
        )
        for fx_id in fx_ids:
            with self.need(f"FxResponseArray '{fx_id.text}' has at least 3 columns"):
                num_cols = self.crsdroot.findtext(
                    f'{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId="{fx_id.text}"]/{{*}}NumCols'
                )
                assert int(num_cols) >= 3

    def check_maxxmbw(self):
        """Check that the oversample ratio of each XMArray is sufficient"""
        xm_ids = self.crsdroot.findall("{*}SupportArray/{*}XMArray/{*}Identifier")
        for xm_id in xm_ids:
            ts_xma = float(
                self.crsdroot.findtext(
                    f'{{*}}SupportArray/{{*}}XMArray[{{*}}Identifier="{xm_id.text}"]/{{*}}TsXMA'
                )
            )
            max_xm_bw = float(
                self.crsdroot.findtext(
                    f'{{*}}SupportArray/{{*}}XMArray[{{*}}Identifier="{xm_id.text}"]/{{*}}MaxXMBW'
                )
            )
            with self.need(f"XMArray '{xm_id}' OSR > 1.1"):
                assert 1.1 * max_xm_bw <= 1 / ts_xma

    def check_numacfs(self):
        """ACFs are properly counted"""
        self._check_num("{*}Antenna/{*}NumACFs", "{*}Antenna/{*}AntCoordFrame")

    def check_numapcs(self):
        """APCs are properly counted"""
        self._check_num("{*}Antenna/{*}NumAPCs", "{*}Antenna/{*}AntPhaseCenter")

    def check_numapats(self):
        """APATs are properly counted"""
        self._check_num("{*}Antenna/{*}NumAPATs", "{*}Antenna/{*}AntPattern")

    def check_ant_pol_ref(self):
        """AntPolRef is mathematically sound"""
        for pattern in self.crsdroot.findall("{*}Antenna/{*}AntPattern"):
            pat_id = pattern.findtext("{*}Identifier")
            with self.need(f"AntPattern {pat_id} AntPolRef has unit amplitude"):
                assert float(pattern.findtext("{*}AntPolRef/{*}AmpX")) ** 2 + float(
                    pattern.findtext("{*}AntPolRef/{*}AmpY")
                ) ** 2 == con.Approx(1.0)

    def check_ant_gain_phase(self):
        """Pattern Gain and Phase are zero at 0, 0"""
        for gp_array in self.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
            identifier = gp_array.findtext("{*}Identifier")
            x0 = float(gp_array.findtext("{*}X0"))
            y0 = float(gp_array.findtext("{*}Y0"))
            xss = float(gp_array.findtext("{*}XSS"))
            yss = float(gp_array.findtext("{*}YSS"))
            xind_00 = -x0 / xss
            yind_00 = -y0 / yss
            data_node = self.crsdroot.find(
                f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{identifier}']"
            )
            num_rows = int(data_node.findtext("{*}NumRows"))
            num_cols = int(data_node.findtext("{*}NumCols"))
            with self.want(f"GPArray {identifier} contains (0, 0)"):
                assert xind_00 >= 0
                assert xind_00 <= num_rows - 1
                assert yind_00 >= 0
                assert yind_00 <= num_cols - 1
                with self.precondition(f"GPArray {identifier} has a sample at (0, 0)"):
                    assert round(xind_00) == con.Approx(xind_00)
                    assert round(yind_00) == con.Approx(yind_00)
                    support_array = self._get_support_array(identifier)
                    data = support_array[int(round(xind_00)), int(round(yind_00))]
                    with self.want(
                        f"GPArray {identifier} is has data in sample at (0, 0)"
                    ):
                        assert not data.recordmask
                        with self.need(f"GPArray {identifier} has zero Gain at (0, 0)"):
                            assert data["Gain"] == con.Approx(0.0)
                        with self.need(
                            f"GPArray {identifier} has zero Phase at (0, 0)"
                        ):
                            assert data["Phase"] == con.Approx(0.0)

    def check_ant_gp_extent(self):
        """Antenna GainPhase array does not go outside [-1, 1]"""
        for gp_array in self.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
            identifier = gp_array.findtext("{*}Identifier")
            x0 = float(gp_array.findtext("{*}X0"))
            y0 = float(gp_array.findtext("{*}Y0"))
            xss = float(gp_array.findtext("{*}XSS"))
            yss = float(gp_array.findtext("{*}YSS"))
            data_node = self.crsdroot.find(
                f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{identifier}']"
            )
            num_rows = int(data_node.findtext("{*}NumRows"))
            num_cols = int(data_node.findtext("{*}NumCols"))
            with self.need(f"GPArray {identifier} is within unit square"):
                assert x0 >= -1
                assert y0 >= -1
                assert x0 + (num_rows - 1) * xss <= con.Approx(1.0)
                assert y0 + (num_cols - 1) * yss <= con.Approx(1.0)

    def check_ant_gp_size(self):
        """Antenna GainPhase array is at least 2x2 in size"""
        for gp_array in self.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
            identifier = gp_array.findtext("{*}Identifier")
            data_node = self.crsdroot.find(
                f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{identifier}']"
            )
            num_rows = int(data_node.findtext("{*}NumRows"))
            num_cols = int(data_node.findtext("{*}NumCols"))
            with self.need(f"GPArray {identifier} is at least 2x2"):
                assert num_rows >= 2
                assert num_cols >= 2

    def check_ant_gp_nan(self):
        """Antenna GainPhase array contains no NaN outside of NODATA"""
        for gp_array in self.crsdroot.findall("{*}SupportArray/{*}GainPhaseArray"):
            identifier = gp_array.findtext("{*}Identifier")
            with self.precondition():
                support_array = self._get_support_array(identifier)
                with self.need(
                    f"GPArray {identifier} contains no NaN outside of NODATA values"
                ):
                    assert np.all(support_array == support_array)

    def assert_symmetric_positive_semidefinite(self, matrix, name):
        """Assert argument is a symmetric, positive-semidefinite matrix"""
        matrix = np.asarray(matrix)
        with self.need(f"{name} is symmetric"):
            assert np.all(matrix == matrix.T)
        with self.need(f"{name} has only non-negative eigenvalues"):
            assert np.all(np.linalg.eigvalsh(matrix) >= 0)

    def check_error_mono_bistatic(self):
        """The correct SARImage ErrorParameters branch is used"""
        collect_type = self.crsdroot.findtext("{*}SARInfo/{*}CollectType")
        with self.precondition():
            assert self.crsdroot.find("{*}ErrorParameters") is not None
            assert collect_type == "MONOSTATIC"
            with self.need("Monostatic ErrorParameters branch is used"):
                assert (
                    self.crsdroot.find("{*}ErrorParameters/{*}SARImage/{*}Monostatic")
                    is not None
                )
        with self.precondition():
            assert self.crsdroot.find("{*}ErrorParameters") is not None
            assert collect_type == "BISTATIC"
            with self.need("Bistatic ErrorParameters branch is used"):
                assert (
                    self.crsdroot.find("{*}ErrorParameters/{*}SARImage/{*}Bistatic")
                    is not None
                )

    def check_error_cov_single_sensor(self):
        """Covariance matrices are mathematically sound for TxSensor, RcvSensor or Monostatic"""
        with self.precondition():
            assert self.crsdroot.find("{*}ErrorParameters") is not None
            assert (
                self.crsdroot.find("{*}ErrorParameters/{*}SARImage/{*}Bistatic") is None
            )
            self.assert_symmetric_positive_semidefinite(
                self.xmlhelp.load("{*}ErrorParameters//{*}PVCov"), "PVCov"
            )
            self.assert_symmetric_positive_semidefinite(
                self.xmlhelp.load("{*}ErrorParameters//{*}TimeFreqCov"), "TimeFreqCov"
            )

    def check_error_cov_bistatic(self):
        """Covariance matrices are mathematically sound for Bistatic"""
        with self.precondition():
            assert (
                self.crsdroot.find("{*}ErrorParameters/{*}SARImage/{*}Bistatic")
                is not None
            )
            self.assert_symmetric_positive_semidefinite(
                self.xmlhelp.load("{*}ErrorParameters//{*}TxPVCov"), "TxPVCov"
            )
            self.assert_symmetric_positive_semidefinite(
                self.xmlhelp.load("{*}ErrorParameters//{*}RcvPVCov"), "RcvPVCov"
            )
            pv_cov = np.block(
                [
                    [
                        self.xmlhelp.load("{*}ErrorParameters//{*}TxPVCov"),
                        self.xmlhelp.load("{*}ErrorParameters//{*}TxRcvPVCov"),
                    ],
                    [
                        self.xmlhelp.load("{*}ErrorParameters//{*}TxRcvPVCov").T,
                        self.xmlhelp.load("{*}ErrorParameters//{*}RcvPVCov"),
                    ],
                ]
            )
            self.assert_symmetric_positive_semidefinite(pv_cov, "Full PVCov (12x12)")
            self.assert_symmetric_positive_semidefinite(
                self.xmlhelp.load("{*}ErrorParameters//{*}TimeFreqCov"), "TimeFreqCov"
            )

    def check_block_order(self):
        """Data blocks are in the correct order and do not overlap"""
        with self.precondition():
            assert self.kvp_list is not None
            prev_block = "XML"
            prev_end = (
                int(self.kvp_list["XML_BLOCK_BYTE_OFFSET"])
                + int(self.kvp_list["XML_BLOCK_SIZE"])
                + 2
            )
            for block in BINARY_BLOCK_ORDER:
                if f"{block}_BLOCK_BYTE_OFFSET" in self.kvp_list:
                    block_offset = int(self.kvp_list.get(f"{block}_BLOCK_BYTE_OFFSET"))
                    with self.need(f"{block} comes after end of {prev_block}"):
                        assert prev_end <= block_offset
                    prev_block = block
                    prev_end = block_offset + int(self.kvp_list[f"{block}_BLOCK_SIZE"])

    def check_post_header_section_terminator_and_pad(self):
        """Section terminator ends the header"""
        with self.precondition():
            assert self.file is not None
            assert self.kvp_list is not None
            xml_offset = int(self.kvp_list["XML_BLOCK_BYTE_OFFSET"])
            self.file.seek(0)
            header_and_pad = self.file.read(xml_offset)
            end_of_header = header_and_pad.find(skcrsd.SECTION_TERMINATOR)
            with self.need("section terminator exists before XML block"):
                assert end_of_header > 0
            with self.need("pad between header and XML is zeros"):
                assert (
                    np.count_nonzero(
                        np.frombuffer(
                            header_and_pad[end_of_header + 2 :], dtype=np.uint8
                        )
                    )
                    == 0
                )

    def check_post_xml_section_terminator(self):
        """Section terminator is after the XML"""
        with self.precondition():
            assert self.file is not None
            assert self.kvp_list is not None
            xml_offset = int(self.kvp_list["XML_BLOCK_BYTE_OFFSET"])
            xml_size = int(self.kvp_list["XML_BLOCK_SIZE"])
            self.file.seek(xml_offset + xml_size)
            with self.need("section terminator at end of XML block"):
                assert self.file.read(2) == skcrsd.SECTION_TERMINATOR

    def check_pad_before_binary_blocks(self):
        """Pad before binary blocks is null bytes"""
        with self.precondition():
            assert self.file is not None
            assert self.kvp_list is not None
            xml_offset = int(self.kvp_list["XML_BLOCK_BYTE_OFFSET"])
            xml_size = int(self.kvp_list["XML_BLOCK_SIZE"])
            previous_block_end = xml_offset + xml_size + 2
            previous_block = "XML"
            for block in BINARY_BLOCK_ORDER:
                with self.precondition():
                    assert f"{block}_BLOCK_BYTE_OFFSET" in self.kvp_list
                    block_offset = int(self.kvp_list[f"{block}_BLOCK_BYTE_OFFSET"])
                    block_size = int(self.kvp_list[f"{block}_BLOCK_SIZE"])
                    block_end = block_offset + block_size
                    self.file.seek(previous_block_end)
                    pad_bytes = self.file.read(block_offset - previous_block_end)
                    pad_bytes_array = np.frombuffer(pad_bytes, dtype=np.uint8)
                    with self.need(
                        f"pad bytes between {previous_block} and {block} are all zero"
                    ):
                        assert np.count_nonzero(pad_bytes_array) == 0
                    previous_block = block
                    previous_block_end = block_end

    def check_support_block_size_and_packing(self):
        """Support block is correctly sized and packed"""
        support_array_offset_size = {}
        for support_array_node in self.crsdroot.findall(
            "{*}Data/{*}Support/{*}SupportArray"
        ):
            array_id = support_array_node.findtext("{*}SAId")

            support_array_offset_size[array_id] = (
                int(support_array_node.findtext("{*}ArrayByteOffset")),
                (
                    int(support_array_node.findtext("{*}NumRows"))
                    * int(support_array_node.findtext("{*}NumCols"))
                    * int(support_array_node.findtext("{*}BytesPerElement"))
                ),
            )
        prev_end = 0
        sorted_arrays = sorted(support_array_offset_size.items(), key=lambda x: x[1])
        for array_id, (offset, size) in sorted_arrays:
            with self.need(f"Support array {array_id} starts at offset {prev_end}"):
                assert offset == prev_end
            prev_end = offset + size
        with self.precondition():
            assert self.kvp_list is not None
            with self.need(
                f"SUPPORT_BLOCK_SIZE matches the end of the last support array {sorted_arrays[-1][0]}"
            ):
                assert prev_end == int(self.kvp_list["SUPPORT_BLOCK_SIZE"])

    def check_ppp_block_size_and_packing(self):
        """PPP block is correctly sized and packed"""
        with self.precondition():
            assert self.crsdroot.find("{*}Data/{*}Transmit") is not None
            ppp_array_offset_size = {}

            num_bytes_ppp = int(
                self.crsdroot.findtext("{*}Data/{*}Transmit/{*}NumBytesPPP")
            )
            for ppp_array_node in self.crsdroot.findall(
                "{*}Data/{*}Transmit/{*}TxSequence"
            ):
                array_id = ppp_array_node.findtext("{*}Identifier")

                ppp_array_offset_size[array_id] = (
                    int(ppp_array_node.findtext("{*}PPPArrayByteOffset")),
                    (int(ppp_array_node.findtext("{*}NumPulses")) * num_bytes_ppp),
                )
            prev_end = 0
            sorted_arrays = sorted(ppp_array_offset_size.items(), key=lambda x: x[1])
            for array_id, (offset, size) in sorted_arrays:
                with self.need(f"PPP array {array_id} starts at offset {prev_end}"):
                    assert offset == prev_end
                prev_end = offset + size
            with self.precondition():
                assert self.kvp_list is not None
                with self.need(
                    f"PPP_BLOCK_SIZE matches the end of the last PPP array {sorted_arrays[-1][0]}"
                ):
                    assert prev_end == int(self.kvp_list["PPP_BLOCK_SIZE"])

    def check_pvp_block_size_and_packing(self):
        """PVP block is correctly sized and packed"""
        with self.precondition():
            assert self.crsdroot.find("{*}Data/{*}Receive") is not None
            pvp_array_offset_size = {}

            num_bytes_pvp = int(
                self.crsdroot.findtext("{*}Data/{*}Receive/{*}NumBytesPVP")
            )
            for channel_node in self.crsdroot.findall("{*}Data/{*}Receive/{*}Channel"):
                array_id = channel_node.findtext("{*}Identifier")

                pvp_array_offset_size[array_id] = (
                    int(channel_node.findtext("{*}PVPArrayByteOffset")),
                    (int(channel_node.findtext("{*}NumVectors")) * num_bytes_pvp),
                )
            prev_end = 0
            sorted_arrays = sorted(pvp_array_offset_size.items(), key=lambda x: x[1])
            for array_id, (offset, size) in sorted_arrays:
                with self.need(f"PVP array {array_id} starts at offset {prev_end}"):
                    assert offset == prev_end
                prev_end = offset + size
            with self.precondition():
                assert self.kvp_list is not None
                with self.need(
                    f"PVP_BLOCK_SIZE matches the end of the last PVP array {sorted_arrays[-1][0]}"
                ):
                    assert prev_end == int(self.kvp_list["PVP_BLOCK_SIZE"])

    def check_signal_block_size_and_packing(self):
        """Signal block is correctly sized and packed"""
        with self.precondition():
            assert self.crsdroot.find("{*}Data/{*}Receive") is not None
            assert self.crsdroot.find("{*}Data/{*}Receive/{*}SignalCompression") is None
            signal_array_offset_size = {}

            signal_dtype_str = self.crsdroot.findtext(
                "{*}Data/{*}Receive/{*}SignalArrayFormat"
            )
            signal_dtype = skcrsd.binary_format_string_to_dtype(signal_dtype_str)
            num_bytes_samp = signal_dtype.itemsize
            for channel_node in self.crsdroot.findall("{*}Data/{*}Receive/{*}Channel"):
                array_id = channel_node.findtext("{*}Identifier")

                signal_array_offset_size[array_id] = (
                    int(channel_node.findtext("{*}SignalArrayByteOffset")),
                    (
                        int(channel_node.findtext("{*}NumVectors"))
                        * int(channel_node.findtext("{*}NumSamples"))
                        * num_bytes_samp
                    ),
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

    def check_compressed_signal_block(self):
        """Metadata properly indicates signal arrays are stored in compressed format"""
        with self.precondition():
            compressed_size_str = self.crsdroot.findtext(
                "{*}Data/{*}Receive/{*}SignalCompression/{*}CompressedSignalSize"
            )
            assert compressed_size_str is not None
            for rcv_chan in self.crsdroot.findall("{*}Data/{*}Receive/{*}Channel"):
                with self.need(
                    f"SignalArrayByteOffset is 0 for ChId={rcv_chan.findtext('{*}Identifier')}"
                ):
                    assert int(rcv_chan.findtext("{*}SignalArrayByteOffset")) == 0
            with self.precondition():
                assert self.kvp_list is not None
                with self.need(
                    "SIGNAL_BLOCK_SIZE is set equal to CompressedSignalSize"
                ):
                    assert int(self.kvp_list["SIGNAL_BLOCK_SIZE"]) == int(
                        compressed_size_str
                    )

    def check_end_of_file_at_last_block(self):
        """Last block is at the end of the file."""
        with self.precondition():
            assert self.kvp_list is not None
            assert self.file is not None
            for block in reversed(BINARY_BLOCK_ORDER):
                if f"{block}_BLOCK_BYTE_OFFSET" in self.kvp_list:
                    self.file.seek(0, os.SEEK_END)
                    file_size = self.file.tell()
                    block_offset = int(self.kvp_list[f"{block}_BLOCK_BYTE_OFFSET"])
                    block_size = int(self.kvp_list[f"{block}_BLOCK_SIZE"])
                    with self.need(f"{block}_BLOCK is at the end of the file"):
                        assert file_size == block_offset + block_size
                    return

    def check_against_schema(self):
        """The XML matches the schema."""
        with self.need(
            f"Schema available for checking xml whose root tag = {self.crsdroot.tag}"
        ):
            assert self.schema is not None
            schema = etree.XMLSchema(file=str(self.schema))
            with self.need("XML passes schema"):
                assert schema.validate(self.crsdroot), schema.error_log

    def check_refgeom(self):
        """The ReferenceGeometry parameters are consistent with the other metadata"""
        with self.precondition():
            newroot = skcrsd.ElementWrapper(copy.deepcopy(self.crsdroot))
            ref_tx_id = self.crsdroot.findtext("{*}TxSequence/{*}RefTxId")
            ppps = None if ref_tx_id is None else self._get_sequence_ppps(ref_tx_id)
            ref_ch_id = self.crsdroot.findtext("{*}Channel/{*}RefChId")
            pvps = None if ref_ch_id is None else self._get_channel_pvps(ref_ch_id)
            dta = None
            if ref_ch_id is not None:
                dta_id = self.crsdroot.findtext(
                    f"{{*}}Channel/{{*}}Parameters[{{*}}Identifier='{ref_ch_id}']"
                    "/{*}SARImage/{*}DwellTimes/{*}Array/{*}DTAId"
                )
                dta = None if dta_id is None else self._get_support_array(dta_id)

            newroot["ReferenceGeometry"] = skcrsd.compute_reference_geometry(
                newroot.elem.getroottree(),
                pvps=pvps,
                ppps=ppps,
                dta=dta,
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

                        if issubclass(
                            np.asarray(expected_val).dtype.type, numbers.Number
                        ):
                            actual_val = con.Approx(actual_val, atol=1e-6, rtol=0)
                        with self.need(
                            f"{parent_key}/{key} matches defined calculation"
                        ):
                            assert np.all(expected_val == actual_val)

            _compare_children(
                skcrsd.ElementWrapper(self.crsdroot)["ReferenceGeometry"],
                newroot["ReferenceGeometry"],
                "ReferenceGeometry",
            )

    def assert_iac_matches_ecf(self, iac_coord, ecf_coord, tol=1.0):
        """Asserts that the IAC and ECF coordinates are a matched set"""
        assert np.linalg.norm(ecf_coord - self.iac_to_ecf(iac_coord)) < tol

    def iac_to_ecf(self, iac_coord):
        """Converts ImageAreaCoordinates to ECF"""
        if (
            self.crsdroot.find("{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar")
            is not None
        ):
            iarp = self.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}ECF")
            uiax = self.xmlhelp.load(
                "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"
            )
            uiay = self.xmlhelp.load(
                "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"
            )
            return iarp + uiax * iac_coord[0] + uiay * iac_coord[1]
        else:
            iarp_llh = self.xmlhelp.load("{*}SceneCoordinates/{*}IARP/{*}LLH")
            uiax_ll = self.xmlhelp.load(
                "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAXLL"
            )
            uiay_ll = self.xmlhelp.load(
                "{*}SceneCoordinates/{*}ReferenceSurface/{*}HAE/{*}uIAYLL"
            )
            llh_coord = iarp_llh.copy()
            llh_coord[:2] += uiax_ll * iac_coord[0] + uiay_ll * iac_coord[1]
            return sarkit.wgs84.geodetic_to_cartesian(llh_coord)
