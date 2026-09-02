import asyncio
import builtins
import contextlib
import copy
import queue
import threading

import lxml.etree
import numpy as np
from aiohttp import web

import sarkit.crsd as skcrsd
import sarkit.wgs84


# Python's built in http.server does not support the Range header.  aiohttp does
def _run_aiohttp_server(app, loop, ready_event, stop_event, msg_queue):
    asyncio.set_event_loop(loop)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 8080)
    loop.run_until_complete(site.start())

    ready_event.set()
    msg_queue.put(site.name)
    # Wait for the stop event to be set
    loop.run_until_complete(stop_event.wait())

    # Cleanup when stop event is set
    loop.run_until_complete(runner.cleanup())
    loop.close()


@contextlib.contextmanager
def static_http_server(static_dir):
    ready_event = threading.Event()
    stop_event = asyncio.Event()
    msg_queue = queue.Queue()

    app = web.Application()
    app.add_routes([web.static("/", static_dir)])

    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=_run_aiohttp_server,
        args=(app, loop, ready_event, stop_event, msg_queue),
        daemon=True,
    )
    thread.start()

    if ready_event.wait(timeout=10):
        url = msg_queue.get()
        yield url

    loop.call_soon_threadsafe(stop_event.set)
    thread.join()


def simple_open_read(filename, *args, **kwargs):
    """Open a file, returning a file-like object with some methods supported"""

    class _SimpleFile:
        def __init__(self, filename):
            self._file = builtins.open(filename, "rb")

            self.read = self._file.read
            self.readinto = self._file.readinto
            self.readline = self._file.readline
            self.seek = self._file.seek
            self.tell = self._file.tell
            self.close = self._file.close

        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, *args, **kwargs):
            return self.close()

    return _SimpleFile(filename)


def replace_planar_with_hae(root_ew):
    """Given an ElementWrapper of a CRSD/CPHD root, replace Planar ReferenceSurface with HAE"""
    sc_ew = root_ew["SceneCoordinates"]
    uiax = sc_ew["ReferenceSurface"]["Planar"]["uIAX"]
    uiay = sc_ew["ReferenceSurface"]["Planar"]["uIAY"]
    iarp_ecf = sc_ew["IARP"]["ECF"]
    iarp_llh = sc_ew["IARP"]["LLH"]
    sc_ew["ReferenceSurface"]["HAE"]["uIAXLL"] = np.deg2rad(
        (sarkit.wgs84.cartesian_to_geodetic(iarp_ecf + uiax) - iarp_llh)[:2]
    )
    sc_ew["ReferenceSurface"]["HAE"]["uIAYLL"] = np.deg2rad(
        (sarkit.wgs84.cartesian_to_geodetic(iarp_ecf + uiay) - iarp_llh)[:2]
    )
    del sc_ew["ReferenceSurface"]["Planar"]


def _remove(root, pattern):
    if (elem := root.find(pattern)) is not None:
        elem.getparent().remove(elem)
    else:
        print(f"Cannot find {pattern=}")


def _replace_error(crsd_etree, sensor_type):
    sar_error = crsd_etree.find("{*}ErrorParameters/{*}SARImage")
    elem_ns = lxml.etree.QName(sar_error).namespace
    retval = copy.deepcopy(sar_error.find("{*}Monostatic"))
    retval.tag = f"{{{elem_ns}}}{sensor_type}Sensor"
    sar_error.addnext(retval)
    helper = skcrsd.XmlHelper(crsd_etree)
    ndx = {"Tx": 0, "Rcv": 1}[sensor_type]
    helper.set_elem(
        retval.find(".//{*}TimeFreqCov"),
        skcrsd.MtxType((3, 3)).parse_elem(retval.find(".//{*}TimeFreqCov"))[
            [ndx, 2], :
        ][:, [ndx, 2]],
    )
    time_decorr = copy.deepcopy(retval.find(f".//{{*}}{{{sensor_type}}}TimeDecorr"))
    if time_decorr is not None:
        time_decorr.tag = f"{{{elem_ns}}}TimeDecorr"
        _remove(retval, "{*}TxTimeDecorr")
        _remove(retval, "{*}RcvTimeDecorr")
        retval.find(".//{*}ClockFreqDecorr").addprevious(time_decorr)
    sar_error.getparent().remove(sar_error)


def _repack_support_arrays(crsd_etree):
    offset = 0
    for array in crsd_etree.findall("{*}Data/{*}Support/{*}SupportArray"):
        array.find("{*}ArrayByteOffset").text = str(offset)
        offset += (
            int(array.findtext("{*}NumRows"))
            * int(array.findtext("{*}NumCols"))
            * int(array.findtext("{*}BytesPerElement"))
        )
    return offset


def crsdsar_xml_to_crsdtx(crsd_etree: lxml.etree.ElementTree) -> None:
    """Modify a CRSDsar ElementTree into a CRSDtx ElementTree in place."""
    crsd_etree.find(".//{*}RefPulseIndex").text = crsd_etree.find(
        ".//{*}RefVectorPulseIndex"
    ).text
    ns = lxml.etree.QName(crsd_etree.getroot()).namespace
    crsd_etree.getroot().tag = f"{{{ns}}}CRSDtx"
    _remove(crsd_etree, "{*}SARInfo")
    _remove(crsd_etree, "{*}ReceiveInfo")
    _remove(crsd_etree, "{*}Global/{*}Receive")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ExtendedArea")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ImageGrid")
    _remove(crsd_etree, "{*}Data/{*}Receive")
    _remove(crsd_etree, "{*}Channel")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}SARImage")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}RcvParameters")
    _remove(crsd_etree, "{*}DwellPolynomials")
    _remove(crsd_etree, "{*}PVP")
    _replace_error(crsd_etree, "Tx")


def crsdsar_xml_to_crsdrcv(crsd_etree: lxml.etree.ElementTree) -> None:
    """Modify a CRSDsar ElementTree into a CRSDtx ElementTree in place."""
    ns = lxml.etree.QName(crsd_etree.getroot()).namespace
    crsd_etree.getroot().tag = f"{{{ns}}}CRSDrcv"
    _remove(crsd_etree, "{*}SARInfo")
    _remove(crsd_etree, "{*}TransmitInfo")
    _remove(crsd_etree, "{*}Global/{*}Transmit")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ExtendedArea")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ImageGrid")
    _remove(crsd_etree, "{*}Data/{*}Transmit")
    _remove(crsd_etree, "{*}TxSequence")
    _remove(crsd_etree, "{*}Channel/{*}Parameters/{*}SARImage")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}SARImage")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}TxParameters")
    _remove(crsd_etree, "{*}DwellPolynomials")
    fx_ids = [
        x.text
        for x in crsd_etree.findall("{*}SupportArray/{*}FxResponseArray/{*}Identifier")
    ]
    xm_ids = [
        x.text for x in crsd_etree.findall("{*}SupportArray/{*}XMArray/{*}Identifier")
    ]
    _remove(crsd_etree, "{*}SupportArray/{*}FxResponseArray")
    _remove(crsd_etree, "{*}SupportArray/{*}XMArray")
    for x in fx_ids + xm_ids:
        _remove(
            crsd_etree,
            f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{x}']",
        )
    nsa = crsd_etree.find("{*}Data/{*}Support/{*}NumSupportArrays")
    nsa.text = str(int(nsa.text) - len(fx_ids + xm_ids))
    _repack_support_arrays(crsd_etree)
    _remove(crsd_etree, "{*}PPP")
    tx_pulse_index_offset = int(crsd_etree.findtext("{*}PVP/{*}TxPulseIndex/{*}Offset"))
    _remove(crsd_etree, "{*}PVP/{*}TxPulseIndex")
    for pvp_offset in crsd_etree.findall("{*}PVP/*/{*}Offset"):
        if int(pvp_offset.text) > tx_pulse_index_offset:
            pvp_offset.text = str(int(pvp_offset.text) - 1)
    crsd_etree.find("{*}Data/{*}Receive/{*}NumBytesPVP").text = str(
        int(crsd_etree.findtext("{*}Data/{*}Receive/{*}NumBytesPVP")) - 8
    )
    _replace_error(crsd_etree, "Rcv")
