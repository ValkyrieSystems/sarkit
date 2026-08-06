import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt

from . import _xml as skcphd_xml


def compute_dwelltimes_using_poly(
    ch_id: str,
    iax: npt.ArrayLike,
    iay: npt.ArrayLike,
    cphd_xmltree: lxml.etree.ElementTree,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute center of dwell times and dwell times for scene points using polynomials.

    Parameters
    ----------
    ch_id : str
        Channel unique identifier
    iax, iay : array_like
        Image area coordinates (in meters) of the scene points for which to compute the dwell times
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML

    Returns
    -------
    t_cod : ndarray
        Center of dwell times (sec) for the scene points relative to the CollectionStart time
    t_dwell : ndarray
        Dwell times (sec) for which the channel signal array contains the echo signals from the scene points
    """
    iax, iay = np.broadcast_arrays(iax, iay)

    ew = skcphd_xml.ElementWrapper(cphd_xmltree.getroot())
    chan_dt = ew["Channel"].find("Parameters", Identifier=ch_id)["DwellTimes"]
    cod_poly = ew["Dwell"].find("CODTime", Identifier=chan_dt["CODId"])["CODTimePoly"]
    dwell_poly = ew["Dwell"].find("DwellTime", Identifier=chan_dt["DwellId"])[
        "DwellTimePoly"
    ]
    t_cod = npp.polyval2d(iax, iay, cod_poly)
    t_dwell = npp.polyval2d(iax, iay, dwell_poly)
    return t_cod, t_dwell
