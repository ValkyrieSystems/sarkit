"""Calculations related to the CPHD signal model."""

import numpy as np
import numpy.typing as npt


def compute_t_ref(
    xmt: npt.ArrayLike,
    rcv: npt.ArrayLike,
    srp: npt.ArrayLike,
    txc: npt.ArrayLike,
    trc: npt.ArrayLike,
) -> np.ndarray:
    """Compute the reference times for the given collection geometry parameters.

    Parameters
    ----------
    xmt : (..., 3) array_like
        Transmit APC positions at times ``txc`` in ECF coordinates with X, Y, Z components in meters in the last
        dimension.
    rcv : (..., 3) array_like
        Receive APC positions at times ``trc`` in ECF coordinates with X, Y, Z components in meters in the last
        dimension.
    srp : (..., 3) array_like
        Stabilization Reference Point positions in ECF coordinates with X, Y, Z components in meters in the last
        dimension.
    txc : (...) array_like
        Time the center of the pulse is at the Transmit APC in seconds.
    trc : (...) array_like
        Receive time for the center of the echo from the SRP relative to Collection Start Time in seconds.

    Returns
    -------
    t_ref : (...) ndarray
        Reference times

    See Also
    --------
    compute_t_ref_from_pvps
    """
    r_xmt = np.linalg.norm(np.asarray(xmt) - np.asarray(srp), axis=-1)
    r_rcv = np.linalg.norm(np.asarray(rcv) - np.asarray(srp), axis=-1)
    return np.asarray(txc) + r_xmt / (r_xmt + r_rcv) * (
        np.asarray(trc) - np.asarray(txc)
    )


def compute_t_ref_from_pvps(pvp_array: np.ndarray) -> np.ndarray:
    """Compute the reference times for each vector in a PVP array.

    Parameters
    ----------
    pvp_array : ndarray
        Array of PVPs

    Returns
    -------
    t_ref : (...) ndarray
        Reference times

    See Also
    --------
    compute_t_ref
    """
    return compute_t_ref(
        pvp_array["TxPos"],
        pvp_array["RcvPos"],
        pvp_array["SRPPos"],
        pvp_array["TxTime"],
        pvp_array["RcvTime"],
    )
