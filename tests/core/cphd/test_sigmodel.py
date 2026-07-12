import numpy as np

import sarkit.cphd as skcphd


def test_compute_t_ref():
    rng = np.random.default_rng()
    xmt = [10, 20, 30]
    rcv = 100 * rng.random((6, 5, 4, 3)) - 50
    srp = [0, 0, 0]
    txc = rng.random((6, 1, 4))
    trc = txc + 0.1 + rng.random((6, 1, 4))
    t_ref = skcphd.compute_t_ref(
        xmt,
        rcv,
        srp,
        txc,
        trc,
    )
    assert (txc < t_ref).all()
    assert (trc > t_ref).all()

    # move xmt farther away, then t_ref should be delayed
    t_ref2 = skcphd.compute_t_ref(
        2 * np.array(xmt),
        rcv,
        srp,
        txc,
        trc,
    )
    assert (t_ref < t_ref2).all()
    assert (txc < t_ref2).all()
    assert (trc > t_ref2).all()

    # move rcv farther away, then t_ref should be advanced
    t_ref3 = skcphd.compute_t_ref(
        xmt,
        2 * rcv,
        srp,
        txc,
        trc,
    )
    assert (t_ref > t_ref3).all()
    assert (txc < t_ref3).all()
    assert (trc > t_ref3).all()


def test_compute_t_ref_from_pvps(example_cphd):
    with example_cphd.open("rb") as f, skcphd.Reader(f) as r:
        pvps = r.read_pvps(r.metadata.xmltree.findtext(".//{*}RefChId"))
    t_ref = skcphd.compute_t_ref_from_pvps(pvps)
    assert t_ref.size == pvps.size
