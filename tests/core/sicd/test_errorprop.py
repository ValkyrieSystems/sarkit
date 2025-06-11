import numpy as np
import pytest

import sarkit.sicd.projection as sicdproj


def test_compute_ric_basis_vectors():
    for uvec in sicdproj.compute_ric_basis_vectors([1, 2, 3], [4, 5, 6]):
        assert uvec.shape == (3,)
        assert np.linalg.norm(uvec) == pytest.approx(1.0)


@pytest.mark.parametrize("frame", ("ECF", "RICF", "RICI"))
def test_compute_ecef_pv_covariance(frame):
    a = np.random.default_rng().random((6, 6))
    c_pv = a @ a.T
    c_pv_ecef = sicdproj.compute_ecef_pv_covariance(c_pv, [1, 2, 3], [4, 5, 6], frame)
    if frame != "RICI":
        assert np.linalg.eigvalsh(c_pv) == pytest.approx(np.linalg.eigvalsh(c_pv_ecef))
    pos_eigvals = np.linalg.eigvalsh(c_pv[:3, :3])
    pos_ecef_eigvals = np.linalg.eigvalsh(c_pv_ecef[:3, :3])
    assert pos_eigvals == pytest.approx(pos_ecef_eigvals)
