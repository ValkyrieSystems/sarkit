import numpy as np
import pytest

import sarkit.sicd.projection as sicdproj


def test_compute_ric_basis_vectors():
    for uvec in sicdproj.compute_ric_basis_vectors([1, 2, 3], [4, 5, 6]):
        assert uvec.shape == (3,)
        assert np.linalg.norm(uvec) == pytest.approx(1.0)


@pytest.mark.parametrize("frame", ("ECF", "RICF", "RICI"))
def test_compute_ecef_pv_transformation(frame):
    t = sicdproj.compute_ecef_pv_transformation([1, 2, 3], [4, 5, 6], frame)
    if frame != "RICI":
        assert t @ t.T == pytest.approx(np.eye(6))
    assert t[:3, :3] @ t[:3, :3].T == pytest.approx(np.eye(3))
