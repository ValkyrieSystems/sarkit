import pathlib

import lxml.etree
import numpy as np
import pytest

import sarkit.sicd.projection as sicdproj

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


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


def test_compute_composite_error_no_apo_mono():
    sicd_xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    projmeta = sicdproj.MetadataParams.from_xml(sicd_xmltree)
    assert projmeta.is_monostatic()
    proj_set_0 = sicdproj.compute_projection_sets(projmeta, [0, 0])
    sens_mat = sicdproj.compute_sensitivity_matrices(projmeta)

    # No error stats
    c_rgaz = sicdproj.compute_composite_error_no_apo_mono(
        proj_set_0,
        sens_mat,
        sicdproj._params.ErrorStatParams(),
    )
    assert c_rgaz is None

    # No component
    c_rgaz = sicdproj.compute_composite_error_no_apo_mono(
        proj_set_0,
        sens_mat,
        sicdproj._params.ErrorStatParams(
            C_SCP_RGAZ=np.eye(2),
        ),
    )
    assert c_rgaz is not None

    # Component
    c_rgaz = sicdproj.compute_composite_error_no_apo_mono(
        proj_set_0,
        sens_mat,
        sicdproj._params.ErrorStatParams(
            C_AIF_APV=np.eye(6),
            AIF="RICF",
            VAR_RB=1.0,
            VAR_CLK_SF=1.1,
            VAR_TROP=1.2,
            VAR_IONO=1.3,
            C_UI=np.eye(2),
        ),
    )
    assert c_rgaz is not None


def test_compute_composite_error_apo_mono():
    sicd_xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    projmeta = sicdproj.MetadataParams.from_xml(sicd_xmltree)
    assert projmeta.is_monostatic()
    sens_mat = sicdproj.compute_sensitivity_matrices(projmeta)

    # No error stats
    c_rgaz = sicdproj.compute_composite_error_apo_mono(
        sens_mat,
        sicdproj._params.ApoErrorParams(),
    )
    assert c_rgaz is None

    # No component
    c_rgaz = sicdproj.compute_composite_error_apo_mono(
        sens_mat,
        sicdproj.ApoErrorParams(
            C_SCPAPO_RGAZ=np.eye(2),
        ),
    )
    assert c_rgaz is not None

    # Component
    c_rgaz = sicdproj.compute_composite_error_apo_mono(
        sens_mat,
        sicdproj.ApoErrorParams(
            C_SCPAPO_RGAZ=np.eye(2),
            C_APOM=np.eye(8),
        ),
    )
    assert c_rgaz is not None
