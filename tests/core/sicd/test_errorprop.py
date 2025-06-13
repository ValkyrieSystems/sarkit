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
        sicdproj.ErrorStatParams(),
    )
    assert c_rgaz is None

    # No component
    c_rgaz = sicdproj.compute_composite_error_no_apo_mono(
        proj_set_0,
        sens_mat,
        sicdproj.ErrorStatParams(
            C_SCP_RGAZ=np.eye(2),
        ),
    )
    assert c_rgaz is not None

    # Component
    c_rgaz = sicdproj.compute_composite_error_no_apo_mono(
        proj_set_0,
        sens_mat,
        sicdproj.ErrorStatParams(
            component_mono=sicdproj.ComponentErrorStatMono(
                C_AIF_APV=np.eye(6),
                AIF="RICF",
                VAR_RB=1.0,
                VAR_CLK_SF=1.1,
                VAR_TROP=1.2,
                VAR_IONO=1.3,
            ),
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
        sicdproj.ApoErrorParams(),
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


def test_compute_composite_error_no_apo_bi():
    sicd_xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.4.0.xml")
    projmeta = sicdproj.MetadataParams.from_xml(sicd_xmltree)
    assert projmeta.is_bistatic()
    proj_set_0 = sicdproj.compute_projection_sets(projmeta, [0, 0])
    sens_mat = sicdproj.compute_sensitivity_matrices(projmeta)

    # No error stats
    c_rgaz = sicdproj.compute_composite_error_no_apo_bi(
        proj_set_0,
        sens_mat,
        sicdproj.ErrorStatParams(),
        projmeta.t_SCP_COA,
    )
    assert c_rgaz is None

    # No component
    c_rgaz = sicdproj.compute_composite_error_no_apo_bi(
        proj_set_0,
        sens_mat,
        sicdproj.ErrorStatParams(
            C_SCP_RRdot=np.eye(2),
        ),
        projmeta.t_SCP_COA,
    )
    assert c_rgaz is not None

    # Component
    c_rgaz = sicdproj.compute_composite_error_no_apo_bi(
        proj_set_0,
        sens_mat,
        sicdproj.ErrorStatParams(
            component_bi=sicdproj.ComponentErrorStatBi(
                C_XIF_XPV=np.eye(6),
                XIF="RICF",
                C_RIF_RPV=np.eye(6),
                RIF="RICI",
                CC_XIF_RIF_XPV_RPV=np.eye(6),
                C_XRTF=np.eye(4),
                C_ATM=np.eye(2),
            ),
            C_UI=np.eye(2),
        ),
        projmeta.t_SCP_COA,
    )
    assert c_rgaz is not None


def test_compute_composite_error_apo_bi():
    sicd_xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.4.0.xml")
    projmeta = sicdproj.MetadataParams.from_xml(sicd_xmltree)
    assert projmeta.is_bistatic()
    proj_set_0 = sicdproj.compute_projection_sets(projmeta, [0, 0])
    sens_mat = sicdproj.compute_sensitivity_matrices(projmeta)

    # No error stats
    c_rgaz = sicdproj.compute_composite_error_apo_bi(
        proj_set_0,
        sens_mat,
        sicdproj.ApoErrorParams(),
        projmeta.t_SCP_COA,
    )
    assert c_rgaz is None

    # No component
    c_rgaz = sicdproj.compute_composite_error_apo_bi(
        proj_set_0,
        sens_mat,
        sicdproj.ApoErrorParams(
            C_SCPAPO_RRdot=np.eye(2),
        ),
        projmeta.t_SCP_COA,
    )
    assert c_rgaz is not None

    # Component
    c_rgaz = sicdproj.compute_composite_error_apo_bi(
        proj_set_0,
        sens_mat,
        sicdproj.ApoErrorParams(
            C_APOXR=np.eye(16),
        ),
        projmeta.t_SCP_COA,
    )
    assert c_rgaz is not None
