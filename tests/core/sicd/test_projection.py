import copy
import dataclasses
import functools
import pathlib

import lxml.etree
import numpy as np
import pytest

import sarkit.sicd.projection as sicdproj
import sarkit.sicd.projection._sensitivity
import sarkit.wgs84

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


@pytest.fixture
def example_proj_metadata():
    etree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    return sicdproj.MetadataParams.from_xml(etree)


@pytest.fixture
def example_proj_metadata_bi():
    etree = lxml.etree.parse(DATAPATH / "example-sicd-1.4.0.xml")
    proj_metadata = sicdproj.MetadataParams.from_xml(etree)
    assert not proj_metadata.is_monostatic()
    return proj_metadata


@pytest.fixture(
    params=[DATAPATH / "example-sicd-1.3.0.xml", DATAPATH / "example-sicd-1.4.0.xml"]
)
def mono_and_bi_proj_metadata(request):
    etree = lxml.etree.parse(request.param)
    return sicdproj.MetadataParams.from_xml(etree)


@pytest.fixture(params=[(3, 4, 5, 2), (2,), (1, 2), (2, 2)])
def image_grid_locations(request):
    return np.random.default_rng(12345).uniform(size=request.param)


def test_metadata_params():
    all_attrs = set()
    set_attrs = set()
    for xml_file in (DATAPATH / "syntax_only/sicd").glob("*.xml"):
        etree = lxml.etree.parse(xml_file)
        proj_metadata = sicdproj.MetadataParams.from_xml(etree)
        pm_dict = dataclasses.asdict(proj_metadata)
        all_attrs.update(pm_dict.keys())
        set_attrs.update(k for k, v in pm_dict.items() if v is not None)
    unset_attrs = all_attrs - set_attrs
    assert not unset_attrs


def test_metadata_params_without_optional_apcs():
    etree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    assert sicdproj.MetadataParams.from_xml(etree).Rcv_Poly is not None
    for el in etree.findall(".//{*}RcvAPCIndex") + etree.findall(".//{*}RcvApcPoly"):
        el.getparent().remove(el)
    assert sicdproj.MetadataParams.from_xml(etree).Rcv_Poly is None


def test_metadata_params_is_monostatic(example_proj_metadata):
    example_proj_metadata.Collect_Type = "MONOSTATIC"
    assert example_proj_metadata.is_monostatic()

    example_proj_metadata.Collect_Type = "BISTATIC"
    assert not example_proj_metadata.is_monostatic()

    example_proj_metadata.Collect_Type = "NOT_A_REAL_COLLECT_TYPE"
    with pytest.raises(ValueError, match="must be MONOSTATIC or BISTATIC"):
        example_proj_metadata.is_monostatic()


def test_image_plane_parameters_roundtrip(example_proj_metadata):
    image_grid_locations = np.random.default_rng(12345).uniform(
        low=-24, high=24, size=(3, 4, 5, 2)
    )
    image_plane_points = sicdproj.image_grid_to_image_plane_point(
        example_proj_metadata.SCP,
        example_proj_metadata.uRow,
        example_proj_metadata.uCol,
        image_grid_locations,
    )
    re_image_grid_locations = sicdproj.image_plane_point_to_image_grid(
        example_proj_metadata.SCP,
        example_proj_metadata.uRow,
        example_proj_metadata.uCol,
        image_plane_points,
    )
    assert image_grid_locations == pytest.approx(re_image_grid_locations)


def test_compute_coa_time(example_proj_metadata):
    assert sicdproj.compute_coa_time(
        example_proj_metadata.cT_COA, [0, 0]
    ) == pytest.approx(example_proj_metadata.t_SCP_COA)


def test_compute_coa_pos_vel_mono(example_proj_metadata):
    assert example_proj_metadata.is_monostatic()
    computed_pos_vel = sicdproj.compute_coa_pos_vel(
        example_proj_metadata, example_proj_metadata.t_SCP_COA
    )
    assert computed_pos_vel.ARP_COA == pytest.approx(example_proj_metadata.ARP_SCP_COA)
    assert computed_pos_vel.VARP_COA == pytest.approx(
        example_proj_metadata.VARP_SCP_COA
    )


def test_compute_coa_pos_vel_bi(example_proj_metadata_bi):
    computed_pos_vel = sicdproj.compute_coa_pos_vel(
        example_proj_metadata_bi, example_proj_metadata_bi.t_SCP_COA
    )
    assert computed_pos_vel.GRP_COA == pytest.approx(example_proj_metadata_bi.SCP)
    assert computed_pos_vel.tx_COA == pytest.approx(example_proj_metadata_bi.tx_SCP_COA)
    assert computed_pos_vel.tr_COA == pytest.approx(example_proj_metadata_bi.tr_SCP_COA)
    assert computed_pos_vel.Xmt_COA == pytest.approx(
        example_proj_metadata_bi.Xmt_SCP_COA
    )
    assert computed_pos_vel.VXmt_COA == pytest.approx(
        example_proj_metadata_bi.VXmt_SCP_COA
    )
    assert computed_pos_vel.Rcv_COA == pytest.approx(
        example_proj_metadata_bi.Rcv_SCP_COA
    )
    assert computed_pos_vel.VRcv_COA == pytest.approx(
        example_proj_metadata_bi.VRcv_SCP_COA
    )


def test_scp_projection_set_mono(example_proj_metadata):
    assert example_proj_metadata.is_monostatic()
    r_scp_coa, rdot_scp_coa = sicdproj.compute_scp_coa_r_rdot(example_proj_metadata)
    scp_proj_set = sicdproj.compute_projection_sets(example_proj_metadata, [0, 0])
    assert scp_proj_set.t_COA == pytest.approx(example_proj_metadata.t_SCP_COA)
    assert scp_proj_set.ARP_COA == pytest.approx(example_proj_metadata.ARP_SCP_COA)
    assert scp_proj_set.VARP_COA == pytest.approx(example_proj_metadata.VARP_SCP_COA)
    assert scp_proj_set.R_COA == pytest.approx(r_scp_coa)
    assert scp_proj_set.Rdot_COA == pytest.approx(rdot_scp_coa)


def test_scp_projection_set_bi(example_proj_metadata_bi):
    assert not example_proj_metadata_bi.is_monostatic()
    r_scp_coa, rdot_scp_coa = sicdproj.compute_scp_coa_r_rdot(example_proj_metadata_bi)
    scp_proj_set = sicdproj.compute_projection_sets(example_proj_metadata_bi, [0, 0])
    assert scp_proj_set.t_COA == pytest.approx(example_proj_metadata_bi.t_SCP_COA)
    assert scp_proj_set.tx_COA == pytest.approx(example_proj_metadata_bi.tx_SCP_COA)
    assert scp_proj_set.tr_COA == pytest.approx(example_proj_metadata_bi.tr_SCP_COA)
    assert scp_proj_set.Xmt_COA == pytest.approx(example_proj_metadata_bi.Xmt_SCP_COA)
    assert scp_proj_set.VXmt_COA == pytest.approx(example_proj_metadata_bi.VXmt_SCP_COA)
    assert scp_proj_set.Rcv_COA == pytest.approx(example_proj_metadata_bi.Rcv_SCP_COA)
    assert scp_proj_set.VRcv_COA == pytest.approx(example_proj_metadata_bi.VRcv_SCP_COA)
    assert scp_proj_set.R_Avg_COA == pytest.approx(r_scp_coa)
    assert scp_proj_set.Rdot_Avg_COA == pytest.approx(rdot_scp_coa)


@pytest.mark.parametrize(
    "pm_fixture_name", ("example_proj_metadata", "example_proj_metadata_bi")
)
def test_scp_coa_slant_plane_normal(pm_fixture_name, request):
    proj_metadata = request.getfixturevalue(pm_fixture_name)
    u_spn_scp_coa = sicdproj.compute_scp_coa_slant_plane_normal(proj_metadata)

    # unit vector
    assert np.linalg.norm(u_spn_scp_coa) == pytest.approx(1.0)

    # points away from earth
    assert np.linalg.norm(proj_metadata.SCP) < np.linalg.norm(
        proj_metadata.SCP + u_spn_scp_coa
    )


def test_compute_pt_r_rdot_parameters_mono(example_proj_metadata):
    """From Vol. 3:
    For a Monostatic Image: Input the COA ARP position and velocity for both COA APC
    positions and velocities. The resulting range and range rate will be the range and range rate of
    the ARP relative to the point PT.
    """

    pt_r_rdot_params = sicdproj.compute_pt_r_rdot_parameters(
        example_proj_metadata.LOOK,
        example_proj_metadata.ARP_SCP_COA,
        example_proj_metadata.VARP_SCP_COA,
        example_proj_metadata.ARP_SCP_COA,
        example_proj_metadata.VARP_SCP_COA,
        example_proj_metadata.SCP,
    )
    r_scp, rdot_scp = sicdproj.compute_scp_coa_r_rdot(example_proj_metadata)
    assert pt_r_rdot_params.R_Avg_PT == pytest.approx(r_scp)
    assert pt_r_rdot_params.Rdot_Avg_PT == pytest.approx(rdot_scp)


def test_r_rdot_to_ground_plane(example_proj_metadata):
    im_coords = np.random.default_rng(12345).uniform(
        low=-24.0, high=24.0, size=(3, 4, 5, 2)
    )
    proj_sets_mono = sicdproj.compute_projection_sets(example_proj_metadata, im_coords)
    scp_spn = sicdproj.compute_scp_coa_slant_plane_normal(example_proj_metadata)
    gpp_tgt_mono = sicdproj.r_rdot_to_ground_plane_mono(
        example_proj_metadata.LOOK,
        proj_sets_mono,
        example_proj_metadata.SCP,
        scp_spn,
    )

    # Per Volume 3: The bistatic function defined may also be used for a monostatic image.
    gpp_tgt_bi, delta_gp, success = sicdproj.r_rdot_to_ground_plane_bi(
        example_proj_metadata.LOOK,
        example_proj_metadata.SCP,
        sicdproj.ProjectionSetsBi(
            t_COA=proj_sets_mono.t_COA,
            tx_COA=proj_sets_mono.t_COA,
            tr_COA=proj_sets_mono.t_COA,
            Xmt_COA=proj_sets_mono.ARP_COA,
            VXmt_COA=proj_sets_mono.VARP_COA,
            Rcv_COA=proj_sets_mono.ARP_COA,
            VRcv_COA=proj_sets_mono.VARP_COA,
            R_Avg_COA=proj_sets_mono.R_COA,
            Rdot_Avg_COA=proj_sets_mono.Rdot_COA,
        ),
        example_proj_metadata.SCP,
        scp_spn,
    )
    assert gpp_tgt_mono == pytest.approx(gpp_tgt_bi)
    assert np.isfinite(delta_gp).all()
    assert success


@pytest.mark.parametrize("scalar_hae", (True, False))
@pytest.mark.parametrize(
    "mdata_name", ("example_proj_metadata", "example_proj_metadata_bi")
)
def test_r_rdot_to_hae_surface(mdata_name, scalar_hae, request):
    proj_metadata = request.getfixturevalue(mdata_name)
    rng = np.random.default_rng(12345)
    im_coords = rng.uniform(low=-24.0, high=24.0, size=(3, 4, 5, 2))
    hae0 = proj_metadata.SCP_HAE
    if not scalar_hae:
        hae0 += rng.uniform(low=-24.0, high=24.0, size=im_coords.shape[:-1])
    proj_sets = sicdproj.compute_projection_sets(proj_metadata, im_coords)
    spp_tgt, _, success = sicdproj.r_rdot_to_constant_hae_surface(
        proj_metadata.LOOK,
        proj_metadata.SCP,
        proj_sets,
        hae0,
    )
    assert success
    spp_llh = sarkit.wgs84.cartesian_to_geodetic(spp_tgt)
    assert spp_llh[..., 2] == pytest.approx(hae0, abs=1e-6)

    bad_index = (1, 2, 3)
    bad_proj_sets = copy.deepcopy(proj_sets)
    if proj_metadata.is_monostatic():
        bad_proj_sets.R_COA[bad_index] *= 1e6
    else:
        bad_proj_sets.R_Avg_COA[bad_index] *= 1e6

    spp_tgt_w_bad, _, success = sicdproj.r_rdot_to_constant_hae_surface(
        proj_metadata.LOOK,
        proj_metadata.SCP,
        bad_proj_sets,
        hae0,
    )
    assert not success
    mismatched_index = np.argwhere((spp_tgt != spp_tgt_w_bad).any(axis=-1)).squeeze()
    assert np.array_equal(bad_index, mismatched_index)


@pytest.mark.parametrize(
    "mdata_name",
    (
        "example_proj_metadata",
        pytest.param(
            "example_proj_metadata_bi",
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
    ),
)
def test_r_rdot_to_dem_surface(mdata_name, request):
    proj_metadata = request.getfixturevalue(mdata_name)

    # SCP projection set intersects "DEM" that is a constant HAE=SCP_HAE surface at... SCP
    proj_set = sicdproj.compute_projection_sets(proj_metadata, [0, 0])
    s = sicdproj.r_rdot_to_dem_surface(
        proj_metadata.LOOK,
        proj_metadata.SCP,
        proj_set,
        ecef2dem_func=lambda x: sarkit.wgs84.cartesian_to_geodetic(x)[..., -1]
        - proj_metadata.SCP_HAE,
        hae_min=proj_metadata.SCP_HAE - 10.0,
        hae_max=proj_metadata.SCP_HAE + 10.0,
        delta_dist_dem=1.0,
    )
    assert np.allclose(s, proj_metadata.SCP, atol=0.01)

    # Make some faux "DEMs" that are sinusoids atop an HAE to test multiple-intersections case
    n_rrc = 24
    proj_sets = sicdproj.compute_projection_sets(proj_metadata, np.zeros((n_rrc, 2)))
    rrc, _, _ = sicdproj.r_rdot_to_constant_hae_surface(
        proj_metadata.LOOK,
        proj_metadata.SCP,
        proj_sets,
        proj_metadata.SCP_HAE + np.linspace(-10, 10, n_rrc),
    )
    rrc_lat, _, rrc_hae = sarkit.wgs84.cartesian_to_geodetic(rrc).T
    # np.interp inputs must be increasing
    ((min_lat, hae0), (max_lat, hae1)) = sorted(
        [(rrc_lat[0], rrc_hae[0]), (rrc_lat[-1], rrc_hae[-1])], key=lambda x: x[0]
    )

    def get_ripply_dem(freq=4.0):
        fake_lat_dem = functools.partial(
            np.interp,
            xp=np.linspace(min_lat, max_lat, num=1 << 10),
            fp=np.linspace(hae0, hae1, num=1 << 10)
            + np.sin(2 * np.pi * freq * np.linspace(0, 1, 1 << 10)),
        )

        def ecef2dem_func(ecef):
            llh = sarkit.wgs84.cartesian_to_geodetic(ecef)
            return llh[..., -1] - fake_lat_dem(llh[..., 0])

        return ecef2dem_func

    dem_funcs = {
        "is_rrc": get_ripply_dem(0),
        "some_ripples": get_ripply_dem(4.0),
        "more_ripples": get_ripply_dem(8.0),
        "too_low": lambda x: get_ripply_dem(0)(x) - 1000.0,
    }
    results = {}
    for label, func in dem_funcs.items():
        results[label] = sicdproj.r_rdot_to_dem_surface(
            proj_metadata.LOOK,
            proj_metadata.SCP,
            proj_set,
            ecef2dem_func=func,
            hae_min=proj_metadata.SCP_HAE - 20.0,
            hae_max=proj_metadata.SCP_HAE + 20.0,
            delta_dist_dem=1.0,
        )
        if label == "too_low":
            assert results[label].size == 0
        else:
            expected_r = (
                proj_set.R_COA if proj_metadata.is_monostatic() else proj_set.R_Avg_COA
            )
            assert np.linalg.norm(
                proj_metadata.ARP_SCP_COA - results[label], axis=-1
            ) == pytest.approx(float(expected_r))
    assert len(results["more_ripples"]) > len(results["some_ripples"])
    assert len(results["is_rrc"]) > 1


def _projection_sets_smoketest(mdata, gridlocs):
    proj_set = sicdproj.compute_projection_sets(mdata, gridlocs)
    if mdata.is_monostatic():
        assert np.all([proj_set.R_COA, proj_set.Rdot_COA])
        gpp_tgt = sicdproj.r_rdot_to_ground_plane_mono(
            mdata.LOOK,
            proj_set,
            mdata.SCP,
            sicdproj.compute_scp_coa_slant_plane_normal(mdata),
        )
    else:
        assert np.all([proj_set.R_Avg_COA, proj_set.Rdot_Avg_COA])
        gpp_tgt, _, _ = sicdproj.r_rdot_to_ground_plane_bi(
            mdata.LOOK,
            mdata.SCP,
            proj_set,
            mdata.SCP,
            sicdproj.compute_scp_coa_slant_plane_normal(mdata),
        )

    assert gpp_tgt.shape == gridlocs.shape[:-1] + (3,)

    spp_tgt, _, _ = sicdproj.r_rdot_to_constant_hae_surface(
        mdata.LOOK, mdata.SCP, proj_set, mdata.SCP_HAE
    )
    assert spp_tgt.shape == gridlocs.shape[:-1] + (3,)


def test_r_rdot_from_rgazim_rgazcomp(example_proj_metadata, image_grid_locations):
    example_proj_metadata.IFA = "RGAZCOMP"
    example_proj_metadata.Grid_Type = "RGAZIM"
    example_proj_metadata.AzSF = 2.0
    _projection_sets_smoketest(example_proj_metadata, image_grid_locations)


def test_r_rdot_from_rgzero(example_proj_metadata, image_grid_locations):
    example_proj_metadata.IFA = "RMA"
    example_proj_metadata.Grid_Type = "RGZERO"
    example_proj_metadata.cT_CA = np.array([1.0, 0.0001])
    example_proj_metadata.cDRSF = np.array([[1.0, 0.0001], [1.0, 0.0001]])
    example_proj_metadata.R_CA_SCP = 10000
    _projection_sets_smoketest(example_proj_metadata, image_grid_locations)


def test_r_rdot_from_xrgycr(mono_and_bi_proj_metadata, image_grid_locations):
    mono_and_bi_proj_metadata.Grid_Type = "XRGYCR"
    _projection_sets_smoketest(mono_and_bi_proj_metadata, image_grid_locations)


def test_r_rdot_from_xctyat(mono_and_bi_proj_metadata, image_grid_locations):
    mono_and_bi_proj_metadata.Grid_Type = "XCTYAT"
    _projection_sets_smoketest(mono_and_bi_proj_metadata, image_grid_locations)


def test_r_rdot_from_plane(mono_and_bi_proj_metadata, image_grid_locations):
    mono_and_bi_proj_metadata.IFA = "RMA"
    mono_and_bi_proj_metadata.Grid_Type = "PLANE"
    mono_and_bi_proj_metadata.cT_CA = np.array([1.0, 0.0001])
    mono_and_bi_proj_metadata.cDRSF = np.array([[1.0, 0.0001], [1.0, 0.0001]])
    mono_and_bi_proj_metadata.R_CA_SCP = 10000
    _projection_sets_smoketest(mono_and_bi_proj_metadata, image_grid_locations)


def test_apos_from_xml():
    all_attrs = set()
    set_attrs = set()
    for xml_file in (DATAPATH / "syntax_only/sicd").glob("*.xml"):
        etree = lxml.etree.parse(xml_file)
        if not sicdproj.AdjustableParameterOffsets.exists(etree):
            continue
        apos = sicdproj.AdjustableParameterOffsets.from_xml(etree)
        apo_dict = dataclasses.asdict(apos)
        all_attrs.update(apo_dict.keys())
        set_attrs.update(k for k, v in apo_dict.items() if v is not None)
    unset_attrs = all_attrs - set_attrs
    assert not unset_attrs


def test_apo_mono(example_proj_metadata):
    meta = example_proj_metadata
    assert meta.is_monostatic()
    apos = sicdproj.AdjustableParameterOffsets(
        delta_ARP_SCP_COA=np.array([10000.0, 11000.0, 12000.0]),
        delta_VARP=np.array([1500.0, 1600.0, 1700.0]),
        delta_tx_SCP_COA=10.0,
        delta_tr_SCP_COA=11.0,
    )
    proj_set = sicdproj.compute_projection_sets(meta, [0, 0])
    adjust_proj_set = sicdproj.apply_apos(meta, proj_set, apos)

    # Make sure things that were supposed to change did
    assert adjust_proj_set.t_COA == proj_set.t_COA
    assert np.all(adjust_proj_set.ARP_COA - proj_set.ARP_COA == [10000, 11000, 12000])
    assert np.all(adjust_proj_set.VARP_COA - proj_set.VARP_COA == [1500, 1600, 1700])
    assert adjust_proj_set.R_COA != proj_set.R_COA
    assert adjust_proj_set.Rdot_COA == proj_set.Rdot_COA


def test_apo_bi(example_proj_metadata_bi):
    meta = example_proj_metadata_bi
    assert meta.is_bistatic()
    apos = sicdproj.AdjustableParameterOffsets(
        delta_Xmt_SCP_COA=np.array([10000.0, 11000.0, 12000.0]),
        delta_VXmt=np.array([1500.0, 1600.0, 1700.0]),
        f_Clk_X_SF=10.0,
        delta_tx_SCP_COA=11.0,
        delta_Rcv_SCP_COA=np.array([20000.0, 21000.0, 22000.0]),
        delta_VRcv=np.array([2500.0, 2600.0, 2700.0]),
        f_Clk_R_SF=20.0,
        delta_tr_SCP_COA=21.0,
    )
    proj_set = sicdproj.compute_projection_sets(meta, [0, 0])
    adjust_proj_set = sicdproj.apply_apos(meta, proj_set, apos)

    # Make sure things that were supposed to change did
    assert adjust_proj_set.t_COA == proj_set.t_COA
    assert adjust_proj_set.tx_COA - proj_set.tx_COA == pytest.approx(11, abs=0.1)
    assert adjust_proj_set.tr_COA - proj_set.tr_COA == pytest.approx(21, abs=0.1)
    assert np.all(adjust_proj_set.Xmt_COA != proj_set.Xmt_COA)
    assert np.all(adjust_proj_set.VXmt_COA - proj_set.VXmt_COA == [1500, 1600, 1700])
    assert np.all(adjust_proj_set.Rcv_COA != proj_set.Rcv_COA)
    assert np.all(adjust_proj_set.VRcv_COA - proj_set.VRcv_COA == [2500, 2600, 2700])
    assert adjust_proj_set.R_Avg_COA != proj_set.R_Avg_COA
    assert adjust_proj_set.Rdot_Avg_COA != proj_set.Rdot_Avg_COA


def test_sensitivity_matrices(example_proj_metadata):
    mats = sarkit.sicd.projection._sensitivity.compute_sensitivity_matrices(
        example_proj_metadata
    )

    # sensitivity when image plane is already slant should be nearly -identity due to relative orientation of slant and
    # image plane vectors
    assert np.allclose(mats.M_SPXY_IL, -np.eye(2), atol=1e-3)

    assert np.allclose(mats.M_SPXY_GPXY @ mats.M_GPXY_SPXY, np.eye(2))
    assert np.allclose(mats.M_SPXY_IL @ mats.M_IL_SPXY, np.eye(2))
    assert np.allclose(mats.M_GPXY_IL @ mats.M_IL_GPXY, np.eye(2))
    np.allclose(mats.M_SPXY_PT @ mats.M_PT_GPXY, mats.M_SPXY_GPXY)
