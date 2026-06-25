import pathlib

import lxml.etree
import numpy as np
import pytest

import sarkit.sicd as sksicd
import sarkit.wgs84

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


@pytest.mark.parametrize("method", ("monostatic", "bistatic", None))
def test_scp_image_to_ground_mono(method):
    _assert_image_to_ground_projections(DATAPATH / "example-sicd-1.3.0.xml", method)


@pytest.mark.parametrize("method", ("bistatic", None))
def test_scp_image_to_ground_bi(method):
    _assert_image_to_ground_projections(DATAPATH / "example-sicd-1.4.0.xml", method)


def _assert_image_to_ground_projections(sicd_xml, method):
    sicd_xmltree = lxml.etree.parse(sicd_xml)
    xmlhelp = sksicd.XmlHelper(sicd_xmltree)
    scp = xmlhelp.load("{*}GeoData/{*}SCP/{*}ECF")
    image_plane_normal = np.cross(
        xmlhelp.load("{*}Grid/{*}Row/{*}UVectECF"),
        xmlhelp.load("{*}Grid/{*}Col/{*}UVectECF"),
    )

    # Project SCP
    projected_scp, delta_gp, success = sksicd.image_to_ground_plane(
        sicd_xmltree,
        [0, 0],
        scp,
        image_plane_normal,
        method=method,
    )
    assert np.allclose(scp, projected_scp, atol=0.1)
    assert success
    in_shape = np.asarray([0, 0]).shape
    assert projected_scp.shape == in_shape[:-1] + (3,)
    assert delta_gp.shape == in_shape[:-1]

    # Project ND-array around SCP - assumes validity close to SCP
    im_coords = np.random.default_rng(12345).uniform(
        low=-24.0, high=24.0, size=(3, 4, 5, 2)
    )
    plane_coords, delta_gp, success = sksicd.image_to_ground_plane(
        sicd_xmltree,
        im_coords,
        scp,
        image_plane_normal,
        method=method,
    )
    assert plane_coords.shape == im_coords.shape[:-1] + (3,)
    assert delta_gp.shape == im_coords.shape[:-1]

    # Check that points are in the plane
    resid = (plane_coords - scp) @ image_plane_normal
    assert np.allclose(resid, 0)
    assert success

    # Project back to image
    re_im_coords, delta_gp, success = sksicd.scene_to_image(
        sicd_xmltree,
        plane_coords,
    )
    assert re_im_coords.shape == plane_coords.shape[:-1] + (2,)
    assert delta_gp.shape == plane_coords.shape[:-1]
    assert np.isfinite(delta_gp).all()
    assert success
    assert im_coords == pytest.approx(re_im_coords, abs=1e-3)

    re_im_coords, delta_gp, success = sksicd.scene_to_image(
        sicd_xmltree, plane_coords, delta_gp_s2i=1e-9
    )
    assert (delta_gp > 1e-9).any()
    assert not success


@pytest.mark.parametrize(
    "sicd_xml",
    (
        DATAPATH / "example-sicd-1.3.0.xml",  # monostatic
        DATAPATH / "example-sicd-1.4.0.xml",  # bistatic
    ),
)
def test_image_to_constant_hae_surface(sicd_xml):
    sicd_xmltree = lxml.etree.parse(sicd_xml)
    xmlhelp = sksicd.XmlHelper(sicd_xmltree)
    scp = xmlhelp.load("{*}GeoData/{*}SCP/{*}ECF")
    scp_hae = xmlhelp.load("{*}GeoData/{*}SCP/{*}LLH/{*}HAE")

    # Project SCP
    scp_coords = np.array([0, 0])
    projected_scp, delta_hae_max, success = sksicd.image_to_constant_hae_surface(
        sicd_xmltree,
        scp_coords,
        scp_hae,
    )
    assert np.allclose(scp, projected_scp, atol=0.1)
    assert success
    assert projected_scp.shape == scp_coords.shape[:-1] + (3,)
    assert delta_hae_max.shape == scp_coords.shape[:-1]

    # Project ND-array around SCP - assumes validity close to SCP
    im_coords = np.random.default_rng(12345).uniform(
        low=-24.0, high=24.0, size=(3, 4, 5, 2)
    )
    surf_coords, delta_hae_max, success = sksicd.image_to_constant_hae_surface(
        sicd_xmltree,
        im_coords,
        scp_hae,
    )
    assert surf_coords.shape == im_coords.shape[:-1] + (3,)
    assert delta_hae_max.shape == im_coords.shape[:-1]

    # Check that points are on the surface
    surf_coords_llh = sarkit.wgs84.cartesian_to_geodetic(surf_coords)
    assert surf_coords_llh == pytest.approx(scp_hae, abs=1e-3)
    assert success

    # Project back to image
    re_im_coords, delta_hae_max, success = sksicd.scene_to_image(
        sicd_xmltree,
        surf_coords,
    )
    assert re_im_coords.shape == surf_coords.shape[:-1] + (2,)
    assert delta_hae_max.shape == surf_coords.shape[:-1]
    assert np.isfinite(delta_hae_max).all()
    assert success
    assert im_coords == pytest.approx(re_im_coords, abs=1e-3)


@pytest.mark.parametrize(
    "sicd_xml",
    (
        DATAPATH / "example-sicd-1.3.0.xml",  # monostatic
        DATAPATH / "example-sicd-1.4.0.xml",  # bistatic
    ),
)
def test_image_to_dem_surface(sicd_xml):
    sicd_xmltree = lxml.etree.parse(sicd_xml)
    xmlhelp = sksicd.XmlHelper(sicd_xmltree)
    scp_hae = xmlhelp.load("{*}GeoData/{*}SCP/{*}LLH/{*}HAE")

    # Project to constant HAE surface around SCP - assumes validity close to SCP
    im_coords = np.random.default_rng(12345).uniform(low=-24.0, high=24.0, size=(32, 2))
    surf_coords, _, success = sksicd.image_to_constant_hae_surface(
        sicd_xmltree,
        im_coords,
        scp_hae,
    )
    assert success

    def hae_dem_func(ecf):
        return sarkit.wgs84.cartesian_to_geodetic(ecf)[..., -1] - scp_hae

    for im_coord, surf_coord in zip(im_coords, surf_coords):
        hae_dem_coord = sksicd.image_to_dem_surface(
            sicd_xmltree,
            im_coord,
            ecef2dem_func=hae_dem_func,
            hae_min=scp_hae - 10.0,
            hae_max=scp_hae + 10.0,
            delta_dist_dem=1.0,
        )
        # assert only one intersection since DEM is an HAE surface
        assert hae_dem_coord.size == surf_coord.size
        assert np.allclose(hae_dem_coord, surf_coord, atol=0.01)


@pytest.fixture
def mono_1_5():
    xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    old_ew = sksicd.ElementWrapper(xmltree.getroot())
    assert old_ew["CollectionInfo"]["CollectType"] == "MONOSTATIC"
    new_ew = sksicd.ElementWrapper(lxml.etree.Element("{urn:SICD:1.5}SICD"))
    new_ew.from_dict(old_ew.to_dict())
    schema = lxml.etree.XMLSchema(file=sksicd.VERSION_INFO["urn:SICD:1.5"]["schema"])
    schema.assertValid(new_ew.elem)
    return new_ew.elem.getroottree()


def image_to_ground(sicdxml):
    xmlhelp = sksicd.XmlHelper(sicdxml)
    scp = xmlhelp.load("{*}GeoData/{*}SCP/{*}ECF")
    image_plane_normal = np.cross(
        xmlhelp.load("{*}Grid/{*}Row/{*}UVectECF"),
        xmlhelp.load("{*}Grid/{*}Col/{*}UVectECF"),
    )
    im_coords = np.random.default_rng(12345).uniform(
        low=-24.0, high=24.0, size=(3, 4, 5, 2)
    )
    gpp, _, success = sksicd.image_to_ground_plane(
        sicdxml,
        im_coords,
        scp,
        image_plane_normal,
    )
    assert success
    return gpp


def image_to_constant_hae(sicdxml):
    xmlhelp = sksicd.XmlHelper(sicdxml)
    scp_hae = xmlhelp.load("{*}GeoData/{*}SCP/{*}LLH/{*}HAE")
    im_coords = np.random.default_rng(12345).uniform(
        low=-24.0, high=24.0, size=(3, 4, 5, 2)
    )
    surf_coords, _, success = sksicd.image_to_constant_hae_surface(
        sicdxml,
        im_coords,
        scp_hae,
    )
    assert success
    return surf_coords


def image_to_dem(sicdxml):
    xmlhelp = sksicd.XmlHelper(sicdxml)
    scp_hae = xmlhelp.load("{*}GeoData/{*}SCP/{*}LLH/{*}HAE")

    def hae_dem_func(ecf):
        return sarkit.wgs84.cartesian_to_geodetic(ecf)[..., -1] - scp_hae

    im_coord = np.random.default_rng(12345).uniform(low=-24.0, high=24.0, size=(2,))
    hae_dem_coord = sksicd.image_to_dem_surface(
        sicdxml,
        im_coord,
        ecef2dem_func=hae_dem_func,
        hae_min=scp_hae - 10.0,
        hae_max=scp_hae + 10.0,
        delta_dist_dem=1.0,
    )
    return hae_dem_coord


def scene_to_image(sicdxml):
    xmlhelp = sksicd.XmlHelper(sicdxml)
    scp = xmlhelp.load("{*}GeoData/{*}SCP/{*}ECF")
    scene_coords = scp + np.random.default_rng(12345).uniform(
        low=-24.0, high=24.0, size=(3, 4, 5, 3)
    )
    im_coords, _, success = sksicd.scene_to_image(sicdxml, scene_coords)
    assert success
    return im_coords


@pytest.mark.parametrize(
    "projfunc", [image_to_ground, image_to_constant_hae, image_to_dem, scene_to_image]
)
def test_proj_with_apos_mono(mono_1_5, projfunc):
    # project without APOs
    assert mono_1_5.find(".//{*}AdjustableParameterOffsets") is None
    pts_no_apo = projfunc(mono_1_5)

    # project with APOs set to 0
    ew = sksicd.ElementWrapper(mono_1_5.getroot())
    ew["ErrorStatistics"]["AdjustableParameterOffsets"]["ARPPosSCPCOA"] = [0, 0, 0]
    ew["ErrorStatistics"]["AdjustableParameterOffsets"]["ARPVel"] = [0, 0, 0]
    ew["ErrorStatistics"]["AdjustableParameterOffsets"]["TxTimeSCPCOA"] = 0.0
    ew["ErrorStatistics"]["AdjustableParameterOffsets"]["RcvTimeSCPCOA"] = 0.0
    pts_zero_apo = projfunc(mono_1_5)
    assert np.array_equal(pts_no_apo, pts_zero_apo)

    # project with nonzero APOs
    ew["ErrorStatistics"]["AdjustableParameterOffsets"]["ARPPosSCPCOA"] = [1, 0, 0]
    pts_nonzero_apo = projfunc(mono_1_5)
    assert bool(np.all(np.any(pts_no_apo != pts_nonzero_apo, axis=-1)))


@pytest.mark.parametrize(
    "projfunc", [image_to_ground, image_to_constant_hae, image_to_dem, scene_to_image]
)
def test_proj_with_apos_bi(projfunc):
    sicdxml = lxml.etree.parse(DATAPATH / "example-sicd-1.5.xml")
    assert sicdxml.findtext("{*}CollectionInfo/{*}CollectType") == "BISTATIC"

    # project without APOs
    assert sicdxml.find(".//{*}BistaticAdjustableParameterOffsets") is None
    pts_no_apo = projfunc(sicdxml)

    # project with APOs set to 0
    biapos = sksicd.ElementWrapper(sicdxml.getroot())["ErrorStatistics"][
        "BistaticAdjustableParameterOffsets"
    ]
    for plat in ("TxPlatform", "RcvPlatform"):
        biapos[plat]["APCPosSCPCOA"] = [0, 0, 0]
        biapos[plat]["APCVel"] = [0, 0, 0]
        biapos[plat]["TimeSCPCOA"] = 0.0
        biapos[plat]["ClockFreqSF"] = 0.0
    pts_zero_apo = projfunc(sicdxml)
    assert np.array_equal(pts_no_apo, pts_zero_apo)

    # project with nonzero APOs
    biapos["TxPlatform"]["APCPosSCPCOA"] = [1, 0, 0]
    pts_nonzero_apo = projfunc(sicdxml)
    assert bool(np.all(np.any(pts_no_apo != pts_nonzero_apo, axis=-1)))
