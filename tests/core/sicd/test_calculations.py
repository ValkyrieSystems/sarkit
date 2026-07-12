import pathlib

import lxml.etree
import numpy as np

import sarkit.sicd as sksicd

DATAPATH = pathlib.Path(__file__).parents[3] / "data"


def test():
    xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    ew = sksicd.ElementWrapper(xmltree.getroot())
    ss = (ew["Grid"]["Row"]["SS"], ew["Grid"]["Col"]["SS"])
    scppixel = ew["ImageData"]["SCPPixel"]

    np.testing.assert_allclose(sksicd.irowicol_to_rowcol(xmltree, (0, 0)), scppixel)
    np.testing.assert_allclose(sksicd.xrowycol_to_rowcol(xmltree, (0, 0)), scppixel)
    np.testing.assert_allclose(
        sksicd.rowcol_to_xrowycol(xmltree, (1, 1))
        - sksicd.rowcol_to_xrowycol(xmltree, (0, 0)),
        ss,
    )
    np.testing.assert_allclose(
        sksicd.irowicol_to_xrowycol(xmltree, (1, 1))
        - sksicd.irowicol_to_xrowycol(xmltree, (0, 0)),
        ss,
    )


def test_roundtrip_through_irowicol():
    xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    rowcol = np.random.default_rng().uniform(0, 20000, (5, 6, 7, 2))

    # round trip through irowicol
    irowicol = sksicd.rowcol_to_irowicol(xmltree, rowcol)
    xrowycol = sksicd.irowicol_to_xrowycol(xmltree, irowicol)
    irowicol_rt = sksicd.xrowycol_to_irowicol(xmltree, xrowycol)
    rowcol_rt = sksicd.irowicol_to_rowcol(xmltree, irowicol_rt)

    np.testing.assert_allclose(irowicol, irowicol_rt)
    np.testing.assert_allclose(rowcol, rowcol_rt)


def test_roundtrip_direct():
    xmltree = lxml.etree.parse(DATAPATH / "example-sicd-1.3.0.xml")
    rowcol = np.random.default_rng().uniform(0, 20000, (5, 6, 7, 2))

    xrowycol = sksicd.rowcol_to_xrowycol(xmltree, rowcol)
    rowcol_rt = sksicd.xrowycol_to_rowcol(xmltree, xrowycol)

    np.testing.assert_allclose(rowcol, rowcol_rt)
