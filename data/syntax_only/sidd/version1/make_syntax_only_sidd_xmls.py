import argparse
import contextlib
import filecmp
import functools
import itertools
import pathlib
import sys
import tempfile

import lxml.builder
import lxml.etree

import sarkit.sidd as sksidd


def display_remap_info_choice():
    def change_remap_info(sidd_etree, *, remap_info):
        ns = lxml.etree.QName(sidd_etree.getroot()).namespace
        em = lxml.builder.ElementMaker(namespace=ns, nsmap=sidd_etree.getroot().nsmap)

        disp = sidd_etree.find("{*}Display")
        del disp[1:]  # leave only the PixelType

        if remap_info == "color":
            disp.append(
                em.RemapInformation(
                    em.ColorDisplayRemap(em.RemapLUT("0,1,2 4,5,6 7,8,9", size="3"))
                )
            )
        elif remap_info == "mono":
            disp.append(
                em.RemapInformation(
                    em.MonochromeDisplayRemap(
                        em.RemapType("x"),
                    )
                )
            )
        else:
            raise ValueError(remap_info)

    for x in ("color", "mono"):
        yield functools.partial(change_remap_info, remap_info=x)


def measurement_proj_choice():
    def change_measurement_proj(sidd_etree, *, proj_type):
        ns = lxml.etree.QName(sidd_etree.getroot()).namespace
        em = lxml.builder.ElementMaker(namespace=ns, nsmap=sidd_etree.getroot().nsmap)

        new_elem = em(
            proj_type, sidd_etree.find("{*}Measurement")[0][0]
        )  # start with ReferencePoint
        if proj_type == "PolynomialProjection":
            for subelem in (
                "RowColToLat",
                "RowColToLon",
                "RowColToAlt",
                "LatLonToRow",
                "LatLonToCol",
            ):
                new_elem.append(
                    em(
                        subelem,
                        em(
                            "{urn:SICommon:0.1}Coef",
                            "0.0",
                            exponent1="0",
                            exponent2="0",
                        ),
                        order1="0",
                        order2="0",
                    )
                )
        else:
            new_elem.extend(
                (
                    em.SampleSpacing(
                        em("{urn:SICommon:0.1}Row", "0.0"),
                        em("{urn:SICommon:0.1}Col", "0.0"),
                    ),
                    em.TimeCOAPoly(
                        em(
                            "{urn:SICommon:0.1}Coef",
                            "0.0",
                            exponent1="0",
                            exponent2="0",
                        ),
                        order1="0",
                        order2="0",
                    ),
                )
            )
            if proj_type == "PlaneProjection":
                new_elem.append(
                    em.ProductPlane(
                        *[
                            em(
                                f"{d}UnitVector",
                                *[
                                    em("{urn:SICommon:0.1}" + xyz, "0.0")
                                    for xyz in "XYZ"
                                ],
                            )
                            for d in ("Row", "Col")
                        ]
                    )
                )
            if proj_type == "CylindricalProjection":
                new_elem.extend(
                    (
                        em.StripmapDirection(
                            *[em("{urn:SICommon:0.1}" + xyz, "0.0") for xyz in "XYZ"]
                        ),
                        em.CurvatureRadius("0.8"),
                    )
                )
        sidd_etree.find("{*}Measurement")[0] = new_elem

    for proj_type in (
        "PolynomialProjection",
        "GeographicProjection",
        "PlaneProjection",
        "CylindricalProjection",
    ):
        yield functools.partial(change_measurement_proj, proj_type=proj_type)


def error_stats_choice():
    def change_errorstats_choice(sidd_etree, *, comp_scp_type):
        ns = lxml.etree.QName(sidd_etree.getroot()).namespace
        em = lxml.builder.ElementMaker(namespace=ns, nsmap=sidd_etree.getroot().nsmap)

        # no trailing optional nodes so we can be lazy about placement
        assert sidd_etree.find("{*}ErrorStatistics") is None
        assert sidd_etree.find("{*}Radiometric") is None
        assert sidd_etree.find("{*}Annotations") is None

        errstats = em.ErrorStatistics(em("{urn:SICommon:0.1}CompositeSCP"))
        if comp_scp_type == "RgAz":
            errstats[0].append(
                em(
                    "{urn:SICommon:0.1}RgAzErr",
                    *[
                        em("{urn:SICommon:0.1}" + x, "0.0")
                        for x in ("Rg", "Az", "RgAz")
                    ],
                )
            )
        elif comp_scp_type == "RowCol":
            errstats[0].append(
                em(
                    "{urn:SICommon:0.1}RowColErr",
                    *[
                        em("{urn:SICommon:0.1}" + x, "0.0")
                        for x in ("Row", "Col", "RowCol")
                    ],
                )
            )
        else:
            raise ValueError(comp_scp_type)
        sidd_etree.getroot().append(errstats)

    for x in ("RgAz", "RowCol"):
        yield functools.partial(change_errorstats_choice, comp_scp_type=x)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Don't write the files, just return the status. Return code 0 means nothing would change.",
    )
    config = parser.parse_args(args)
    with (
        tempfile.TemporaryDirectory()
        if config.check
        else contextlib.nullcontext(config.output_dir)
    ) as outdir:
        outdir = pathlib.Path(outdir)
        matches_schema = True
        for index, mods in enumerate(
            itertools.zip_longest(
                display_remap_info_choice(),
                measurement_proj_choice(),
                error_stats_choice(),
            )
        ):
            etree = lxml.etree.parse(
                pathlib.Path(__file__).parent
                / "manual-syntax-only-sidd-1.0-minimal.xml"
            )
            for mod in mods:
                if mod is not None:
                    mod(etree)
            version_ns = lxml.etree.QName(etree.getroot()).namespace
            lxml.etree.cleanup_namespaces(etree, top_nsmap={None: version_ns})
            version_info = sksidd.VERSION_INFO[version_ns]
            schema = lxml.etree.XMLSchema(file=version_info["schema"])
            lxml.etree.indent(etree, space=" " * 4)
            filename = f"{index:04d}-syntax-only-sidd-{version_info['version']}.xml"
            if not schema(etree):
                print(f"Warning for {filename}:")
                print(schema.error_log)
                print()
                matches_schema = False
            etree.write(
                outdir / filename,
                pretty_print=True,
            )
            return not matches_schema
        if config.check:
            diff = filecmp.dircmp(pathlib.Path(__file__).parent, outdir)
            checks_out = not bool(
                diff.diff_files
                or {
                    pathlib.Path(__file__).name,
                    "manual-syntax-only-sidd-1.0.xml",
                    "manual-syntax-only-sidd-1.0-minimal.xml",
                }.symmetric_difference(diff.left_only)
            )
            return not checks_out


if __name__ == "__main__":
    sys.exit(main())
