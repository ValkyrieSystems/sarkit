import argparse
import contextlib
import filecmp
import functools
import pathlib
import tempfile

import lxml.etree

import sarkit.sicd as sksicd

STUB_DIR = pathlib.Path(__file__).parent / "stubs"


def update_1_1_0_to_1_2_1(etree):
    new_enums = ("V:RHC", "V:LHC", "H:RHC", "H:LHC", "RHC:V", "RHC:H", "LHC:V", "LHC:H")
    for elem, new_enum in zip(
        etree.findall(
            "{*}RadarCollection/{*}RcvChannels/{*}ChanParameters/{*}TxRcvPolarization"
        ),
        new_enums,
    ):
        elem.text = new_enum
    etree.find("{*}ImageFormation/{*}TxRcvPolarizationProc").text = new_enums[-1]
    for elem in etree.iter():
        elem.tag = f"{{urn:SICD:1.2.1}}{lxml.etree.QName(elem).localname}"


def update_1_1_0_to_1_3_0(etree):
    etree.find(
        "{*}RadarCollection/{*}TxPolarization"
    ).text = "OTHER-InsertAnyStringHere"
    for index, elem in enumerate(
        etree.findall("{*}RadarCollection/{*}TxSequence/{*}TxStep/{*}TxPolarization")
    ):
        elem.text = f"OTHER{index}"
    for elem in etree.findall(
        "{*}RadarCollection/{*}RcvChannels/{*}ChanParameters/{*}TxRcvPolarization"
    ):
        elem.text = "OTHER123456:LHC"
    etree.find("{*}ImageFormation/{*}TxRcvPolarizationProc").text = "Y:OTHER-notY"
    new_elem = lxml.etree.parse(STUB_DIR / "ErrorStatistics.Unmodeled.xml").getroot()
    etree.find("{*}ErrorStatistics/{*}AdditionalParms").addprevious(new_elem)
    for elem in etree.iter():
        elem.tag = f"{{urn:SICD:1.3.0}}{lxml.etree.QName(elem).localname}"


def update_1_1_0_to_1_4_0(etree):
    update_1_1_0_to_1_3_0(etree)
    etree.find("{*}CollectionInfo/{*}Classification").addnext(
        lxml.etree.Element("InformationSecurityMarking")
    )

    etree.find("{*}SCPCOA/{*}GrazeAng").text = "-90.0"
    etree.find("{*}SCPCOA/{*}IncidenceAng").text = "179.9999"

    err_elem = etree.find("{*}ErrorStatistics")
    if etree.findtext("{*}CollectionInfo/{*}CollectType") == "MONOSTATIC":
        err_elem.append(
            lxml.etree.parse(
                STUB_DIR / "ErrorStatistics.AdjustableParameterOffsets.xml"
            ).getroot()
        )
    else:
        err_elem[:] = []
        err_elem.extend(
            lxml.etree.parse(STUB_DIR / "ErrorStatistics-Bistatic.xml").getroot()
        )

        etree.find("{*}SCPCOA").append(
            lxml.etree.parse(STUB_DIR / "SCPCOA.Bistatic.xml").getroot()
        )

    for elem in etree.iter():
        elem.tag = f"{{urn:SICD:1.4.0}}{lxml.etree.QName(elem).localname}"


def update_version(etree, urn):
    converter = {
        "urn:SICD:1.1.0": lambda x: x,
        "urn:SICD:1.2.1": update_1_1_0_to_1_2_1,
        "urn:SICD:1.3.0": update_1_1_0_to_1_3_0,
        "urn:SICD:1.4.0": update_1_1_0_to_1_4_0,
    }[urn]
    return converter(etree)


def _remove(etree, pattern):
    if (elem := etree.find(pattern)) is not None:
        elem.getparent().remove(elem)


def change_image_form_algo(etree, ifa_label):
    _remove(etree, "{*}RgAzComp")
    _remove(etree, "{*}PFA")
    _remove(etree, "{*}RMA")
    ifa_text = ifa_label.split("-")[0].upper()
    etree.find("{*}ImageFormation/{*}ImageFormAlgo").text = ifa_text
    new_elem = lxml.etree.parse(STUB_DIR / f"{ifa_label}.xml").getroot()
    for elem in new_elem.iter():
        elem.tag = f"{{{lxml.etree.QName(etree.getroot()).namespace}}}{lxml.etree.QName(elem).localname}"
    etree.getroot().append(new_elem)


def add_amptable(etree):
    pixeltype_elem = etree.find("{*}ImageData/{*}PixelType")
    pixeltype_elem.text = "AMP8I_PHS8I"
    _remove(etree, "{*}ImageData/{*}AmpTable")
    ns = f"{{{lxml.etree.QName(etree.getroot()).namespace}}}"
    new_amptable = lxml.etree.Element(ns + "AmpTable", size="256")
    for index in range(256):
        amp = lxml.etree.SubElement(new_amptable, ns + "Amplitude", index=str(index))
        amp.text = str(round(index * 0.001, 3))
    pixeltype_elem.addnext(new_amptable)


def set_version_ifa(etree, urn, ifa_label):
    change_image_form_algo(etree, ifa_label=ifa_label)
    update_version(etree, urn)


def set_version_collect_type(etree, urn, collecttype):
    etree.find("{*}CollectionInfo/{*}CollectType").text = collecttype
    update_version(etree, urn)


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

        for index, mods in enumerate(
            (
                functools.partial(
                    set_version_ifa, urn="urn:SICD:1.1.0", ifa_label="RgAzComp"
                ),
                functools.partial(
                    set_version_ifa, urn="urn:SICD:1.2.1", ifa_label="PFA"
                ),
                functools.partial(
                    set_version_ifa, urn="urn:SICD:1.3.0", ifa_label="RMA-RMAT"
                ),
                functools.partial(
                    set_version_ifa, urn="urn:SICD:1.1.0", ifa_label="RMA-RMCR"
                ),
                functools.partial(
                    set_version_ifa, urn="urn:SICD:1.2.1", ifa_label="RMA-INCA"
                ),
                functools.partial(
                    set_version_collect_type,
                    urn="urn:SICD:1.4.0",
                    collecttype="MONOSTATIC",
                ),
                functools.partial(
                    set_version_collect_type,
                    urn="urn:SICD:1.4.0",
                    collecttype="BISTATIC",
                ),
                add_amptable,
            )
        ):
            etree = lxml.etree.parse(
                pathlib.Path(__file__).parent / "manual-syntax-only-sicd-1.1.0.xml",
            )
            mods(etree)
            lxml.etree.cleanup_namespaces(etree)
            version_info = sksicd.VERSION_INFO[
                lxml.etree.QName(etree.getroot()).namespace
            ]
            schema = lxml.etree.XMLSchema(file=version_info["schema"])
            schema.assertValid(etree)
            lxml.etree.indent(etree, space=" " * 4)
            etree.write(
                outdir / f"{index:04d}-syntax-only-sicd-{version_info['version']}.xml",
                pretty_print=True,
            )

        if config.check:
            diff = filecmp.dircmp(pathlib.Path(__file__).parent, outdir)
            checks_out = not bool(
                diff.diff_files
                or {
                    pathlib.Path(__file__).name,
                    "manual-syntax-only-sicd-1.0.1.xml",
                    "manual-syntax-only-sicd-1.1.0.xml",
                    "stubs",
                }.symmetric_difference(diff.left_only)
            )
            return not checks_out


if __name__ == "__main__":
    exit(main())
