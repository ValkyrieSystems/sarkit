import argparse
import contextlib
import filecmp
import pathlib
import sys
import tempfile

import lxml.builder
import lxml.etree

import sarkit.crsd as skcrsd

STUB_DIR = pathlib.Path(__file__).parent / "stubs"


def _remove(etree, pattern):
    if (elem := etree.find(pattern)) is not None:
        elem.getparent().remove(elem)
    else:
        print(f"Cannot find {pattern=}")


def _replace(etree, pattern, new_element):
    if (old_elem := etree.find(pattern)) is not None:
        old_elem.addnext(new_element)
        _remove(etree, pattern)


def _get_stub(stub_name, version_ns):
    new_elem = lxml.etree.parse(STUB_DIR / stub_name).getroot()
    for elem in new_elem.iter():
        elem.tag = f"{{{version_ns}}}{lxml.etree.QName(elem).localname}"
    return new_elem


def _add_sa_to_data(sa_elem):
    ns = lxml.etree.QName(sa_elem).namespace
    em = lxml.builder.ElementMaker(namespace=ns, nsmap={None: ns})
    sa_elem.getroottree().find("{*}Data/{*}Support").append(
        em.SupportArray(
            em.SAId(sa_elem.findtext("{*}Identifier")),
            em.NumRows("24"),
            em.NumCols("8"),
            em.BytesPerElement("2"),
            em.ArrayByteOffset("9999"),
        )
    )


def change_sar_choices(etree):
    ns = lxml.etree.QName(etree.getroot()).namespace
    _replace(
        etree,
        "{*}ErrorParameters/{*}SARImage/{*}Monostatic",
        _get_stub("ErrorParameters.SARImage.Bistatic.xml", ns),
    )
    _replace(
        etree,
        "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar",
        _get_stub("SceneCoordinates.ReferenceSurface.HAE.xml", ns),
    )
    new_xmarray = _get_stub("SupportArray.XMArray.xml", ns)
    etree.findall("{*}SupportArray/{*}FxResponseArray")[-1].addnext(
        new_xmarray,
    )
    _add_sa_to_data(new_xmarray)
    xmid_elem = lxml.etree.Element(f"{{{ns}}}XMId")
    xmid_elem.text = "XM Identifier"
    etree.find("{*}TxSequence/{*}Parameters/{*}RefPulseIndex").addnext(xmid_elem)
    etree.find("{*}TxSequence/{*}TxWFType").text = "LFM w XM"
    etree.find("{*}PPP/{*}FxResponseIndex").addnext(_get_stub("PPP.XMIndex.xml", ns))
    etree.find("{*}Data/{*}Transmit/{*}NumBytesPPP").text = str(
        8 + int(etree.findtext("{*}Data/{*}Transmit/{*}NumBytesPPP"))
    )
    _replace(
        etree,
        "{*}Channel/{*}Parameters/{*}SARImage/{*}DwellTimes/{*}Polynomials",
        _get_stub("Channel.Parameters.SARImage.DwellTimes.Array.xml", ns),
    )
    new_dtarray = _get_stub("SupportArray.DwellTimeArray.xml", ns)
    etree.findall("{*}SupportArray/{*}XMArray")[-1].addnext(
        new_dtarray,
    )
    _add_sa_to_data(new_dtarray)
    _remove(etree, "{*}DwellPolynomials")
    etree.find("{*}Data/{*}Receive/{*}NumCRSDChannels").addnext(
        _get_stub("Data.Receive.SignalCompression.xml", ns)
    )
    return etree


def make_tx_flavor(etree):
    ns = lxml.etree.QName(etree.getroot()).namespace
    etree.getroot().tag = f"{{{ns}}}CRSDtx"
    _remove(etree, "{*}SARInfo")
    _remove(etree, "{*}ReceiveInfo")
    _remove(etree, "{*}Global/{*}Receive")
    _remove(etree, "{*}SceneCoordinates/{*}ExtendedArea")
    _remove(etree, "{*}SceneCoordinates/{*}ImageGrid")
    _remove(etree, "{*}Data/{*}Receive")
    _remove(etree, "{*}Channel")
    _remove(etree, "{*}ReferenceGeometry/{*}SARImage")
    _remove(etree, "{*}ReferenceGeometry/{*}RcvParameters")
    _remove(etree, "{*}DwellPolynomials")
    _remove(etree, "{*}PVP")
    _remove(etree, "{*}ErrorParameters/{*}SARImage")
    stub = _get_stub("ErrorParameters.XXSensor.xml", ns)
    stub.tag = f"{{{ns}}}TxSensor"
    etree.find("{*}ErrorParameters").append(stub)
    return etree


def make_rcv_flavor(etree):
    ns = lxml.etree.QName(etree.getroot()).namespace
    etree.getroot().tag = f"{{{ns}}}CRSDrcv"
    _remove(etree, "{*}SARInfo")
    _remove(etree, "{*}TransmitInfo")
    _remove(etree, "{*}Global/{*}Transmit")
    _remove(etree, "{*}SceneCoordinates/{*}ExtendedArea")
    _remove(etree, "{*}SceneCoordinates/{*}ImageGrid")
    for elem in etree.findall("{*}SupportArray/{*}FxResponseArray/{*}Identifier"):
        _remove(
            etree, "{*}Data/{*}Support/{*}SupportArray[{*}SAId='" + f"{elem.text}']"
        )
    _remove(etree, "{*}Data/{*}Transmit")
    _remove(etree, "{*}TxSequence")
    _remove(etree, "{*}Channel/{*}Parameters/{*}SARImage")
    _remove(etree, "{*}ReferenceGeometry/{*}SARImage")
    _remove(etree, "{*}ReferenceGeometry/{*}TxParameters")
    _remove(etree, "{*}DwellPolynomials")
    _remove(etree, "{*}SupportArray/{*}FxResponseArray")
    _remove(etree, "{*}PPP")
    _remove(etree, "{*}PVP/{*}TxPulseIndex")
    etree.find("{*}Data/{*}Receive/{*}NumBytesPVP").text = str(
        int(etree.findtext("{*}Data/{*}Receive/{*}NumBytesPVP")) - 8
    )
    _remove(etree, "{*}ErrorParameters/{*}SARImage")
    stub = _get_stub("ErrorParameters.XXSensor.xml", ns)
    stub.tag = f"{{{ns}}}RcvSensor"
    etree.find("{*}ErrorParameters").append(stub)
    return etree


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
            ((lambda x: x), change_sar_choices, make_tx_flavor, make_rcv_flavor)
        ):
            etree = lxml.etree.parse(
                pathlib.Path(__file__).parent
                / "manual-syntax-only-crsd-mono-sar-1.0.xml"
            )
            mods(etree)
            version_ns = lxml.etree.QName(etree.getroot()).namespace
            lxml.etree.cleanup_namespaces(etree, top_nsmap={None: version_ns})
            version_info = skcrsd.VERSION_INFO[version_ns]
            schema = lxml.etree.XMLSchema(file=version_info["schema"])
            lxml.etree.indent(etree, space=" " * 4)
            filename = f"{index:04d}-syntax-only-crsd-{version_info['version']}.xml"
            if not schema(etree):
                print(f"Warning for {filename}:")
                print(schema.error_log)
                print()
            etree.write(
                outdir / filename,
                pretty_print=True,
            )
        if config.check:
            diff = filecmp.dircmp(pathlib.Path(__file__).parent, outdir)
            checks_out = not bool(
                diff.diff_files
                or {
                    pathlib.Path(__file__).name,
                    "manual-syntax-only-crsd-mono-sar-1.0.xml",
                    "stubs",
                }.symmetric_difference(diff.left_only)
            )
            return not checks_out


if __name__ == "__main__":
    sys.exit(main())
