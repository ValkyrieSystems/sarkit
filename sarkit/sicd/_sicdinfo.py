"""Command line utility for inspecting a SICD file"""

import argparse
import os
import sys

import lxml.etree

import sarkit.sicd as sksicd

try:
    from smart_open import open
except ImportError:
    pass


def _parser():
    parser = argparse.ArgumentParser(description="Display information about SICD files")
    parser.add_argument("filename")
    parser.add_argument(
        "--xml", "-x", action="store_true", help="Extract XML formatted for display"
    )
    parser.add_argument(
        "--segments",
        "-s",
        action="store_true",
        help="Display Image Segment information",
    )
    parser.add_argument("--raw", choices=["XML"], help="Extract raw bytes of a block")
    return parser


def main(args=None):
    config = _parser().parse_args(args)

    with open(config.filename, "rb") as file, sksicd.NitfReader(file) as reader:
        if config.segments:
            for imseg in reader.jbp["ImageSegments"]:
                subhdr = imseg["subheader"]
                iid1 = subhdr["IID1"].value
                nrows = subhdr["NROWS"].value
                ncols = subhdr["NCOLS"].value
                pvtype = subhdr["PVTYPE"].value
                nbpp = subhdr["NBPP"].value
                subcat1 = subhdr["ISUBCAT00001"].value
                subcat2 = subhdr["ISUBCAT00002"].value
                print(
                    f"{iid1} {nrows:8d} x {ncols}   {nbpp:2d} {pvtype} ({subcat1}, {subcat2})"
                )

        if config.xml or config.raw == "XML":
            deseg = reader.jbp["DataExtensionSegments"][
                0
            ]  # SICD XML must be in first DES
            file.seek(deseg["DESDATA"].get_offset(), os.SEEK_SET)
            raw_xml = file.read(deseg["DESDATA"].size)

            if config.xml:
                parser = lxml.etree.XMLParser(remove_blank_text=True)
                tree = lxml.etree.fromstring(raw_xml, parser)
                pretty = lxml.etree.tostring(tree, pretty_print=True)
                print(pretty.decode(), end="")

            if config.raw == "XML":
                sys.stdout.buffer.write(raw_xml)


if __name__ == "__main__":
    main()
