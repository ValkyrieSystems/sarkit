"""Command line utility for inspecting a SIDD file"""

import argparse
import os
import sys

import jbpy
import lxml.etree

try:
    from smart_open import open
except ImportError:
    pass


def _parser():
    parser = argparse.ArgumentParser(description="Display information about SIDD files")
    parser.add_argument("filename")
    parser.add_argument("--image-number", type=int)
    parser.add_argument(
        "--segments",
        "-s",
        action="store_true",
        help="Display Image Segment information",
    )
    parser.add_argument(
        "--xml", "-x", action="store_true", help="Extract XML formatted for display"
    )
    parser.add_argument(
        "--raw", choices=["XML", "image"], help="Extract raw bytes of a block"
    )
    return parser


def main(args=None):
    config = _parser().parse_args(args)

    ntf = jbpy.Jbp()
    with open(config.filename, "rb") as file:
        ntf.load(file)

    images = {}
    for imseg in ntf["ImageSegments"]:
        subhdr = imseg["subheader"]
        iid1 = subhdr["IID1"].value
        if iid1.startswith("SIDD"):
            image_number = int(iid1[4:7])
            segment_number = int(iid1[7:])
        images.setdefault(image_number, {})
        images[image_number][segment_number] = imseg

    if config.image_number is not None:
        images = {config.image_number: images[config.image_number]}

    if config.segments:
        for image_number in sorted(images.keys()):
            print(f"Image number {image_number}")
            for segment_number in sorted(images[image_number].keys()):
                subhdr = images[image_number][segment_number]["subheader"]
                iid1 = subhdr["IID1"].value
                nrows = subhdr["NROWS"].value
                ncols = subhdr["NCOLS"].value
                pvtype = subhdr["PVTYPE"].value
                irep = subhdr["IREP"].value
                nbpp = subhdr["NBPP"].value
                subcat1 = subhdr["ISUBCAT00001"].value
                irepbands = [
                    field.value.strip()
                    for field in sorted(
                        subhdr.find_all("IREPBAND\\d+"), key=lambda x: x.name
                    )
                ]
                print(
                    f"{iid1} {nrows:8d} x {ncols}   {nbpp:2d} {pvtype} {irep} {subcat1} {irepbands}"
                )

    if config.raw == "image":
        with open(config.filename, "rb") as file:
            for image_number in sorted(images.keys()):
                for segment_number in sorted(images[image_number].keys()):
                    imseg = images[image_number][segment_number]
                    file.seek(imseg["Data"].get_offset(), os.SEEK_SET)
                    imdata = file.read(imseg["Data"].size)
                    sys.stdout.buffer.write(imdata)

    if config.xml or config.raw == "XML":
        for image_number in sorted(images.keys()):
            with open(config.filename, "rb") as file:
                deseg = ntf["DataExtensionSegments"][image_number - 1]
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
