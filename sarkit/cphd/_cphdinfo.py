"""Command line utility for inspecting a CPHD file"""

import argparse
import sys

import lxml.etree

import sarkit.cphd as skcphd

try:
    from smart_open import open
except ImportError:
    pass

READ_SIZE = 128 * 1024 * 1024


def _parser():
    parser = argparse.ArgumentParser(description="Display information about CPHD files")
    parser.add_argument("filename")
    parser.add_argument(
        "--xml", "-x", action="store_true", help="Extract XML formatted for display"
    )
    parser.add_argument("--channels", "-c", action="store_true", help="List channels")
    parser.add_argument(
        "--raw",
        choices=["XML", "SUPPORT", "PVP", "SIGNAL"],
        help="Extract raw bytes of a BLOCK",
    )
    return parser


def main(args=None):
    config = _parser().parse_args(args)

    with open(config.filename, "rb") as file:
        _, kvp_list = skcphd.read_file_header(file)

        file.seek(int(kvp_list["XML_BLOCK_BYTE_OFFSET"]))
        xmlstr = file.read(int(kvp_list["XML_BLOCK_SIZE"]))

        if config.xml:
            parser = lxml.etree.XMLParser(remove_blank_text=True)
            tree = lxml.etree.fromstring(xmlstr, parser)
            pretty = lxml.etree.tostring(tree, pretty_print=True)
            print(pretty.decode(), end="")

        if config.raw:
            file.seek(int(kvp_list[f"{config.raw}_BLOCK_BYTE_OFFSET"]))
            bytes_remaining = int(kvp_list[f"{config.raw}_BLOCK_SIZE"])
            while bytes_remaining:
                data = file.read(min(READ_SIZE, bytes_remaining))
                if not data:
                    raise IOError("Reached end of file")
                bytes_remaining -= len(data)
                sys.stdout.buffer.write(data)

        if config.channels:
            root = lxml.etree.fromstring(xmlstr)
            for node in root.findall("./{*}Data/{*}Channel/{*}Identifier"):
                print(node.text)


if __name__ == "__main__":
    main()
