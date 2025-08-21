import argparse
import pathlib

from sarkit.verification._sidd_consistency import SiddConsistency

try:
    from smart_open import open
except ImportError:
    pass


def _parser():
    parser = argparse.ArgumentParser(
        description="Analyze a SIDD and display inconsistencies"
    )
    parser.add_argument("file_name", help="SIDD or SIDD XML to check")
    parser.add_argument(
        "--schema", type=pathlib.Path, help="Use a supplied schema file", default=None
    )
    SiddConsistency.add_cli_args(parser)
    return parser


def main(args=None):
    config = _parser().parse_args(args)

    with open(config.file_name, "rb") as file:
        sidd_con = SiddConsistency.from_file(file, config.schema)
    return sidd_con.run_cli(config)


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(int(main()))
