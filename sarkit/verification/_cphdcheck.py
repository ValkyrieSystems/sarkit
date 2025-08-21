import argparse
import pathlib

from sarkit.verification._cphd_consistency import CphdConsistency

try:
    from smart_open import open
except ImportError:
    pass


def _parser():
    parser = argparse.ArgumentParser(
        description="Analyze a CPHD and display inconsistencies"
    )
    parser.add_argument("file_name", help="CPHD or CPHD XML to check")
    parser.add_argument(
        "--schema",
        type=pathlib.Path,
        help="Use a supplied schema file (attempts version-specific schema if omitted)",
    )
    parser.add_argument(
        "--thorough",
        action="store_true",
        help=(
            "Run checks that may seek/read through large portions of the file. "
            "Ignored when file_name is XML"
        ),
    )
    CphdConsistency.add_cli_args(parser)
    return parser


def main(args=None):
    config = _parser().parse_args(args)
    with open(config.file_name, "rb") as f:
        cphd_con = CphdConsistency.from_file(
            file=f,
            schema=config.schema,
            thorough=config.thorough,
        )
        return cphd_con.run_cli(config)


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(int(main()))
