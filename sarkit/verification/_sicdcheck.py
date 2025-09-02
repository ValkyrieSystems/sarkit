import argparse
import pathlib

from sarkit.verification._sicd_consistency import SicdConsistency

try:
    from smart_open import open
except ImportError:
    pass


def _parser():
    parser = argparse.ArgumentParser(
        description="Analyze a SICD and display inconsistencies"
    )
    parser.add_argument("file_name", help="SICD or SICD XML to check")
    parser.add_argument(
        "--schema",
        type=pathlib.Path,
        help="Use a supplied schema file (attempts version-specific schema if omitted)",
    )
    SicdConsistency.add_cli_args(parser)
    return parser


def main(args=None):
    config = _parser().parse_args(args)
    with open(config.file_name, "rb") as f:
        sicd_con = SicdConsistency.from_file(
            file=f,
            schema=config.schema,
        )
    # file doesn't need to stay open once object is instantiated
    return sicd_con.run_cli(config)


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(int(main()))
