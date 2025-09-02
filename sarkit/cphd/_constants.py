import importlib.resources
from typing import Final

SCHEMA_DIR = importlib.resources.files("sarkit.cphd.schemas")
SECTION_TERMINATOR: Final[bytes] = b"\f\n"
DEFINED_HEADER_KEYS: Final[set] = {
    "XML_BLOCK_SIZE",
    "XML_BLOCK_BYTE_OFFSET",
    "SUPPORT_BLOCK_SIZE",
    "SUPPORT_BLOCK_BYTE_OFFSET",
    "PVP_BLOCK_SIZE",
    "PVP_BLOCK_BYTE_OFFSET",
    "SIGNAL_BLOCK_SIZE",
    "SIGNAL_BLOCK_BYTE_OFFSET",
    "CLASSIFICATION",
    "RELEASE_INFO",
}

# Keys in ascending order
VERSION_INFO: Final[dict] = {
    "http://api.nsgreg.nga.mil/schema/cphd/1.0.1": {
        "version": "1.0.1",
        "date": "2018-05-21T00:00:00Z",
        "schema": SCHEMA_DIR / "CPHD_schema_V1.0.1_2018_05_21.xsd",
    },
    "http://api.nsgreg.nga.mil/schema/cphd/1.1.0": {
        "version": "1.1.0",
        "date": "2021-11-30T00:00:00Z",
        "schema": SCHEMA_DIR / "CPHD_schema_V1.1.0_2021_11_30_FINAL.xsd",
    },
}
