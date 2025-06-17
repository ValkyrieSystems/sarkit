import importlib.resources
from typing import Final

SCHEMA_DIR = importlib.resources.files("sarkit.crsd.schemas")
SECTION_TERMINATOR: Final[bytes] = b"\f\n"
DEFINED_HEADER_KEYS: Final[set] = {
    "XML_BLOCK_SIZE",
    "XML_BLOCK_BYTE_OFFSET",
    "SUPPORT_BLOCK_SIZE",
    "SUPPORT_BLOCK_BYTE_OFFSET",
    "PPP_BLOCK_SIZE",
    "PPP_BLOCK_BYTE_OFFSET",
    "PVP_BLOCK_SIZE",
    "PVP_BLOCK_BYTE_OFFSET",
    "SIGNAL_BLOCK_SIZE",
    "SIGNAL_BLOCK_BYTE_OFFSET",
    "CLASSIFICATION",
    "RELEASE_INFO",
}

VERSION_INFO: Final[dict] = {
    "http://api.nsgreg.nga.mil/schema/crsd/1.0": {
        "version": "1.0",
        "date": "2025-02-25T00:00:00Z",
        "schema": SCHEMA_DIR / "NGA.STND.0080-2_1.0_CRSD_schema_2025_02_25.xsd",
    },
}
