import importlib.resources
from typing import Final, TypedDict

import numpy as np

SPECIFICATION_IDENTIFIER: Final[str] = (
    "SIDD Volume 1 Design & Implementation Description Document"
)

SCHEMA_DIR = importlib.resources.files("sarkit.sidd.schemas")


class VersionInfoType(TypedDict):
    version: str
    date: str
    schema: importlib.resources.abc.Traversable


# Keys must be in ascending order
VERSION_INFO: Final[dict[str, VersionInfoType]] = {
    "urn:SIDD:2.0.0": {
        "version": "2.0",
        "date": "2019-05-31T00:00:00Z",
        "schema": SCHEMA_DIR / "version2" / "SIDD_schema_V2.0.0_2019_05_31.xsd",
    },
    "urn:SIDD:3.0.0": {
        "version": "3.0",
        "date": "2021-11-30T00:00:00Z",
        "schema": SCHEMA_DIR / "version3" / "SIDD_schema_V3.0.0.xsd",
    },
}


# Table 2-6 NITF 2.1 Image Sub-Header Population for Supported Pixel Type
class _PixelTypeDict(TypedDict):
    IREP: str
    IREPBANDn: list[str]
    IMODE: str
    NBPP: int
    dtype: np.dtype


PIXEL_TYPES: Final[dict[str, _PixelTypeDict]] = {
    "MONO8I": {
        "IREP": "MONO",
        "IREPBANDn": ["M"],
        "IMODE": "B",
        "NBPP": 8,
        "dtype": np.dtype(np.uint8),
    },
    "MONO8LU": {
        "IREP": "MONO",
        "IREPBANDn": ["LU"],
        "IMODE": "B",
        "NBPP": 8,
        "dtype": np.dtype(np.uint8),
    },
    "MONO16I": {
        "IREP": "MONO",
        "IREPBANDn": ["M"],
        "IMODE": "B",
        "NBPP": 16,
        "dtype": np.dtype(np.uint16),
    },
    "RGB8LU": {
        "IREP": "RGB/LUT",
        "IREPBANDn": ["LU"],
        "IMODE": "B",
        "NBPP": 8,
        "dtype": np.dtype(np.uint8),
    },
    "RGB24I": {
        "IREP": "RGB",
        "IREPBANDn": ["R", "G", "B"],
        "IMODE": "P",
        "NBPP": 8,
        "dtype": np.dtype([("R", np.uint8), ("G", np.uint8), ("B", np.uint8)]),
    },
}

LI_MAX: Final[int] = 9_999_999_998
ILOC_MAX: Final[int] = 99_999
