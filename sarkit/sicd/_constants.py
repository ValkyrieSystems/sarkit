import importlib.resources
import importlib.resources.abc
from typing import Any, Final, TypedDict

import numpy as np

SPECIFICATION_IDENTIFIER: Final[str] = (
    "SICD Volume 1 Design & Implementation Description Document"
)

SCHEMA_DIR = importlib.resources.files("sarkit.sicd.schemas")


class VersionInfoType(TypedDict):
    version: str
    date: str
    schema: importlib.resources.abc.Traversable


# Keys must be in ascending order
VERSION_INFO: Final[dict[str, VersionInfoType]] = {
    "urn:SICD:1.1.0": {
        "version": "1.1",
        "date": "2014-09-30T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.1.0_2014_09_30.xsd",
    },
    "urn:SICD:1.2.1": {
        "version": "1.2.1",
        "date": "2018-12-13T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.2.1_2018_12_13.xsd",
    },
    "urn:SICD:1.3.0": {
        "version": "1.3.0",
        "date": "2021-11-30T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.3.0_2021_11_30.xsd",
    },
    "urn:SICD:1.4.0": {
        "version": "1.4.0",
        "date": "2023-10-26T00:00:00Z",
        "schema": SCHEMA_DIR / "SICD_schema_V1.4.0_2024_05_01.xsd",
    },
}


PIXEL_TYPES: Final[dict[str, dict[str, Any]]] = {
    "RE32F_IM32F": {
        "bytes": 8,
        "pvtype": "R",
        "subcat": ("I", "Q"),
        "dtype": np.dtype(np.complex64),
    },
    "RE16I_IM16I": {
        "bytes": 4,
        "pvtype": "SI",
        "subcat": ("I", "Q"),
        "dtype": np.dtype([("real", np.int16), ("imag", np.int16)]),
    },
    "AMP8I_PHS8I": {
        "bytes": 2,
        "pvtype": "INT",
        "subcat": ("M", "P"),
        "dtype": np.dtype([("amp", np.uint8), ("phase", np.uint8)]),
    },
}

# Segmentation Algorithm Constants
IS_SIZE_MAX: Final[int] = 9_999_999_998
ILOC_MAX: Final[int] = 99_999
