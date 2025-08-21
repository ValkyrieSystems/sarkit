import subprocess

import pytest


@pytest.mark.parametrize("cmd", ("cphdcheck", "crsdcheck", "sicdcheck", "siddcheck"))
def test_consistency_cli(cmd):
    subprocess.run([cmd, "-h"], check=True)
