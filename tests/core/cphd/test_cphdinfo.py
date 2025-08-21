import subprocess

import lxml.etree

import tests.utils


def test_noarg(example_cphd):
    subprocess.run(["cphdinfo", example_cphd], check=True)


def test_xml(example_cphd):
    proc = subprocess.run(
        ["cphdinfo", "-x", example_cphd], stdout=subprocess.PIPE, check=True
    )

    tree = lxml.etree.fromstring(proc.stdout)
    assert tree is not None


def test_channel(example_cphd):
    proc = subprocess.run(
        ["cphdinfo", "-c", example_cphd], stdout=subprocess.PIPE, check=True
    )
    assert proc.stdout.decode().splitlines() == ["1"]


def test_raw(example_cphd):
    proc = subprocess.run(
        ["cphdinfo", "-x", example_cphd], stdout=subprocess.PIPE, check=True
    )
    pretty_xml = proc.stdout
    proc = subprocess.run(
        ["cphdinfo", "--raw", "XML", example_cphd], stdout=subprocess.PIPE, check=True
    )
    raw_xml = proc.stdout

    assert len(raw_xml) <= len(pretty_xml)
    assert lxml.etree.tostring(
        lxml.etree.fromstring(raw_xml), pretty_print=True
    ) == lxml.etree.tostring(lxml.etree.fromstring(pretty_xml), pretty_print=True)


def test_smart_open(example_cphd):
    with tests.utils.static_http_server(example_cphd.parent) as server_url:
        proc = subprocess.run(
            ["cphdinfo", "-x", f"{server_url}/{example_cphd.name}"],
            stdout=subprocess.PIPE,
            check=True,
        )

        tree = lxml.etree.fromstring(proc.stdout)
        assert tree is not None
