import subprocess

import jbpy
import lxml.etree

import tests.utils


def test_noarg(example_sicd):
    subprocess.run(["sicdinfo", example_sicd], check=True)


def test_xml(example_sicd):
    proc = subprocess.run(
        ["sicdinfo", "-x", example_sicd], stdout=subprocess.PIPE, check=True
    )

    tree = lxml.etree.fromstring(proc.stdout)
    assert tree is not None


def test_channel(example_sicd):
    proc = subprocess.run(
        ["sicdinfo", "-s", example_sicd], stdout=subprocess.PIPE, check=True
    )
    assert len(proc.stdout.decode().splitlines()) == 1


def test_raw(example_sicd):
    proc = subprocess.run(
        ["sicdinfo", "-x", example_sicd], stdout=subprocess.PIPE, check=True
    )
    pretty_xml = proc.stdout
    proc = subprocess.run(
        ["sicdinfo", "--raw", "XML", example_sicd],
        stdout=subprocess.PIPE,
        check=True,
    )
    raw_xml = proc.stdout

    with example_sicd.open("rb") as file:
        jbp = jbpy.Jbp().load(file)
        assert len(raw_xml) == jbp["DataExtensionSegments"][0]["DESDATA"].get_size()

    raw_tree = lxml.etree.fromstring(raw_xml)
    lxml.etree.indent(raw_tree, "")
    pretty_tree = lxml.etree.fromstring(pretty_xml)
    lxml.etree.indent(pretty_tree, "")
    assert lxml.etree.tostring(raw_tree, method="c14n") == lxml.etree.tostring(
        pretty_tree, method="c14n"
    )


def test_smart_open(example_sicd):
    with tests.utils.static_http_server(example_sicd.parent) as server_url:
        proc = subprocess.run(
            ["sicdinfo", "-x", f"{server_url}/{example_sicd.name}"],
            stdout=subprocess.PIPE,
            check=True,
        )

        tree = lxml.etree.fromstring(proc.stdout)
        assert tree is not None
