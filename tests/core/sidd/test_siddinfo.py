import subprocess

import lxml.etree

import sarkit.sidd as sksidd
import tests.utils


def test_noarg(example_sidd):
    subprocess.run(["siddinfo", example_sidd], check=True)


def test_xml(example_sidd):
    proc = subprocess.run(
        ["siddinfo", "-x", example_sidd], stdout=subprocess.PIPE, check=True
    )

    tree = lxml.etree.fromstring(proc.stdout)
    assert tree is not None


def test_segments(example_sidd):
    proc = subprocess.run(
        ["siddinfo", "-s", example_sidd], stdout=subprocess.PIPE, check=True, text=True
    )
    assert len(proc.stdout.splitlines()) == 2


def test_raw_xml(example_sidd):
    proc = subprocess.run(
        ["siddinfo", "-x", example_sidd], stdout=subprocess.PIPE, check=True
    )
    pretty_xml = proc.stdout
    proc = subprocess.run(
        ["siddinfo", "--raw", "XML", example_sidd], stdout=subprocess.PIPE, check=True
    )
    raw_xml = proc.stdout

    assert len(raw_xml) <= len(pretty_xml)
    assert lxml.etree.tostring(
        lxml.etree.fromstring(raw_xml), pretty_print=True
    ) == lxml.etree.tostring(lxml.etree.fromstring(pretty_xml), pretty_print=True)


def test_raw_image(example_sidd):
    proc = subprocess.run(
        ["siddinfo", example_sidd, "--raw", "image"], stdout=subprocess.PIPE, check=True
    )
    raw_img = proc.stdout

    with example_sidd.open("rb") as file, sksidd.NitfReader(file) as reader:
        data = reader.read_image(0)

    assert data.tobytes() == raw_img


def test_smart_open(example_sidd):
    with tests.utils.static_http_server(example_sidd.parent) as server_url:
        proc = subprocess.run(
            ["siddinfo", "-x", f"{server_url}/{example_sidd.name}"],
            stdout=subprocess.PIPE,
            check=True,
        )

        tree = lxml.etree.fromstring(proc.stdout)
        assert tree is not None
