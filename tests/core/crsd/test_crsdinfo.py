import subprocess

import lxml.etree


def test_noarg(example_crsdsar):
    subprocess.run(["crsdinfo", example_crsdsar], check=True)


def test_xml(example_crsdsar):
    proc = subprocess.run(
        ["crsdinfo", "-x", example_crsdsar], stdout=subprocess.PIPE, check=True
    )

    tree = lxml.etree.fromstring(proc.stdout)
    assert tree is not None


def test_channel(example_crsdsar):
    proc = subprocess.run(
        ["crsdinfo", "-c", example_crsdsar], capture_output=True, text=True, check=True
    )
    assert proc.stdout.splitlines() == ["the channel"]


def test_raw(example_crsdsar):
    proc = subprocess.run(
        ["crsdinfo", "-x", example_crsdsar], stdout=subprocess.PIPE, check=True
    )
    pretty_xml = proc.stdout
    proc = subprocess.run(
        ["crsdinfo", "--raw", "XML", example_crsdsar],
        stdout=subprocess.PIPE,
        check=True,
    )
    raw_xml = proc.stdout

    assert len(raw_xml) <= len(pretty_xml)
    assert lxml.etree.tostring(
        lxml.etree.fromstring(raw_xml), pretty_print=True
    ) == lxml.etree.tostring(lxml.etree.fromstring(pretty_xml), pretty_print=True)


def test_smart_open():
    proc = subprocess.run(
        ["crsdinfo", "-x", r"https://www.govsco.com/content/spotlight.crsd"],
        stdout=subprocess.PIPE,
        check=True,
    )

    tree = lxml.etree.fromstring(proc.stdout)
    assert tree is not None
