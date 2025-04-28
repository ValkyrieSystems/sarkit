import datetime
import filecmp
import io
import logging
import pathlib
import tempfile

import pytest

import sarkit._nitf_io


def test_string_ascii_conv(caplog):
    conv = sarkit._nitf_io.StringAscii("test", 5)
    assert conv.to_bytes("") == b"     "
    assert conv.to_bytes("abc") == b"abc  "
    assert conv.to_bytes("abcde") == b"abcde"
    with caplog.at_level(logging.WARNING, logger="sarkit._nitf_io"):
        assert conv.to_bytes("abcdef") == b"abcde"
        assert len(caplog.records) == 1
        assert "truncated" in caplog.records[0].message
    with pytest.raises(ValueError):
        conv.to_bytes("\N{LATIN CAPITAL LETTER O WITH STROKE}")

    assert conv.from_bytes(b"     ") == ""
    assert conv.from_bytes(b"abc  ") == "abc"
    assert conv.from_bytes(b"abcde") == "abcde"
    with pytest.raises(ValueError):
        conv.from_bytes(b"\xd8")


def test_string_iso_conv(caplog):
    conv = sarkit._nitf_io.StringISO8859_1("test", 5)
    assert conv.to_bytes("") == b"     "
    assert conv.to_bytes("abc") == b"abc  "
    assert conv.to_bytes("abcde") == b"abcde"
    with caplog.at_level(logging.WARNING, logger="sarkit._nitf_io"):
        assert conv.to_bytes("abcdef") == b"abcde"
        assert len(caplog.records) == 1
        assert "truncated" in caplog.records[0].message
    assert conv.to_bytes("\N{LATIN CAPITAL LETTER O WITH STROKE}") == b"\xd8    "

    assert conv.from_bytes(b"     ") == ""
    assert conv.from_bytes(b"abc  ") == "abc"
    assert conv.from_bytes(b"abcde") == "abcde"
    assert conv.from_bytes(b"\xd8") == "\N{LATIN CAPITAL LETTER O WITH STROKE}"


def test_string_utf8_conv(caplog):
    conv = sarkit._nitf_io.StringUtf8("test", 5)
    assert conv.to_bytes("") == b"     "
    assert conv.to_bytes("abc") == b"abc  "
    assert conv.to_bytes("abcde") == b"abcde"
    with caplog.at_level(logging.WARNING, logger="sarkit._nitf_io"):
        assert conv.to_bytes("abcdef") == b"abcde"
        assert len(caplog.records) == 1
        assert "truncated" in caplog.records[0].message
    assert conv.to_bytes("\N{LATIN CAPITAL LETTER O WITH STROKE}") == b"\xc3\x98   "

    assert conv.from_bytes(b"     ") == ""
    assert conv.from_bytes(b"abc  ") == "abc"
    assert conv.from_bytes(b"abcde") == "abcde"
    assert conv.from_bytes(b"\xc3\x98") == "\N{LATIN CAPITAL LETTER O WITH STROKE}"


def test_bytes_conv():
    conv = sarkit._nitf_io.Bytes("test", 5)
    assert conv.to_bytes(b"asdf") == b"asdf"
    assert conv.from_bytes(b"asdf") == b"asdf"


def test_integer_conv(caplog):
    conv = sarkit._nitf_io.Integer("test", 5)
    assert conv.to_bytes(0) == b"00000"
    assert conv.to_bytes(10) == b"00010"
    assert conv.to_bytes(-123) == b"-0123"
    with caplog.at_level(logging.WARNING, logger="sarkit._nitf_io"):
        assert conv.to_bytes(123456) == b"12345"
        assert len(caplog.records) == 1
        assert "truncated" in caplog.records[0].message

    assert conv.from_bytes(b"00000") == 0
    assert conv.from_bytes(b"00010") == 10
    assert conv.from_bytes(b"-0123") == -123


def test_rgb_conv():
    conv = sarkit._nitf_io.RGB("test", 3)
    assert conv.to_bytes((1, 2, 3)) == b"\01\02\03"
    assert conv.from_bytes(b"\01\02\03") == (1, 2, 3)


def test_segmentlist():
    seglist = sarkit._nitf_io.SegmentList(
        "test", lambda n: sarkit._nitf_io.ImageSegment(n, None), minimum=2, maximum=4
    )
    assert len(seglist) == 2
    seglist.set_size(4)
    assert len(seglist) == 4
    assert isinstance(seglist[-1], sarkit._nitf_io.ImageSegment)
    with pytest.raises(ValueError):
        seglist.set_size(5)
    with pytest.raises(ValueError):
        seglist.set_size(1)


def test_range_any():
    check = sarkit._nitf_io.AnyRange()
    assert check.isvalid(1)
    assert check.isvalid("")
    assert check.isvalid("a")


def test_range_minmax():
    check = sarkit._nitf_io.MinMax(-10, 10)
    assert not check.isvalid(-10.1)
    assert check.isvalid(-10)
    assert check.isvalid(0)
    assert check.isvalid(10)
    assert not check.isvalid(10.1)


def test_range_regex():
    check = sarkit._nitf_io.Regex("[abc]+")
    assert check.isvalid("aa")
    assert check.isvalid("cb")
    assert not check.isvalid("ad")
    assert not check.isvalid("")


def test_range_constant():
    check = sarkit._nitf_io.Constant("foobar")
    assert check.isvalid("foobar")
    assert not check.isvalid("foobar1")
    assert not check.isvalid("1foobar")


def test_range_enum():
    check = sarkit._nitf_io.Enum(["A", "B"])
    assert check.isvalid("A")
    assert check.isvalid("B")
    assert not check.isvalid("C")


def test_range_anyof():
    check = sarkit._nitf_io.AnyOf(
        sarkit._nitf_io.Constant("A"),
        sarkit._nitf_io.Constant("B"),
    )
    assert check.isvalid("A")
    assert check.isvalid("B")
    assert not check.isvalid("C")


def test_range_not():
    check = sarkit._nitf_io.Not(
        sarkit._nitf_io.Constant("A"),
    )
    assert not check.isvalid("A")
    assert check.isvalid("B")


def emtpy_nitf():
    ntf = sarkit._nitf_io.Nitf()
    ntf["FileHeader"]["OSTAID"].value = "Here"
    ntf["FileHeader"]["FSCLAS"].value = "U"
    return ntf


def add_imseg(ntf):
    ntf["FileHeader"]["NUMI"].value += 1
    idx = ntf["FileHeader"]["NUMI"].value - 1
    ntf["ImageSegments"][idx]["SubHeader"]["IID1"].value = "Unit Test"
    ntf["ImageSegments"][idx]["SubHeader"]["IDATIM"].value = datetime.datetime(
        1955, 11, 5
    ).strftime("%Y%m%d%H%M%S")
    ntf["ImageSegments"][idx]["SubHeader"]["ISCLAS"].value = "U"
    ntf["ImageSegments"][idx]["SubHeader"]["PVTYPE"].value = "INT"
    ntf["ImageSegments"][idx]["SubHeader"]["IREP"].value = "MONO"
    ntf["ImageSegments"][idx]["SubHeader"]["ICAT"].value = "SAR"
    ntf["ImageSegments"][idx]["SubHeader"]["ABPP"].value = 8
    ntf["ImageSegments"][idx]["SubHeader"]["IC"].value = "NC"
    ntf["ImageSegments"][idx]["SubHeader"]["NBANDS"].value = 1
    ntf["ImageSegments"][idx]["SubHeader"]["IREPBAND00001"].value = "M"
    ntf["ImageSegments"][idx]["SubHeader"]["IMODE"].value = "B"
    ntf["ImageSegments"][idx]["SubHeader"]["NBPR"].value = 1
    ntf["ImageSegments"][idx]["SubHeader"]["NBPC"].value = 1
    ntf["ImageSegments"][idx]["SubHeader"]["NPPBH"].value = 30
    ntf["ImageSegments"][idx]["SubHeader"]["NPPBV"].value = 20
    ntf["ImageSegments"][idx]["SubHeader"]["NROWS"].value = 20
    ntf["ImageSegments"][idx]["SubHeader"]["NCOLS"].value = 30
    ntf["ImageSegments"][idx]["SubHeader"]["NBPP"].value = 8
    ntf["ImageSegments"][idx]["SubHeader"]["ILOC"].value = (0, 0)
    ntf["ImageSegments"][idx]["Data"].size = 20 * 30
    return ntf


def check_roundtrip(ntf):
    ntf.finalize()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        orig = tmpdir / "orig.ntf"
        second = tmpdir / "copy.ntf"
        with orig.open("wb") as fd:
            ntf.dump(fd)
        ntf2 = sarkit._nitf_io.Nitf()
        with orig.open("rb") as fd:
            ntf2.load(fd)
        assert ntf == ntf2
        with second.open("wb") as fd:
            ntf2.dump(fd)

        assert filecmp.cmp(orig, second)
    return ntf2


def test_fileheader(capsys):
    ntf = emtpy_nitf()
    header = ntf["FileHeader"]
    assert "UDHOFL" not in header
    header["UDHDL"].value = 10
    assert "UDHOFL" in header
    header["UDHD"].value = b"1234567"  # UDHDL - 3

    assert "XHDLOFL" not in header
    assert "XHD" not in header
    header["XHDL"].value = 10
    assert "XHDLOFL" in header
    header["XHD"].value = b"1234567"  # XHDL - 3

    check_roundtrip(ntf)

    assert len(ntf["ImageSegments"]) == 0
    header["NUMI"].value = 2
    assert len(ntf["ImageSegments"]) == 2
    assert "LI002" in header
    header["NUMI"].value = 1
    header["LI001"].value = 10
    assert "LI002" not in header
    assert len(ntf["ImageSegments"]) == 1
    assert "LISH002" not in header
    header["NUMI"].value = 0
    assert len(ntf["ImageSegments"]) == 0
    assert "LI001" not in header
    assert "LISH001" not in header

    assert len(ntf["GraphicSegments"]) == 0
    header["NUMS"].value = 2
    assert len(ntf["GraphicSegments"]) == 2
    header["NUMS"].value = 1
    header["LS001"].value = 10
    header["LSSH001"].value = 300
    assert len(ntf["GraphicSegments"]) == 1
    # leave 1 segment to test placeholder logic
    ntf["GraphicSegments"][0]["SubHeader"].value = b"\0" * 300

    assert len(ntf["TextSegments"]) == 0
    header["NUMT"].value = 2
    assert len(ntf["TextSegments"]) == 2
    header["NUMT"].value = 1
    header["LT001"].value = 10
    header["LTSH001"].value = 300
    assert len(ntf["TextSegments"]) == 1
    # leave 1 segment to test placeholder logic
    ntf["TextSegments"][0]["SubHeader"].value = b"\0" * 300

    assert len(ntf["DESegments"]) == 0
    header["NUMDES"].value = 2
    assert len(ntf["DESegments"]) == 2
    header["NUMDES"].value = 1
    header["LD001"].value = 10
    assert len(ntf["DESegments"]) == 1
    header["NUMDES"].value = 0
    assert len(ntf["DESegments"]) == 0
    assert "LD001" not in header
    assert "LDSH001" not in header

    assert len(ntf["RESegments"]) == 0
    header["NUMRES"].value = 2
    assert len(ntf["RESegments"]) == 2
    header["NUMRES"].value = 1
    header["LRE001"].value = 10
    header["LRESH001"].value = 300
    assert len(ntf["RESegments"]) == 1
    ntf["RESegments"][0]["SubHeader"].value = b"\0" * 300
    # leave 1 segment to test placeholder logic

    ntf.finalize()
    ntf.print()
    captured = capsys.readouterr()
    assert "GraphicSegment" in captured.out
    assert "TextSegment" in captured.out
    assert "RESegment" in captured.out


def test_imseg():
    ntf = emtpy_nitf()
    add_imseg(ntf)
    check_roundtrip(ntf)
    add_imseg(ntf)
    assert ntf["FileHeader"]["NUMI"].value == 2
    check_roundtrip(ntf)
    subheader = ntf["ImageSegments"][0]["SubHeader"]
    assert "IGEOLO" not in subheader
    subheader["ICORDS"].value = "G"
    assert "IGEOLO" in subheader

    assert "ICOM1" not in subheader
    subheader["NICOM"].value = 1
    subheader["ICOM1"].value = "This is a comment"

    # Change number of bands
    assert subheader["NBANDS"].value == 1
    subheader["NBANDS"].value = 2
    assert "NELUT00001" not in subheader
    assert "NELUT00002" not in subheader
    subheader["NLUTS00002"].value = 1
    assert "NELUT00001" not in subheader
    assert "NELUT00002" in subheader
    subheader["NELUT00002"].value = 4
    assert subheader["LUTD000021"].size == 4
    subheader["LUTD000021"].value = b"\0" * 4

    check_roundtrip(ntf)

    assert "XBANDS" not in subheader
    subheader["NBANDS"].value = 0
    assert "XBANDS" in subheader
    assert "IREPBAND00001" not in subheader
    assert "IREPBAND00010" not in subheader
    subheader["XBANDS"].value = 10
    assert "IREPBAND00001" in subheader
    assert "IREPBAND00010" in subheader

    check_roundtrip(ntf)

    assert "COMRAT" not in subheader
    subheader["IC"].value = "C7"
    assert "COMRAT" in subheader

    assert "UDOFL" not in subheader
    assert "UDID" not in subheader
    subheader["UDIDL"].value = 100
    assert "UDOFL" in subheader
    assert "UDID" in subheader
    assert subheader["UDID"].size == 100 - 3

    assert "IXSOFL" not in subheader
    assert "IXSHD" not in subheader
    subheader["IXSHDL"].value = 100
    assert "IXSOFL" in subheader
    assert "IXSHD" in subheader
    assert subheader["IXSHD"].size == 100 - 3


def test_deseg():
    ntf = emtpy_nitf()
    assert ntf["FileHeader"]["NUMDES"].value == 0
    assert len(ntf["DESegments"]) == 0
    ntf["FileHeader"]["NUMDES"].value += 1
    assert len(ntf["DESegments"]) == 1

    ntf["DESegments"][0]["DESDATA"].size = 10

    subheader = ntf["DESegments"][0]["SubHeader"]
    subheader["DESID"].value = "mydesid"
    subheader["DESVER"].value = 1
    subheader["DESCLAS"].value = "U"
    assert "DESSHF" not in subheader
    subheader["DESSHL"].value = 10
    assert subheader["DESSHF"].size == 10
    subheader["DESSHF"].value = "abcd"

    check_roundtrip(ntf)

    # Test XML_DATA_CONTENT (see STDI-0002 Volume 2 Appendix F)
    assert isinstance(subheader["DESSHF"], sarkit._nitf_io.Field)
    subheader["DESID"].value = "XML_DATA_CONTENT"
    subheader["DESVER"].value = 1

    subheader["DESSHL"].value = 0
    assert isinstance(subheader["DESSHF"], sarkit._nitf_io.XmlDataContentSubheader)
    assert len(subheader["DESSHF"]) == 0

    with pytest.raises(ValueError):
        subheader["DESSHL"].value = 1  # must exactly match a length of fields

    subheader["DESSHL"].value = 5
    assert "DESCRC" in subheader["DESSHF"]
    assert "DESSHFT" not in subheader["DESSHF"]

    subheader["DESSHL"].value = 283
    assert "DESCRC" in subheader["DESSHF"]
    assert "DESSHFT" in subheader["DESSHF"]
    assert "DESSHLPG" not in subheader["DESSHF"]

    subheader["DESSHL"].value = 773
    assert "DESCRC" in subheader["DESSHF"]
    assert "DESSHFT" in subheader["DESSHF"]
    assert "DESSHLPG" in subheader["DESSHF"]
    assert "DESSHABS" in subheader["DESSHF"]

    # Fill out fields that don't have default values
    subheader["DESSHF"]["DESSHFT"].value = "XML"
    subheader["DESSHF"]["DESSHDT"].value = "1955-11-05"
    subheader["DESSHF"]["DESSHSD"].value = "2015-10-21"
    check_roundtrip(ntf)


def test_field(caplog):
    callbackinfo = {"called": False}

    def callback(fld):
        callbackinfo["called"] = True

    field = sarkit._nitf_io.Field(
        "MyField",
        "Description",
        5,
        sarkit._nitf_io.BCSN,
        sarkit._nitf_io.MinMax(10, 100),
        sarkit._nitf_io.Integer,
        setter_callback=callback,
    )
    assert not callbackinfo["called"]
    field.size = 6
    assert callbackinfo["called"]
    field.value = 50
    assert field.isvalid()
    assert field.value == 50
    assert field.encoded_value == b"000050"
    field.value = 1
    assert not field.isvalid()

    with pytest.raises(ValueError):
        stream = io.BytesIO(b"abcdefghi")
        field.load(stream)

    field = sarkit._nitf_io.Field(
        "MyField",
        "Description",
        5,
        sarkit._nitf_io.BCSN,
        sarkit._nitf_io.AnyRange(),
        sarkit._nitf_io.StringUtf8,
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sarkit._nitf_io"):
        stream = io.BytesIO(b"abcdefghi")
        field.load(stream)
        assert len(caplog.records) == 1
        assert "Invalid" in caplog.records[0].message

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sarkit._nitf_io"):
        field.value = "abc"
        assert len(caplog.records) == 1
        assert "Invalid" in caplog.records[0].message

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sarkit._nitf_io"):
        sarkit._nitf_io.Field(
            "MyField",
            "Description",
            5,
            sarkit._nitf_io.BCSN,
            sarkit._nitf_io.AnyRange(),
            sarkit._nitf_io.StringUtf8,
            default="abc",
        )
        assert len(caplog.records) == 1
        assert "Invalid" in caplog.records[0].message


def test_binaryplaceholder():
    bp = sarkit._nitf_io.BinaryPlaceholder("placeholder", 10)
    initial_data = b"0123456789"
    stream = io.BytesIO(initial_data)
    start = stream.tell()
    bp.dump(stream)
    stop = stream.tell()
    assert stop - start == 10
    assert stream.getvalue() == initial_data


def test_clevel():
    ntf = emtpy_nitf()
    ntf["FileHeader"]["NUMI"].value = 2
    ntf["ImageSegments"][0]["SubHeader"]["IDLVL"].value = 1
    ntf["ImageSegments"][0]["SubHeader"]["IALVL"].value = 0
    ntf["ImageSegments"][0]["SubHeader"]["ILOC"].value = (500, 500)
    ntf["ImageSegments"][0]["SubHeader"]["NROWS"].value = 1000
    ntf["ImageSegments"][0]["SubHeader"]["NCOLS"].value = 1000

    ntf["ImageSegments"][1]["SubHeader"]["IDLVL"].value = 2
    ntf["ImageSegments"][1]["SubHeader"]["IALVL"].value = 1
    ntf["ImageSegments"][1]["SubHeader"]["ILOC"].value = (0, 0)
    ntf["ImageSegments"][1]["SubHeader"]["NROWS"].value = 1
    ntf["ImageSegments"][1]["SubHeader"]["NCOLS"].value = 1
    assert ntf._clevel_ccs_extent() == 3
    assert ntf._clevel_image_size() == 3
    ntf["ImageSegments"][1]["SubHeader"]["ILOC"].value = (2000, 2000)
    assert ntf._clevel_ccs_extent() == 5
    assert ntf._clevel_image_size() == 3
    ntf["ImageSegments"][1]["SubHeader"]["NROWS"].value = 1
    ntf["ImageSegments"][1]["SubHeader"]["NCOLS"].value = 1000
    assert ntf._clevel_ccs_extent() == 5
    assert ntf._clevel_image_size() == 3
    ntf["ImageSegments"][1]["SubHeader"]["NROWS"].value = 8000
    ntf["ImageSegments"][1]["SubHeader"]["NCOLS"].value = 1
    assert ntf._clevel_ccs_extent() == 6
    assert ntf._clevel_image_size() == 5
    ntf["ImageSegments"][1]["SubHeader"]["NROWS"].value = 1
    ntf["ImageSegments"][1]["SubHeader"]["NCOLS"].value = 65000
    assert ntf._clevel_ccs_extent() == 7
    assert ntf._clevel_image_size() == 6
    ntf["ImageSegments"][1]["SubHeader"]["NROWS"].value = 1
    ntf["ImageSegments"][1]["SubHeader"]["NCOLS"].value = 99_999_998
    assert ntf._clevel_ccs_extent() == 9
    assert ntf._clevel_image_size() == 7

    ntf["FileHeader"]["FL"].value = 100
    assert ntf._clevel_file_size() == 3

    ntf["FileHeader"]["FL"].value = 99_999_999_999
    assert ntf._clevel_file_size() == 9

    ntf["ImageSegments"][0]["SubHeader"]["NPPBH"].value = 0
    ntf["ImageSegments"][0]["SubHeader"]["NPPBV"].value = 0
    ntf["ImageSegments"][1]["SubHeader"]["NPPBH"].value = 0
    ntf["ImageSegments"][1]["SubHeader"]["NPPBV"].value = 0
    assert ntf._clevel_image_blocking() == 3
    ntf["ImageSegments"][1]["SubHeader"]["NPPBH"].value = 3000
    ntf["ImageSegments"][1]["SubHeader"]["NPPBV"].value = 3000
    assert ntf._clevel_image_blocking() == 5
