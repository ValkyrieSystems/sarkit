import filecmp
import pathlib

import numpy as np

import sarkit._nitf.nitf


def test_iq_band_interleaved_by_block(tmp_path):
    in_nitf = pathlib.Path(__file__).parent / "data/iq.nitf"
    with sarkit._nitf.nitf.NITFReader(str(in_nitf)) as reader:
        data = reader.read()
        data_raw = reader.read_raw()
        assert data_raw.ndim == data.ndim + 1
        assert data_raw.size == data.size * 2
        assert np.iscomplexobj(data)

        data_offset = reader.nitf_details.img_segment_offsets[0]
        manual_bytes = in_nitf.read_bytes()[data_offset : data_offset + data.nbytes]
        manual_data_raw = np.frombuffer(manual_bytes, dtype=data_raw.dtype).reshape(
            data_raw.shape
        )
        manual_data = (
            manual_data_raw[0] + 1j * manual_data_raw[1]
        )  # interleaved by block
        assert np.array_equal(data, manual_data)

        out_nitf = tmp_path / "out.nitf"
        writer_details = sarkit._nitf.nitf.NITFWritingDetails(
            reader.nitf_details.nitf_header,
            (sarkit._nitf.nitf.ImageSubheaderManager(reader.get_image_header(0)),),
            reader.image_segment_collections,
        )
        with sarkit._nitf.nitf.NITFWriter(
            str(out_nitf), writing_details=writer_details
        ) as writer:
            writer.write(data)
        assert filecmp.cmp(in_nitf, out_nitf, shallow=False)


def test_write_filehandle(tmp_path):
    in_nitf = pathlib.Path(__file__).parent / "data/iq.nitf"
    with sarkit._nitf.nitf.NITFReader(str(in_nitf)) as reader:
        data = reader.read()
        writer_details = sarkit._nitf.nitf.NITFWritingDetails(
            reader.nitf_details.nitf_header,
            (sarkit._nitf.nitf.ImageSubheaderManager(reader.get_image_header(0)),),
            reader.image_segment_collections,
        )

    out_nitf = tmp_path / "output.nitf"
    with out_nitf.open("wb") as fd:
        with sarkit._nitf.nitf.NITFWriter(fd, writing_details=writer_details) as writer:
            writer.write(data)

        assert not fd.closed
