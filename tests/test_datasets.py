import io
import zipfile

import numpy as np
from PIL import Image

from thermalpipe.datasets import SplitZipReader, load_raw16


def test_split_zip_reader_reads_files_across_parts():
    # Build a zip in memory with a few files, then cut it into 3 uneven "parts"
    buffer = io.BytesIO()
    contents = {f"folder/file_{i}.bin": np.random.default_rng(i).bytes(300_000) for i in range(5)}
    with zipfile.ZipFile(buffer, "w") as z:
        for name, data in contents.items():
            z.writestr(name, data)
    blob = buffer.getvalue()
    cuts = [0, 400_000, 900_000, len(blob)]
    parts = {f"part{i}": blob[cuts[i]:cuts[i + 1]] for i in range(3)}

    def fake_fetch(url, start, end):          # pretends to be a web server
        return parts[url][start:end + 1]

    reader = SplitZipReader(list(parts), sizes=[len(p) for p in parts.values()], fetch=fake_fetch)
    with zipfile.ZipFile(reader) as z:
        assert sorted(z.namelist()) == sorted(contents)
        assert z.read("folder/file_3.bin") == contents["folder/file_3.bin"]


def test_load_raw16_keeps_16_bit_values(tmp_path):
    data = (np.arange(64 * 80) % 60000).astype(np.uint16).reshape(64, 80)
    Image.fromarray(data).save(tmp_path / "frame.tiff")
    assert np.array_equal(load_raw16(tmp_path / "frame.tiff"), data)