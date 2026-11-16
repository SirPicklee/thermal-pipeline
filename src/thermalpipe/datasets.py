"""Reading real thermal data, including a few files from a huge split zip on the internet."""

import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def _http_size(url: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        return int(r.headers["Content-Length"])


def _http_range(url: str, start: int, end: int) -> bytes:
    """Bytes start...end (inclusive) of a file on a web server."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as r:
        return r.read()


class SplitZipReader(io.RawIOBase):
    """A read-only 'file' made of several parts (URLs) that downloads only the bytes it is asked for.

    The FLIR dataset is one 12 GB zip cut into 1 GB parts. Python's zipfile only needs the
    table of contents (at the very end) and the bytes of the files we want, so we never
    download the rest.
    """

    BLOCK = 512 * 1024                     # download in 512 kB blocks

    def __init__(self, urls, sizes=None, fetch=_http_range):
        self.urls = list(urls)
        self.fetch = fetch
        self.sizes = list(sizes) if sizes is not None else [_http_size(u) for u in self.urls]
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
        self.size = int(self.starts[-1])
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        self.bytes_downloaded = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _block(self, index: int) -> bytes:
        """One block of the full (virtual) file, possibly spanning two parts."""
        if index not in self._cache:
            start = index * self.BLOCK
            end = min(start + self.BLOCK, self.size) - 1
            chunks = []
            while start <= end:
                part = int(np.searchsorted(self.starts, start, side="right") - 1)
                part_end = min(end, int(self.starts[part + 1]) - 1)
                local = start - int(self.starts[part])
                chunks.append(self.fetch(self.urls[part], local, local + part_end - start))
                start = part_end + 1
            data = b"".join(chunks)
            self.bytes_downloaded += len(data)
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[index] = data
        return self._cache[index]

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        written = 0
        while written < n:
            index, offset = divmod(self.pos, self.BLOCK)
            chunk = self._block(index)[offset: offset + n - written]
            buffer[written: written + len(chunk)] = chunk
            written += len(chunk)
            self.pos += len(chunk)
        return written


def load_raw16(path) -> np.ndarray:
    """A 16-bit thermal TIFF as a numpy array (uint16)."""
    with Image.open(Path(path)) as img:
        return np.array(img).astype(np.uint16)