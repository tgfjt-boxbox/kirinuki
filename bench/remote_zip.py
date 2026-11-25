"""List or extract entries of a remote zip via HTTP Range requests, without downloading the archive.

usage:
  python3 -I bench/remote_zip.py list URL [GLOB]
  python3 -I bench/remote_zip.py get URL OUTDIR NAME [NAME ...]
"""
import fnmatch
import http.client
import io
import os
import sys
import urllib.request
import zipfile


class HttpRangeFile(io.RawIOBase):
    BLOCK = 1 << 20

    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.cache = {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            start = i * self.BLOCK
            end = min(start + self.BLOCK, self.size) - 1
            buf = bytearray()
            for _ in range(20):
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start + len(buf)}-{end}"})
                with urllib.request.urlopen(req) as r:
                    if r.status != 206:
                        raise IOError(f"server ignored Range (status {r.status})")
                    try:
                        buf += r.read()
                    except http.client.IncompleteRead as e:
                        buf += e.partial
                if len(buf) == end - start + 1:
                    break
            else:
                raise IOError(f"could not read bytes {start}-{end}")
            self.cache[i] = bytes(buf)
        return self.cache[i]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, off = divmod(self.pos, self.BLOCK)
            chunk = self._block(i)[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    cmd, url = sys.argv[1], sys.argv[2]
    zf = zipfile.ZipFile(HttpRangeFile(url))
    if cmd == "list":
        pattern = sys.argv[3] if len(sys.argv) > 3 else "*"
        for info in zf.infolist():
            if fnmatch.fnmatch(info.filename, pattern):
                print(info.file_size, info.filename)
    elif cmd == "get":
        outdir = sys.argv[3]
        for name in sys.argv[4:]:
            dest = os.path.join(outdir, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(zf.read(name))
            print(dest)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
