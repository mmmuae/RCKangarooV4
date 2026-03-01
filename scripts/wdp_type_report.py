#!/usr/bin/env python3
import argparse
import pathlib
import struct
import zlib
from collections import Counter

HEADER_FMT = "<4sHHHH64s64s132s96sQQII64s"
RECORD_FMT = "<12s24sBBH"
FOOTER_FMT = "<8sI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_SIZE = struct.calcsize(RECORD_FMT)
FOOTER_SIZE = struct.calcsize(FOOTER_FMT)


def decode_cstr(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def inspect_file(path: pathlib.Path) -> dict:
    with path.open("rb") as fp:
        data = fp.read()

    if len(data) < HEADER_SIZE + FOOTER_SIZE:
        raise ValueError("truncated file")

    file_crc = 0
    header_raw = data[:HEADER_SIZE]
    file_crc = zlib.crc32(header_raw, file_crc)
    (
        magic,
        version,
        header_size,
        range_bits,
        dp_bits,
        worker_id_raw,
        session_tag_raw,
        target_pubkey_raw,
        start_offset_raw,
        chunk_seq,
        created_unix_ms,
        record_count,
        body_crc32,
        _reserved,
    ) = struct.unpack(HEADER_FMT, header_raw)

    if magic != b"WDP1":
        raise ValueError("invalid magic")
    if version != 1:
        raise ValueError(f"unsupported version {version}")
    if header_size != HEADER_SIZE:
        raise ValueError(f"invalid header size {header_size}")

    body_beg = HEADER_SIZE
    body_end = body_beg + record_count * RECORD_SIZE
    if body_end + FOOTER_SIZE != len(data):
        raise ValueError("size mismatch vs record_count")

    body_crc = 0
    types = Counter()
    for off in range(body_beg, body_end, RECORD_SIZE):
        rec_raw = data[off : off + RECORD_SIZE]
        body_crc = zlib.crc32(rec_raw, body_crc)
        file_crc = zlib.crc32(rec_raw, file_crc)
        _x, _d, t, _flags, _r = struct.unpack(RECORD_FMT, rec_raw)
        if t not in (0, 1, 2):
            raise ValueError(f"invalid type {t}")
        types[t] += 1

    footer_raw = data[body_end : body_end + FOOTER_SIZE]
    end_magic, file_crc_expected = struct.unpack(FOOTER_FMT, footer_raw)
    if end_magic[:7] != b"WDP1END":
        raise ValueError("invalid footer magic")

    file_crc = zlib.crc32(end_magic, file_crc)
    if (body_crc & 0xFFFFFFFF) != (body_crc32 & 0xFFFFFFFF):
        raise ValueError("body CRC mismatch")
    if (file_crc & 0xFFFFFFFF) != (file_crc_expected & 0xFFFFFFFF):
        raise ValueError("file CRC mismatch")

    return {
        "file": str(path),
        "record_count": int(record_count),
        "types": dict(types),
        "range_bits": int(range_bits),
        "dp_bits": int(dp_bits),
        "worker_id": decode_cstr(worker_id_raw),
        "session_tag": decode_cstr(session_tag_raw),
        "target_pubkey": decode_cstr(target_pubkey_raw),
        "start_offset": decode_cstr(start_offset_raw),
        "chunk_seq": int(chunk_seq),
        "created_unix_ms": int(created_unix_ms),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Report WDP type distribution with CRC validation")
    ap.add_argument("paths", nargs="+", help=".wdp files or directories")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        path = pathlib.Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.wdp")))
        elif path.is_file():
            files.append(path)

    if not files:
        print("no .wdp files found")
        return 1

    total = Counter()
    rec_total = 0
    for f in files:
        info = inspect_file(f)
        rec_total += info["record_count"]
        total.update(info["types"])
        print(f"{f.name}: rec={info['record_count']} types={info['types']} worker={info['worker_id']} session={info['session_tag']}")

    print(f"TOTAL files={len(files)} rec={rec_total} types={dict(total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
