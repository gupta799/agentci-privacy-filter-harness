"""Restore the checked-in compressed dataset, validating its manifest checksums."""

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("data/combined"))
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise ValueError("Unexpected filename in manifest")
        output = args.directory / name
        if output.exists():
            if hashlib.sha256(output.read_bytes()).hexdigest() == expected["sha256"]:
                print(f"{name}: already verified")
                continue
            raise ValueError(f"{name}: existing file differs; refusing to overwrite it")
        with tempfile.NamedTemporaryFile(dir=args.directory, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            sha = hashlib.sha256()
            try:
                archives = expected.get("archives", [name + ".gz"])
                for archive in archives:
                    if Path(archive).name != archive or not archive.endswith(".jsonl.gz"):
                        raise ValueError("Unexpected archive filename in manifest")
                    with gzip.open(args.directory / archive, "rb") as source:
                        while chunk := source.read(1024 * 1024):
                            temporary.write(chunk)
                            sha.update(chunk)
                temporary.close()
                if sha.hexdigest() != expected["sha256"]:
                    raise ValueError(f"{name}: checksum mismatch")
                os.replace(temporary_path, output)
            finally:
                temporary_path.unlink(missing_ok=True)
        print(f"{name}: restored and verified")


if __name__ == "__main__":
    main()
