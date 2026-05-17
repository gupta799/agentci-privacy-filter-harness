#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentci_privacy_filter.redactor import RedactorError, build_redactor


DEFAULT_FAIL_LABELS = {"PRIVATE_EMAIL", "PRIVATE_PHONE", "ACCOUNT_NUMBER", "SECRET"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail CI when private spans appear in files.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--fail-on",
        default=",".join(sorted(DEFAULT_FAIL_LABELS)),
        help="Comma-separated labels that should fail the gate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fail_labels = {label.strip() for label in args.fail_on.split(",") if label.strip()}
    redactor = build_redactor()
    failed = False

    for path in args.files:
        text = path.read_text()
        try:
            result = redactor.redact(text)
        except RedactorError as exc:
            print(f"{path}: redactor error: {exc}", file=sys.stderr)
            return 2

        matched = sorted(set(result.labels) & fail_labels)
        status = "FAIL" if matched else "PASS"
        print(
            f"{status} {path} backend={result.backend} elapsed_ms={result.elapsed_ms:.1f} "
            f"labels={','.join(result.labels) or '-'}"
        )
        failed = failed or bool(matched)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
