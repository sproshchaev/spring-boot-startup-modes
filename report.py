#!/usr/bin/env python3
"""Merge results/*.json into a table and check the invariants from expected.txt.

Each line of expected.txt reads "<mode>.<metric> < <mode>.<metric>".
Exit code 1 if any invariant fails, or if a mode is missing and --require-all is set.
"""
import argparse
import json
import sys
from pathlib import Path

from bench import MODES

COLUMNS = [("ttfr_ms", "до первого 200, мс"), ("started_s", "Started, с"),
           ("first_req_ms", "1-й запрос, мс"), ("second_req_ms", "2-й запрос, мс"),
           ("rss_mb", "RSS, МБ")]


def table(results):
    head = "| режим | " + " | ".join(t for _, t in COLUMNS) + " |"
    lines = [head, "|---" * (len(COLUMNS) + 1) + "|"]
    for mode in MODES:
        if mode in results:
            m = results[mode]["median"]
            lines.append(f"| {mode} | " + " | ".join(str(m.get(k, "-")) for k, _ in COLUMNS) + " |")
    return "\n".join(lines)


def check(results, expected, require_all):
    ok = True
    for line in expected.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        left, right = (s.strip() for s in line.split("<"))
        values = [results.get(mode, {}).get("median", {}).get(metric)
                  for mode, metric in (side.rsplit(".", 1) for side in (left, right))]
        if None in values:
            status = "FAIL" if require_all else "SKIP"
        else:
            status = "OK" if values[0] < values[1] else "FAIL"
        ok &= status != "FAIL"
        print(f"{status:4}  {line}   ({values[0]} vs {values[1]})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results")
    ap.add_argument("--expected", default="expected.txt")
    ap.add_argument("--require-all", action="store_true")
    a = ap.parse_args()

    results = {}
    for f in sorted(Path(a.dir).glob("*.json")):
        r = json.loads(f.read_text())
        results[r["mode"]] = r
    summary = table(results)
    runtimes = "\n".join(f"- {m}: {results[m]['runtime']}" for m in MODES if m in results)
    text = summary + "\n\n" + runtimes + "\n"
    Path(a.dir, "summary.md").write_text(text)
    print(text)
    sys.exit(0 if check(results, Path(a.expected), a.require_all) else 1)


if __name__ == "__main__":
    main()
