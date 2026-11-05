#!/usr/bin/env python3
"""Merge results/*.json into tables and check the invariants from expected.txt.

Each line of expected.txt reads "<mode>.<metric> < <mode>.<metric>".
Exit code 1 if any invariant fails, or if a mode is missing and --require-all is set.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from bench import MODES

COLUMNS = [("started_s", "Started / Restored, с"), ("ttfr_ms", "до первого 200, мс"),
           ("first_req_ms", "1-й GET, мс"), ("second_req_ms", "2-й GET, мс"),
           ("rss_mib", "RSS, МиБ")]
SPREAD = [c for c in COLUMNS if c[0] in ("ttfr_ms", "first_req_ms")]


def grid(head, rows):
    row = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    return "\n".join([row(["режим", *head]), "|---" * (len(head) + 1) + "|", *map(row, rows)])


def table(results):
    return grid([t for _, t in COLUMNS],
                [[m, *(r["median"].get(k, "-") for k, _ in COLUMNS)] for m, r in results.items()])


def spread(results):
    return grid([f"{t}: min / медиана / max" for _, t in SPREAD],
                [[m, *(f"{r['min'][k]} / {r['median'][k]} / {r['max'][k]}" for k, _ in SPREAD)]
                 for m, r in results.items()])


def machine():
    """CPU model, cores and RAM of the machine that ran the benchmark."""
    cpu = platform.processor() or "unknown CPU"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        # ARM64 kernels often have no "model name" line
        cpu = next((l.split(":", 1)[1].strip() for l in cpuinfo.read_text().splitlines()
                    if l.startswith("model name")), cpu)
    elif platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip() or cpu
    ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    return f"{platform.system()} {platform.machine()}, {cpu}, {os.cpu_count()} CPU, {ram:.1f} GiB RAM"


def classes(c):
    lines = [f"- {point}: {v['events']} событий загрузки, {v['unique']} уникальных имён"
             for point, v in c.items()]
    return "Классы в режиме jvm (`count_classes.py`):\n\n" + "\n".join(lines)


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

    results = {m: json.loads(f.read_text()) for m in MODES if (f := Path(a.dir, f"{m}.json")).exists()}
    sections = [table(results), "Разброс:\n\n" + spread(results),
                "\n".join(f"- {m}: {r['runtime']}" for m, r in results.items()), "Машина: " + machine()]
    if (c := Path(a.dir, "classes.json")).exists():
        sections.append(classes(json.loads(c.read_text())))
    text = "\n\n".join(sections) + "\n"
    Path(a.dir, "summary.md").write_text(text)
    print(text)
    sys.exit(0 if check(results, Path(a.expected), a.require_all) else 1)


if __name__ == "__main__":
    main()
