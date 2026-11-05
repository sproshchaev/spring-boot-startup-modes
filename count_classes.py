#!/usr/bin/env python3
"""Count classes the JVM loads before the service answers, on the first request and on the second.

Runs the extracted jar with -Xlog:class+load and reads the log at three points:
  ready   the port is open, no request has been sent yet
  first   after the first GET /products
  second  after the second GET /products
Each log line "[class,load] <name> source: ..." is one load event. The same name can be
loaded by different class loaders, so unique names are reported next to events.
"""
import json
import re
import time
from pathlib import Path

from bench import (APP_DIR, HEAP, JAR, OUT_DIR, READY_TIMEOUT, get, java_bin, kill_launched, launch, listening_pid,
                   stop, wait_port, wait_port_free)

LOAD = re.compile(r"\[class,load\] (\S+) source:")


def snapshot(log):
    names = LOAD.findall(log.read_text())
    return {"events": len(names), "unique": len(set(names))}


def main():
    app = Path(APP_DIR).resolve()
    log = app / "logs" / "class-load.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    wait_port_free()
    with open(app / "logs" / "count-classes.log", "w") as out:
        proc = launch([java_bin(), *HEAP, f"-Xlog:class+load:file={log}", *JAR], app, out)
        try:
            wait_port(proc, time.perf_counter() + READY_TIMEOUT)
            points = {"ready": snapshot(log)}
            get()
            points["first"] = snapshot(log)
            get()
            points["second"] = snapshot(log)
            stop(proc, listening_pid())
        finally:
            # never leave the JVM holding the port for the next CI step
            if proc.poll() is None:
                kill_launched(proc)
    Path(OUT_DIR).mkdir(exist_ok=True)
    Path(OUT_DIR, "classes.json").write_text(json.dumps(points, indent=2) + "\n")
    print(json.dumps(points))


if __name__ == "__main__":
    main()
