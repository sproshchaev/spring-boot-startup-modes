#!/usr/bin/env python3
"""Prepare one startup mode (training run, checkpoint) and measure it N times.

Metrics per run:
  ttfr_ms       time from process launch to the first HTTP 200 on /products
  first_req_ms  latency of that first successful request
  second_req_ms latency of the next request
  started_s     value from the Spring log line "Started/Restored ... in X seconds"
  rss_mb        resident memory of the listening process after the first request
"""
import argparse
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PORT = 8080
BASE = f"http://127.0.0.1:{PORT}"
HEAP = ["-Xmx256m"]
JAR = ["-jar", "app.jar"]
EXIT_ON_REFRESH = "-Dspring.context.exit=onRefresh"
STARTED = re.compile(r"(?:Started|Restored) \w+ in ([\d.]+) seconds")


@dataclass(frozen=True)
class Trained:
    """A mode that needs a training run producing `artifact` before the measured runs."""
    artifact: str
    train: list
    run: list
    workload: bool = False   # train on a running service instead of exiting at onRefresh
    restore: bool = False    # measured run restores a checkpoint, no jar on the command line


TRAINED = {
    "cds": Trained("app.jsa", ["-XX:ArchiveClassesAtExit=app.jsa", EXIT_ON_REFRESH],
                   ["-XX:SharedArchiveFile=app.jsa"]),
    "aot": Trained("app.aot", ["-XX:AOTCacheOutput=app.aot", EXIT_ON_REFRESH],
                   ["-XX:AOTCache=app.aot"]),
    "aot-train": Trained("app-train.aot", ["-XX:AOTCacheOutput=app-train.aot"],
                         ["-XX:AOTCache=app-train.aot"], workload=True),
    "crac": Trained("cr", ["-XX:CRaCCheckpointTo=cr", "-Dspring.context.checkpoint=onRefresh"],
                    ["-XX:CRaCRestoreFrom=cr"], restore=True),
}
MODES = ["jvm", *TRAINED, "native"]


def java_bin():
    home = os.environ.get("JAVA_HOME")
    return str(Path(home) / "bin" / "java") if home else "java"


def port_busy():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def wait_until(cond, timeout, step=0.05):
    deadline = time.perf_counter() + timeout
    while not cond():
        if time.perf_counter() > deadline:
            return False
        time.sleep(step)
    return True


def wait_port_free():
    if not wait_until(lambda: not port_busy(), 20):
        sys.exit(f"port {PORT} still busy")


def listening_pid():
    out = subprocess.run(["lsof", "-ti", f"tcp:{PORT}", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout.split()
    if not out:
        sys.exit(f"no process listens on port {PORT}, is lsof installed?")
    return int(out[0])


def rss_mb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return round(int(out) / 1024, 1)


def get(path="/products"):
    """GET and return (latency_ms, body)."""
    t = time.perf_counter()
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        body = r.read()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}")
    return (time.perf_counter() - t) * 1000, body


def launch(cmd, cwd, log):
    return subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def kill_launched(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def wait_ready(proc, t0, timeout=90):
    """Poll the port every millisecond, then GET; return (ttfr_ms, first_req_ms)."""
    while True:
        if proc.poll() is not None:
            sys.exit(f"process exited with code {proc.returncode} before answering")
        if time.perf_counter() - t0 > timeout:
            kill_launched(proc)
            sys.exit("no answer within timeout")
        if port_busy():
            try:
                first, _ = get()
                return (time.perf_counter() - t0) * 1000, first
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
        time.sleep(0.001)


def stop(proc, pid, timeout=20):
    """Stop the server that listens on the port; after a CRaC restore it is not `proc` itself."""
    os.kill(pid, signal.SIGTERM)
    if not wait_until(lambda: not port_busy(), timeout):
        os.kill(pid, signal.SIGKILL)
    try:
        # with -XX:AOTCacheOutput the launched JVM still assembles the cache after the server is gone
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_launched(proc)
    wait_port_free()


def train(mode, java, app, log_path):
    """Run the training step of a trained mode and check that it produced the artifact."""
    m = TRAINED[mode]
    target = app / m.artifact
    shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink(missing_ok=True)
    cmd = [java, *HEAP, *m.train, *JAR]
    wait_port_free()
    with open(log_path, "w") as log:
        proc = launch(cmd, app, log)
        if m.workload:
            wait_ready(proc, time.perf_counter())
            for _ in range(300):
                get()
            stop(proc, listening_pid(), timeout=180)
        else:
            proc.wait()
    if not target.exists() or (target.is_dir() and not any(target.iterdir())):
        sys.exit(f"{mode}: training step (exit code {proc.returncode}) did not produce {m.artifact}, "
                 f"see {log_path}")


def command(mode, app, native):
    """Return the command for measured runs, after any training or checkpoint step."""
    java = java_bin()
    if mode == "jvm":
        return [java, *HEAP, *JAR]
    if mode == "native":
        return [str(native), *HEAP]
    train(mode, java, app, app / "logs" / f"{mode}-train.log")
    m = TRAINED[mode]
    return [java, *m.run] if m.restore else [java, *HEAP, *m.run, *JAR]


def measure_once(cmd, app, log_path):
    wait_port_free()
    with open(log_path, "w") as log:
        t0 = time.perf_counter()
        proc = launch(cmd, app, log)
        ttfr, first = wait_ready(proc, t0)
        second, _ = get()
        pid = listening_pid()
        mem = rss_mb(pid)
        info = json.loads(get("/actuator/info")[1])["java"]
        stop(proc, pid)
    found = STARTED.search(Path(log_path).read_text())
    runtime = f"{info['runtime']['name']} {info['runtime']['version']}, {info['jvm']['name']}"
    return {"ttfr_ms": round(ttfr), "first_req_ms": round(first, 1),
            "second_req_ms": round(second, 1), "rss_mb": mem,
            "started_s": float(found.group(1)) if found else None}, runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("--app", default="build/app", help="extracted application directory")
    ap.add_argument("--native", default="target/app", help="native executable")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    app, native = Path(a.app).resolve(), Path(a.native).resolve()
    (app / "logs").mkdir(parents=True, exist_ok=True)
    cmd = command(a.mode, app, native)
    measured = [measure_once(cmd, app, app / "logs" / f"{a.mode}-{i}.log") for i in range(a.runs)]
    runs = [r for r, _ in measured]
    median = {k: statistics.median(r[k] for r in runs if r[k] is not None)
              for k in runs[0] if any(r[k] is not None for r in runs)}
    result = {"mode": a.mode, "runtime": measured[0][1], "runs": a.runs, "median": median, "raw": runs}
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, f"{a.mode}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(a.mode, json.dumps(median))


if __name__ == "__main__":
    main()
