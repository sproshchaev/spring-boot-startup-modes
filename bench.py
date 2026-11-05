#!/usr/bin/env python3
"""Prepare one startup mode (training run, checkpoint) and measure it N times.

Metrics per run:
  ttfr_ms       time from process launch to the first HTTP 200 on /products
  first_req_ms  latency of that first successful request
  second_req_ms latency of the next request
  started_s     value from the Spring log line "Started/Restored ... in X seconds"
  rss_mib       resident memory (MiB) of the listening process after the first request
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
APP_DIR = "build/app"
OUT_DIR = "results"
# (pattern, seconds per unit): a checkpoint taken after refresh restores into a running context
# and logs a lifecycle restart in milliseconds instead of "Restored ... in X seconds"
START_LINES = [(re.compile(r"(?:Started|Restored) \w+ in ([\d.]+) seconds"), 1),
               (re.compile(r"restored JVM running for (\d+) ms"), 0.001)]


@dataclass(frozen=True)
class Trained:
    """A mode that needs a training run producing `artifact` before the measured runs."""
    artifact: str
    train: list
    run: list
    finish: str = "exit"     # how training ends: "exit" at onRefresh, "stop" after a workload,
                             # "checkpoint" with jcmd JDK.checkpoint after a workload
    restore: bool = False    # measured run restores a checkpoint, no jar on the command line


TRAINED = {
    "cds": Trained("app.jsa", ["-XX:ArchiveClassesAtExit=app.jsa", EXIT_ON_REFRESH],
                   ["-XX:SharedArchiveFile=app.jsa"]),
    "aot": Trained("app.aot", ["-XX:AOTCacheOutput=app.aot", EXIT_ON_REFRESH],
                   ["-XX:AOTCache=app.aot"]),
    "aot-train": Trained("app-train.aot", ["-XX:AOTCacheOutput=app-train.aot"],
                         ["-XX:AOTCache=app-train.aot"], finish="stop"),
    "crac": Trained("cr", ["-XX:CRaCCheckpointTo=cr", "-Dspring.context.checkpoint=onRefresh"],
                    ["-XX:CRaCRestoreFrom=cr"], restore=True),
    "crac-warm": Trained("cr-warm", ["-XX:CRaCCheckpointTo=cr-warm"], ["-XX:CRaCRestoreFrom=cr-warm"],
                         finish="checkpoint", restore=True),
}
TRAIN_REQUESTS = 300
MODES = ["jvm", *TRAINED, "native"]


def jdk_tool(name):
    home = os.environ.get("JAVA_HOME")
    return str(Path(home) / "bin" / name) if home else name


def java_bin():
    return jdk_tool("java")


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


def rss_mib(pid):
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


def wait_port(proc, t0, timeout=90):
    """Poll every millisecond until the port opens; exit if the process dies or time runs out."""
    deadline = t0 + timeout
    wait_until(lambda: port_busy() or proc.poll() is not None or time.perf_counter() > deadline,
               timeout, step=0.001)
    if proc.poll() is not None:
        sys.exit(f"process exited with code {proc.returncode} before answering")
    if not port_busy():
        kill_launched(proc)
        sys.exit(f"no open port within {timeout} s")


def wait_ready(proc, t0, timeout=90):
    """Wait for the port, then GET until 200; return (ttfr_ms, first_req_ms)."""
    while True:
        wait_port(proc, t0, timeout)
        try:
            first, _ = get()
            return (time.perf_counter() - t0) * 1000, first
        except (urllib.error.URLError, ConnectionError, OSError):
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


def checkpoint(proc, log, log_path):
    """Take a CRaC checkpoint with jcmd; the JVM writes it and exits on its own.
    If the checkpoint fails (open file or socket) the JVM keeps running and must not hold the port."""
    done = subprocess.run([jdk_tool("jcmd"), str(listening_pid()), "JDK.checkpoint"],
                          stdout=log, stderr=subprocess.STDOUT).returncode == 0
    try:
        proc.wait(timeout=120 if done else 5)
    except subprocess.TimeoutExpired:
        kill_launched(proc)
        sys.exit(f"JVM did not exit after JDK.checkpoint, see {log_path}")


def train(mode, java, app, log_path):
    """Run the training step of a trained mode and check that it produced the artifact."""
    m = TRAINED[mode]
    target = app / m.artifact
    shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink(missing_ok=True)
    cmd = [java, *HEAP, *m.train, *JAR]
    wait_port_free()
    with open(log_path, "w") as log:
        proc = launch(cmd, app, log)
        if m.finish == "exit":
            proc.wait()
        else:
            wait_ready(proc, time.perf_counter())
            for _ in range(TRAIN_REQUESTS):
                get()
            if m.finish == "checkpoint":
                checkpoint(proc, log, log_path)
            else:
                stop(proc, listening_pid(), timeout=180)
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


def started_s(text):
    for pattern, scale in START_LINES:
        if found := pattern.search(text):
            return round(float(found.group(1)) * scale, 3)
    return None


def measure_once(cmd, app, log_path):
    wait_port_free()
    with open(log_path, "w") as log:
        t0 = time.perf_counter()
        proc = launch(cmd, app, log)
        ttfr, first = wait_ready(proc, t0)
        second, _ = get()
        pid = listening_pid()
        mem = rss_mib(pid)
        info = json.loads(get("/actuator/info")[1])["java"]
        stop(proc, pid)
    runtime = f"{info['runtime']['name']} {info['runtime']['version']}, {info['jvm']['name']}"
    return {"ttfr_ms": round(ttfr), "first_req_ms": round(first, 1),
            "second_req_ms": round(second, 1), "rss_mib": mem,
            "started_s": started_s(Path(log_path).read_text())}, runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("--app", default=APP_DIR, help="extracted application directory")
    ap.add_argument("--native", default="target/app", help="native executable")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    app, native = Path(a.app).resolve(), Path(a.native).resolve()
    (app / "logs").mkdir(parents=True, exist_ok=True)
    cmd = command(a.mode, app, native)
    measured = [measure_once(cmd, app, app / "logs" / f"{a.mode}-{i}.log") for i in range(a.runs)]
    runs = [r for r, _ in measured]
    values = {k: v for k in runs[0] if (v := [r[k] for r in runs if r[k] is not None])}
    result = {"mode": a.mode, "runtime": measured[0][1], "runs": a.runs, "raw": runs}
    for name, f in (("median", statistics.median), ("min", min), ("max", max)):
        result[name] = {k: f(v) for k, v in values.items()}
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, f"{a.mode}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(a.mode, json.dumps(result["median"]))


if __name__ == "__main__":
    main()
