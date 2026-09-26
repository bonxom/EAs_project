"""Linux crash/timeout containment, not a hostile-code security sandbox."""

import ctypes
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from moh.execution.protocol import WorkerResult, decode_result


def _subreaper():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("worker containment currently requires Linux")
    # Adopt orphaned grandchildren so killing a group does not leave zombies.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot enable descendant reaping")


def _cleanup(process, pgid=None):
    target_pgid = pgid or (process.pid if process else None)
    if target_pgid is not None:
        try:
            os.killpg(target_pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            os.kill(-target_pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    if process is not None:
        try:
            process.wait(timeout=0.1)
        except (subprocess.TimeoutExpired, OSError):
            pass

    if target_pgid is not None:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                pid, _ = os.waitpid(-target_pgid, os.WNOHANG)
                if pid == 0:
                    time.sleep(0.01)
                    continue
            except (ChildProcessError, OSError):
                break
            except InterruptedError:
                continue


def run_worker(request, limits):
    try:
        source_bytes = request["source"].encode("utf-8")
    except UnicodeEncodeError:
        return WorkerResult("failed", None, "syntax")
    if len(source_bytes) > limits.source_bytes:
        return WorkerResult("failed", None, "source_limit")
    payload = json.dumps(request, allow_nan=False).encode()
    _subreaper()
    env = {
        "PATH": os.defpath,
        "PYTHONHASHSEED": str(request["seed"]),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    read_fd, write_fd = os.pipe()
    process = None
    pgid = None
    output = bytearray()
    result = bytearray()
    try:
        with (
            tempfile.TemporaryDirectory(prefix="moh-worker-") as cwd,
            selectors.DefaultSelector() as selector,
        ):
            deadline = time.monotonic() + limits.timeout_seconds
            process = subprocess.Popen(
                [sys.executable, "-m", "moh.execution.worker", str(write_fd)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                pass_fds=(write_fd,),
                start_new_session=True,
            )
            pgid = process.pid
            os.close(write_fd)
            write_fd = None
            streams = [
                (process.stdin, selectors.EVENT_WRITE, "input"),
                (process.stdout, selectors.EVENT_READ, "output"),
                (process.stderr, selectors.EVENT_READ, "output"),
                (read_fd, selectors.EVENT_READ, "result"),
            ]
            for stream, event, kind in streams:
                os.set_blocking(
                    stream if isinstance(stream, int) else stream.fileno(), False
                )
                selector.register(stream, event, kind)
            offset = 0
            error = None
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = "timeout"
                    break
                for key, _ in selector.select(remaining):
                    if key.data == "input":
                        try:
                            offset += os.write(key.fd, payload[offset : offset + 8192])
                        except BrokenPipeError:
                            offset = len(payload)
                        if offset == len(payload):
                            selector.unregister(key.fileobj)
                            process.stdin.close()
                        continue
                    target = result if key.data == "result" else output
                    limit = (
                        limits.result_bytes
                        if key.data == "result"
                        else limits.output_bytes
                    )
                    chunk = os.read(key.fd, min(8192, max(1, limit + 1 - len(target))))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    target.extend(chunk)
                    if len(target) > limit:
                        error = (
                            "result_limit" if key.data == "result" else "output_limit"
                        )
                        break
                if error:
                    break
            if not error:
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    error = "timeout"
                else:
                    if process.returncode:
                        error = "process_exit"

            _cleanup(process, pgid)

            captured = bytes(output[: limits.output_bytes]).decode(
                "utf-8", errors="replace"
            )
            if error:
                return WorkerResult("failed", None, error, captured)
            try:
                decoded = decode_result(
                    result, request["id"], len(request["coordinates"])
                )
            except ValueError:
                return WorkerResult("failed", None, "protocol", captured)
            return replace(decoded, output=captured)
    finally:
        if process is not None:
            _cleanup(process, pgid)
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
        try:
            os.close(read_fd)
        except OSError:
            pass
        if write_fd is not None:
            try:
                os.close(write_fd)
            except OSError:
                pass
