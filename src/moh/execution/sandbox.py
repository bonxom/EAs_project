"""Crash/timeout containment for worker processes across platforms."""

import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

from moh.execution.protocol import WorkerResult, decode_result

if sys.platform == "win32":
    import msvcrt


def _subreaper():
    if not sys.platform.startswith("linux"):
        return
    # Adopt orphaned grandchildren so killing a group does not leave zombies.
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    except (AttributeError, OSError):
        pass


def _cleanup(process):
    if process is None:
        return
    if hasattr(os, "killpg") and os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass
    try:
        process.wait(timeout=1)
    except (subprocess.TimeoutExpired, OSError):
        pass


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
    output = bytearray()
    result = bytearray()
    try:
        with tempfile.TemporaryDirectory(
            prefix="moh-worker-", ignore_cleanup_errors=True
        ) as cwd:
            deadline = time.monotonic() + limits.timeout_seconds
            if sys.platform == "win32":
                w_handle = msvcrt.get_osfhandle(write_fd)
                os.set_handle_inheritable(w_handle, True)
                child_arg = str(w_handle)
                popen_kwargs = {"close_fds": False}
            else:
                child_arg = str(write_fd)
                popen_kwargs = {"pass_fds": (write_fd,), "start_new_session": True}

            process = subprocess.Popen(
                [sys.executable, "-m", "moh.execution.worker", child_arg],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                **popen_kwargs,
            )
            os.close(write_fd)
            write_fd = None

            try:
                process.stdin.write(payload)
                process.stdin.flush()
            except BrokenPipeError:
                pass
            finally:
                process.stdin.close()

            output_lock = threading.Lock()
            output_limit_exceeded = threading.Event()

            def read_stream(stream):
                while True:
                    try:
                        chunk = stream.read(8192)
                    except (OSError, ValueError):
                        break
                    if not chunk:
                        break
                    with output_lock:
                        output.extend(chunk)
                        if len(output) > limits.output_bytes:
                            output_limit_exceeded.set()
                            break

            stdout_thread = threading.Thread(target=read_stream, args=(process.stdout,))
            stderr_thread = threading.Thread(target=read_stream, args=(process.stderr,))
            stdout_thread.start()
            stderr_thread.start()

            result_limit_exceeded = False

            def read_result_fd():
                nonlocal result_limit_exceeded
                while True:
                    try:
                        chunk = os.read(
                            read_fd, min(8192, max(1, limits.result_bytes + 1 - len(result)))
                        )
                        if not chunk:
                            break
                        result.extend(chunk)
                        if len(result) > limits.result_bytes:
                            result_limit_exceeded = True
                            break
                    except (OSError, ValueError):
                        break

            result_thread = threading.Thread(target=read_result_fd)
            result_thread.start()

            timed_out = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                if output_limit_exceeded.is_set() or result_limit_exceeded:
                    break
                try:
                    process.wait(timeout=min(0.05, max(0.001, remaining)))
                    break
                except subprocess.TimeoutExpired:
                    continue

            error = None
            if output_limit_exceeded.is_set():
                error = "output_limit"
            elif result_limit_exceeded:
                error = "result_limit"
            elif timed_out or time.monotonic() >= deadline:
                error = "timeout"
            elif process.returncode is None:
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    error = "timeout"
                else:
                    if process.returncode != 0 and not result:
                        error = "process_exit"
            elif process.returncode != 0 and not result:
                error = "process_exit"

            if error:
                _cleanup(process)

            result_thread.join(timeout=1.0)
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

            for stream in (process.stdout, process.stderr):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass

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
            _cleanup(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if read_fd is not None:
            try:
                os.close(read_fd)
            except OSError:
                pass
        if write_fd is not None:
            try:
                os.close(write_fd)
            except OSError:
                pass
