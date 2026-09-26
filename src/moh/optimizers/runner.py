"""Linux process containment runner for executable OptimizerProgram objects."""

import ctypes
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moh.core.models import OptimizerProgram
from moh.optimizers.protocol import decode_message, encode_message


@dataclass(frozen=True)
class ProgramLimits:
    timeout_seconds: float = 5.0
    max_message_bytes: int = 1048576  # 1 MB
    max_output_bytes: int = 65536     # 64 KB

    def __post_init__(self):
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        for limit in (self.max_message_bytes, self.max_output_bytes):
            if type(limit) is not int or limit <= 0:
                raise ValueError("byte limits must be positive integers")


def _subreaper() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("optimizer worker containment currently requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "prctl") and libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot enable descendant reaping")


def _cleanup(process: subprocess.Popen | None, pgid: int | None = None) -> None:
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


class OptimizerProgramRunner:
    def run(
        self,
        program: OptimizerProgram,
        request_handler: Callable[[dict[str, Any]], dict[str, Any]],
        limits: ProgramLimits | None = None,
    ) -> dict[str, Any]:
        if not isinstance(program, OptimizerProgram):
            raise TypeError("program must be an OptimizerProgram instance")

        lim = limits or ProgramLimits()
        _subreaper()

        env = {
            "PATH": os.defpath,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONHASHSEED": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }

        p2c_read, p2c_write = os.pipe()
        c2p_read, c2p_write = os.pipe()

        process: subprocess.Popen | None = None
        pgid: int | None = None
        output_buffer = bytearray()
        ipc_buffer = bytearray()
        final_result: dict[str, Any] | None = None
        error_code: str | None = None
        error_message: str = ""

        try:
            with (
                tempfile.TemporaryDirectory(prefix="moh-opt-worker-") as cwd,
                selectors.DefaultSelector() as selector,
            ):
                deadline = time.monotonic() + lim.timeout_seconds
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "moh.optimizers.worker",
                        str(p2c_read),
                        str(c2p_write),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                    pass_fds=(p2c_read, c2p_write),
                    start_new_session=True,
                )
                pgid = process.pid

                # Close parent's unused ends of child pipes
                os.close(p2c_read)
                p2c_read = -1
                os.close(c2p_write)
                c2p_write = -1

                # Send startup payload to child over p2c_write
                startup_bytes = (
                    json.dumps(
                        {
                            "program_id": program.id,
                            "source_code": program.source_code,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                os.write(p2c_write, startup_bytes)

                # Register selector streams
                streams = [
                    (process.stdout, selectors.EVENT_READ, "child_stdout"),
                    (process.stderr, selectors.EVENT_READ, "child_stderr"),
                    (c2p_read, selectors.EVENT_READ, "ipc_from_child"),
                ]
                for stream, event, kind in streams:
                    os.set_blocking(
                        stream if isinstance(stream, int) else stream.fileno(), False
                    )
                    selector.register(stream, event, kind)

                while selector.get_map() and final_result is None and not error_code:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        error_code = "timeout"
                        error_message = "execution timed out"
                        break

                    for key, _ in selector.select(remaining):
                        if key.data in ("child_stdout", "child_stderr"):
                            chunk = os.read(key.fd, 8192)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            if len(output_buffer) < lim.max_output_bytes:
                                space_left = lim.max_output_bytes - len(output_buffer)
                                output_buffer.extend(chunk[:space_left])

                        elif key.data == "ipc_from_child":
                            chunk = os.read(key.fd, 8192)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            ipc_buffer.extend(chunk)

                            # Process lines in ipc_buffer
                            while b"\n" in ipc_buffer:
                                line, _, rest = ipc_buffer.partition(b"\n")
                                ipc_buffer.clear()
                                ipc_buffer.extend(rest)

                                if len(line) > lim.max_message_bytes:
                                    error_code = "message_limit"
                                    error_message = "IPC message size limit exceeded"
                                    break

                                try:
                                    msg = decode_message(line.decode("utf-8"))
                                except (ValueError, TypeError) as exc:
                                    error_code = "protocol_error"
                                    error_message = f"malformed worker IPC payload: {exc}"
                                    break

                                msg_type = msg.get("type")
                                if msg_type == "optimizer_result":
                                    final_result = msg
                                    break

                                # Handle capability request ("generate" or "evaluate")
                                try:
                                    resp_data = request_handler(msg)
                                    resp_bytes = (
                                        encode_message(resp_data) + "\n"
                                    ).encode("utf-8")
                                    os.write(p2c_write, resp_bytes)
                                except Exception as exc:  # noqa: BLE001
                                    # Forward parent handler error as ErrorResponse to worker
                                    err_resp = (
                                        encode_message(
                                            {
                                                "type": "error",
                                                "request_id": msg.get(
                                                    "request_id", "req-unknown"
                                                ),
                                                "code": "handler_error",
                                                "message": str(exc),
                                            }
                                        )
                                        + "\n"
                                    ).encode("utf-8")
                                    try:
                                        os.write(p2c_write, err_resp)
                                    except OSError:
                                        pass

                if not error_code and final_result is None:
                    try:
                        process.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        error_code = "timeout"
                        error_message = "execution timed out waiting for exit"
                    else:
                        if process.returncode != 0 and final_result is None:
                            error_code = "worker_exit"
                            error_message = (
                                f"worker process exited with code {process.returncode}"
                            )

                _cleanup(process, pgid)

                captured = bytes(output_buffer[: lim.max_output_bytes]).decode(
                    "utf-8", errors="replace"
                )

                if error_code:
                    return {
                        "type": "optimizer_result",
                        "status": "failed",
                        "code": error_code,
                        "message": error_message,
                        "output": captured,
                    }

                if final_result is not None:
                    res = dict(final_result)
                    res["output"] = captured
                    return res

                return {
                    "type": "optimizer_result",
                    "status": "failed",
                    "code": "worker_exit",
                    "message": "worker terminated without returning result",
                    "output": captured,
                }

        finally:
            if process is not None:
                _cleanup(process, pgid)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            if p2c_read >= 0:
                try:
                    os.close(p2c_read)
                except OSError:
                    pass
            if p2c_write >= 0:
                try:
                    os.close(p2c_write)
                except OSError:
                    pass
            if c2p_read >= 0:
                try:
                    os.close(c2p_read)
                except OSError:
                    pass
            if c2p_write >= 0:
                try:
                    os.close(c2p_write)
                except OSError:
                    pass
