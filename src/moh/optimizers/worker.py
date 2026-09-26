"""Disposable Linux worker process for executing OptimizerProgram code."""

import json
import math
import os
import sys
from typing import Any, NoReturn

from moh.optimizers.protocol import (
    CapabilityError,
    decode_message,
    encode_message,
)
from moh.optimizers.proxy import ImprovementAPI


def _write_ipc_message(fd: int, msg: Any) -> None:
    if isinstance(msg, str):
        data = (msg + "\n").encode("utf-8")
    elif isinstance(msg, bytes):
        data = msg if msg.endswith(b"\n") else msg + b"\n"
    else:
        data = (encode_message(msg) + "\n").encode("utf-8")
    while data:
        written = os.write(fd, data)
        data = data[written:]


def _is_json_serializable(val: Any) -> bool:
    if val is None or type(val) is bool or isinstance(val, str):
        return True
    if isinstance(val, (int, float)):
        return math.isfinite(val)
    if isinstance(val, list):
        return all(_is_json_serializable(x) for x in val)
    if isinstance(val, dict):
        return all(
            isinstance(k, str) and _is_json_serializable(v) for k, v in val.items()
        )
    return False


def _send_result_and_exit(ipc_write_fd: int, result_dict: dict[str, Any]) -> NoReturn:
    try:
        _write_ipc_message(ipc_write_fd, result_dict)
    except Exception:  # noqa: S110, BLE001
        pass
    try:
        os.close(ipc_write_fd)
    except OSError:
        pass
    sys.exit(0)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(1)

    ipc_read_fd = int(sys.argv[1])
    ipc_write_fd = int(sys.argv[2])

    read_buffer = bytearray()

    def read_ipc_line() -> bytes:
        while b"\n" not in read_buffer:
            chunk = os.read(ipc_read_fd, 4096)
            if not chunk:
                raise CapabilityError("IPC connection closed unexpectedly by parent")
            read_buffer.extend(chunk)
        line, _, rest = read_buffer.partition(b"\n")
        read_buffer.clear()
        read_buffer.extend(rest)
        return bytes(line)

    try:
        startup_line = read_ipc_line()
        startup = json.loads(startup_line.decode("utf-8"))
        program_id = startup.get("program_id")
        source_code = startup.get("source_code")
        if not isinstance(program_id, str) or not isinstance(source_code, str):
            raise TypeError("invalid startup payload types")
    except Exception as exc:  # noqa: BLE001
        _send_result_and_exit(
            ipc_write_fd,
            {
                "type": "optimizer_result",
                "status": "failed",
                "code": "invalid_program",
                "message": f"malformed startup payload: {exc}",
            },
        )

    # 1. Compile source code
    try:
        compiled_code = compile(
            source_code, f"<optimizer_program_{program_id}>", "exec"
        )
    except SyntaxError as exc:
        _send_result_and_exit(
            ipc_write_fd,
            {
                "type": "optimizer_result",
                "status": "failed",
                "code": "invalid_program",
                "message": f"syntax error: {exc}",
            },
        )
    except Exception as exc:  # noqa: BLE001
        _send_result_and_exit(
            ipc_write_fd,
            {
                "type": "optimizer_result",
                "status": "failed",
                "code": "invalid_program",
                "message": f"compilation error: {exc}",
            },
        )

    # 2. Fresh execution namespace
    namespace: dict[str, Any] = {
        "__name__": "__optimizer_program__",
        "__doc__": None,
        "__package__": "",
    }

    # 3. Exec compiled code in child process namespace only
    try:
        exec(compiled_code, namespace, namespace)  # noqa: S102 — child only
    except Exception as exc:  # noqa: BLE001
        _send_result_and_exit(
            ipc_write_fd,
            {
                "type": "optimizer_result",
                "status": "failed",
                "code": "invalid_program",
                "message": f"execution error during definition: {exc}",
            },
        )

    # 4. Resolve and validate entrypoint
    entrypoint = namespace.get("improve_algorithm")
    if not callable(entrypoint):
        _send_result_and_exit(
            ipc_write_fd,
            {
                "type": "optimizer_result",
                "status": "failed",
                "code": "missing_entrypoint",
                "message": "improve_algorithm entrypoint missing or not callable",
            },
        )

    # 5. IPC transport for ImprovementAPI proxy
    def ipc_transport(req_data: Any) -> dict[str, Any]:
        _write_ipc_message(ipc_write_fd, req_data)
        resp_line = read_ipc_line()
        try:
            return decode_message(resp_line.decode("utf-8"))
        except (ValueError, TypeError) as exc:
            raise CapabilityError(f"malformed response from parent: {exc}") from exc

    api = ImprovementAPI(transport=ipc_transport)

    # 6. Execute improve_algorithm(api)
    try:
        result_val = entrypoint(api)
    except CapabilityError as exc:
        _send_result_and_exit(
            ipc_write_fd,
            {
                "type": "optimizer_result",
                "status": "failed",
                "code": "capability_error",
                "message": str(exc),
            },
        )
    except BaseException as exc:  # noqa: BLE001 — catch child runtime exceptions/exits
        _send_result_and_exit(
            ipc_write_fd,
            {
                "type": "optimizer_result",
                "status": "failed",
                "code": "runtime_error",
                "message": f"uncaught exception in improve_algorithm: {type(exc).__name__}: {exc}",
            },
        )

    # 7. Validate return value
    if not _is_json_serializable(result_val):
        _send_result_and_exit(
            ipc_write_fd,
            {
                "type": "optimizer_result",
                "status": "failed",
                "code": "invalid_return",
                "message": f"optimizer returned non-JSON-serializable value: {type(result_val).__name__}",
            },
        )

    _send_result_and_exit(
        ipc_write_fd,
        {
            "type": "optimizer_result",
            "status": "success",
            "result": result_val,
        },
    )


if __name__ == "__main__":
    main()
