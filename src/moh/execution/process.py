"""Deadline-aware ownership and bounded NDJSON worker transport on Linux."""
from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from moh.execution.protocol import strict_json
from moh.execution.sandbox import _cleanup, _subreaper


class CandidateFailure(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Deadline:
    expires_at: float

    @classmethod
    def after(cls, seconds: float, parent: Deadline | None = None) -> Deadline:
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('deadline duration must be positive and finite')
        expires = time.monotonic() + seconds
        return cls(min(expires, parent.expires_at) if parent else expires)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def check(self) -> None:
        if not self.remaining():
            raise CandidateFailure('timeout')


@dataclass(frozen=True)
class ProgramLimits:
    timeout_seconds: float = 30.0
    source_bytes: int = 65536
    request_bytes: int = 1048576
    result_bytes: int = 1048576
    output_bytes: int = 65536
    max_callbacks: int = 100
    batch_size: int = 5

    def __post_init__(self):
        Deadline.after(self.timeout_seconds)
        for name in ('source_bytes', 'request_bytes', 'result_bytes', 'output_bytes',
                     'max_callbacks', 'batch_size'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')


class ProcessScope:
    """Registration and cancellation remain safe while descendants are closing."""
    def __init__(self, deadline, parent=None):
        self.deadline = deadline
        self.parent = parent
        self.lock = threading.RLock()
        self.processes = set()
        self.children = set()
        self.threads = set()
        self.cancelled = None
        self.closed = False
        if parent:
            with parent.lock:
                if parent.closed or parent.cancelled:
                    raise CandidateFailure(parent.cancelled or 'timeout')
                parent.children.add(self)

    def check(self):
        self.deadline.check()
        if self.cancelled or self.closed:
            raise CandidateFailure(self.cancelled or 'timeout')
        if self.parent:
            self.parent.check()

    def register(self, process):
        with self.lock:
            self.processes.add(process)
            if self.cancelled or self.closed:
                self._kill(process)
                raise CandidateFailure(self.cancelled or 'timeout')

    @staticmethod
    def _kill(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def abort(self, code='timeout'):
        with self.lock:
            self.cancelled = self.cancelled or code
            children, processes = tuple(self.children), tuple(self.processes)
        for child in children:
            child.abort(code)
        for process in processes:
            self._kill(process)

    def close(self):
        with self.lock:
            self.closed = True
        self.abort(self.cancelled or 'timeout')
        with self.lock:
            children = tuple(self.children)
        for child in children:
            child.close()
        with self.lock:
            threads = tuple(self.threads)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join()
        with self.lock:
            for process in tuple(self.processes):
                _cleanup(process)
                self.processes.discard(process)
        with self.lock:
            self.closed = True
        if self.parent:
            with self.parent.lock:
                self.parent.children.discard(self)


def encode_frame(value, limit, code):
    try:
        frame = json.dumps(value, allow_nan=False, ensure_ascii=True).encode() + b'\n'
    except (ValueError, TypeError, RecursionError) as exc:
        raise CandidateFailure('protocol') from exc
    if len(frame) > limit:
        raise CandidateFailure(code)
    return frame


class ProcessSupervisor:
    @contextmanager
    def scope(self, deadline, parent_scope=None):
        ownership = ProcessScope(deadline, parent_scope)
        try:
            ownership.check()
            yield ownership
        finally:
            ownership.close()

    def exchange(self, module, request, dispatch, *, limits, deadline, scope):
        """Dispatch is trusted parent code; its infrastructure exceptions propagate."""
        from moh.execution.optimizer_protocol import validate_request

        deadline.check()
        scope.check()
        initial = encode_frame(request, limits.request_bytes, 'request_limit')
        _subreaper()
        env = {'PATH': os.defpath, 'PYTHONHASHSEED': str(request['seed']),
               'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        read_fd, write_fd = os.pipe()
        process = None
        monitor = None
        stop = threading.Event()
        try:
            with tempfile.TemporaryDirectory(prefix='moh-program-') as cwd:
                process = subprocess.Popen(
                    [sys.executable, '-m', module, str(write_fd), str(limits.request_bytes)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    pass_fds=(write_fd,), start_new_session=True, cwd=cwd, env=env,
                )
                os.close(write_fd)
                write_fd = None
                scope.register(process)
                for fd in (read_fd, process.stdin.fileno(), process.stdout.fileno(),
                           process.stderr.fileno()):
                    os.set_blocking(fd, False)

                def drain():
                    total = 0
                    with selectors.DefaultSelector() as output:
                        for stream in (process.stdout, process.stderr):
                            output.register(stream, selectors.EVENT_READ)
                        while not scope.cancelled:
                            if not stop.is_set() and not deadline.remaining():
                                scope.abort('timeout')
                                return
                            events = output.select(0 if stop.is_set() else min(0.01, deadline.remaining()))
                            if stop.is_set() and not events:
                                return
                            for key, _ in events:
                                chunk = os.read(key.fd, 8192)
                                if not chunk:
                                    output.unregister(key.fileobj)
                                total += len(chunk)
                                if total > limits.output_bytes:
                                    scope.abort('output_limit')
                                    return

                with scope.lock:
                    scope.check()
                    monitor = threading.Thread(target=drain, name='moh-output-monitor')
                    scope.threads.add(monitor)
                    monitor.start()
                with selectors.DefaultSelector() as reader:
                    reader.register(read_fd, selectors.EVENT_READ)
                    buffer = bytearray()

                    def check():
                        scope.check()
                        deadline.check()

                    def send(frame):
                        with selectors.DefaultSelector() as writer:
                            writer.register(process.stdin, selectors.EVENT_WRITE)
                            offset = 0
                            while offset < len(frame):
                                check()
                                if writer.select(min(0.01, deadline.remaining())):
                                    try:
                                        offset += os.write(process.stdin.fileno(), frame[offset:offset+8192])
                                    except BrokenPipeError as exc:
                                        raise CandidateFailure('process_exit') from exc

                    send(initial)
                    expected_id = 1
                    callbacks = 0
                    while True:
                        check()
                        if b'\n' not in buffer:
                            if not reader.select(min(0.01, deadline.remaining())):
                                continue
                            chunk = os.read(read_fd, min(8192, limits.result_bytes + 1))
                            if not chunk:
                                check()
                                raise CandidateFailure('protocol' if buffer else 'process_exit')
                            buffer.extend(chunk)
                            if b'\n' not in buffer and len(buffer) > limits.result_bytes:
                                raise CandidateFailure('result_limit')
                            if b'\n' not in buffer:
                                continue
                        frame, _, remainder = buffer.partition(b'\n')
                        buffer = bytearray(remainder)
                        if len(frame) + 1 > limits.result_bytes:
                            raise CandidateFailure('result_limit')
                        try:
                            envelope = strict_json(frame.decode('utf-8'))
                        except (ValueError, UnicodeError) as exc:
                            raise CandidateFailure('protocol') from exc
                        operation, payload = validate_request(envelope, expected_id, request, limits)
                        expected_id += 1
                        if operation == 'finish':
                            try:
                                process.wait(timeout=deadline.remaining())
                            except subprocess.TimeoutExpired as exc:
                                raise CandidateFailure('timeout') from exc
                            stop.set()
                            monitor.join()
                            check()
                            if process.returncode:
                                raise CandidateFailure('process_exit')
                            return payload
                        callbacks += 1
                        if callbacks > limits.max_callbacks:
                            raise CandidateFailure('callback_limit')
                        value = dispatch(operation, payload, deadline)
                        check()
                        send(encode_frame({'id': envelope['id'], 'ok': True, 'value': value,
                                           'error': None}, limits.request_bytes, 'request_limit'))
        finally:
            stop.set()
            if process is not None:
                scope._kill(process)
            if monitor is not None:
                monitor.join()
                with scope.lock:
                    scope.threads.discard(monitor)
            if process is not None:
                with scope.lock:
                    if process in scope.processes:
                        _cleanup(process)
                        scope.processes.discard(process)
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
