"""Run generated optimizers in a child with parent-owned LLM/evaluation services."""

import logging
import multiprocessing
import random
import time
from threading import RLock, Timer

import numpy as np

logger = logging.getLogger(__name__)


class _PipeLogHandler(logging.Handler):
    def __init__(self, connection, lock):
        super().__init__()
        self.connection = connection
        self.connection_lock = lock

    def emit(self, record):
        with self.connection_lock:
            self.connection.send(("log", record.name, record.levelno, record.getMessage()))


class _RemoteLLM:
    def __init__(self, connection, attributes, lock):
        self.connection = connection
        self.connection_lock = lock
        self.__dict__.update(attributes)

    def request(self, name, args, kwargs):
        with self.connection_lock:
            self.connection.send(("request", name, args, kwargs))
            status, value = self.connection.recv()
        if status == "error":
            raise RuntimeError(value)
        return value

    def prompt(self, *args, **kwargs):
        return self.request("prompt", args, kwargs)

    def prompt_batch(self, *args, **kwargs):
        return self.request("prompt_batch", args, kwargs)


def _worker(connection, code, arguments, seed):
    try:
        root_logger = logging.getLogger()
        lock = RLock()
        root_logger.handlers = [_PipeLogHandler(connection, lock)]
        root_logger.setLevel(logging.INFO)
        random.seed(seed)
        np.random.seed(seed)
        namespace = {"__name__": "generated_improver"}
        exec(compile(code, "<candidate_improver>", "exec"), namespace)  # noqa: S102 -- isolated child
        improver = namespace.get("improve_algorithm")
        if not callable(improver):
            raise TypeError("Candidate must define a callable improve_algorithm")
        if arguments is None:
            result = None
        else:
            population, attributes, function_format, task = arguments
            llm = _RemoteLLM(connection, attributes, lock)

            def utility(*args, **kwargs):
                return llm.request("utility", args, kwargs)

            result = improver(population, utility, llm, function_format, task)
        connection.send(("result", result))
    except BaseException as error:  # noqa: BLE001 -- includes generated SystemExit
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


class GeneratedImprover:
    def __init__(self, code, timeout=3600, seed=0):
        self.code = code
        self.timeout = timeout
        self.seed = seed

    def validate(self):
        return self._run(None, {})

    def __call__(self, population, utility, language_model, function_format, task):
        attributes = {
            name: getattr(language_model, name)
            for name in ("batch_size", "model", "temperature")
            if hasattr(language_model, name)
        }
        arguments = (population, attributes, function_format, task)
        services = {
            "utility": utility,
            "prompt": language_model.prompt,
            "prompt_batch": language_model.prompt_batch,
        }
        return self._run(arguments, services)

    def _run(self, arguments, services):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_worker, args=(child, self.code, arguments, self.seed))
        process.start()
        child.close()
        timeout = min(self.timeout, 90) if arguments is None else self.timeout
        deadline = time.monotonic() + timeout
        # Terminate generated code even while the parent is busy evaluating a callback.
        watchdog = Timer(timeout, process.terminate)
        watchdog.daemon = True
        watchdog.start()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Generated improver exceeded {timeout}s timeout")
                if not parent.poll(min(remaining, 0.1)):
                    if not process.is_alive():
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Generated improver exceeded {timeout}s timeout")
                        raise RuntimeError(f"Generated improver exited with code {process.exitcode}")
                    continue
                try:
                    message = parent.recv()
                except EOFError as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Generated improver exceeded {timeout}s timeout") from error
                    raise RuntimeError("Generated improver exited without a result") from error
                if message[0] == "result":
                    return message[1]
                if message[0] == "error":
                    raise RuntimeError(message[1])
                if message[0] == "log":
                    _, name, level, message_text = message
                    logging.getLogger(name).log(level, message_text)
                    continue
                _, name, args, kwargs = message
                try:
                    value = services[name](*args, **kwargs)
                except Exception as error:
                    logger.warning("Generated improver service %s failed: %s", name, error)
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Generated improver exceeded {timeout}s timeout") from error
                    parent.send(("error", f"{type(error).__name__}: {error}"))
                else:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Generated improver exceeded {timeout}s timeout")
                    parent.send(("result", value))
        finally:
            watchdog.cancel()
            parent.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
