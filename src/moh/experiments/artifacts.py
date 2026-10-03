"""Schema 2 artifacts retaining parent-approved program search state."""

from moh.core.programs import OptimizerProgram, _program, _safe_id
from moh.logging import RunRecorder, encode_record


class ProgramRecorder(RunRecorder):
    @classmethod
    def create(cls, root, resolved_config, *, provenance=None, redactions=()):
        recorder = super().create(root, resolved_config, redactions=redactions)
        try:
            (recorder.path / "populations").mkdir()
            recorder.metadata.update(
                {"schema_version": 2, "provenance": provenance or {}}
            )
            recorder._write_run({"status": "running"})
        except BaseException:
            recorder.close()
            raise
        return recorder

    def _artifact(self, folder, identifier, suffix, content):
        _safe_id(identifier)
        path = self.path / folder / (identifier + suffix)
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise ValueError("artifact identity collision")
        else:
            path.write_text(content, encoding="utf-8")

    def save_heuristic(self, heuristic):
        _program(heuristic)
        super().save_heuristic(heuristic)

    def save_optimizer(self, optimizer: OptimizerProgram):
        if not isinstance(optimizer, OptimizerProgram):
            raise TypeError("expected OptimizerProgram")
        self._artifact(
            "optimizers", optimizer.id, ".py", self._redact(optimizer.source_code)
        )

    def checkpoint(self, label, populations, active, winner):
        payload = {"populations": populations, "active": active, "winner": winner}
        self._artifact(
            "populations", label, ".json", encode_record(self._safe_value(payload))
        )

    def emit(self, event, payload):
        if {"schema_version", "sequence", "event"} & payload.keys():
            raise ValueError("reserved event envelope key")
        record = {
            "schema_version": 2,
            "sequence": self.sequence,
            "event": event,
            **payload,
        }
        self.events.write(encode_record(self._safe_value(record)) + "\n")
        self.events.flush()
        self.sequence += 1
