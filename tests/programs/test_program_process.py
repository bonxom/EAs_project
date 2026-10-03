import pytest

from moh.execution.process import CandidateFailure, Deadline, ProgramLimits


def test_limits_and_deadline():
    with pytest.raises(ValueError):
        ProgramLimits(timeout_seconds=float('nan'))
    with pytest.raises(ValueError):
        ProgramLimits(batch_size=True)
    deadline = Deadline.after(0.001)
    assert Deadline.after(10, parent=deadline).expires_at == deadline.expires_at
    with pytest.raises(CandidateFailure):
        Deadline(0).check()


def test_cancelling_already_closed_child_is_idempotent():
    from moh.execution.process import ProcessSupervisor
    supervisor = ProcessSupervisor()
    deadline = Deadline.after(1)
    with supervisor.scope(deadline) as parent:
        with supervisor.scope(deadline, parent_scope=parent) as child:
            child.close()
            parent.abort()
        child.close()
    parent.close()
