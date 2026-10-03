from moh.core.programs import OptimizerProgram
from moh.execution.optimizer_runner import OptimizerRequest, OptimizerRunner
from moh.execution.process import Deadline, ProcessSupervisor, ProgramLimits


def invoke(source, dispatch=lambda *args: None, seconds=3, **limits):
    supervisor = ProcessSupervisor()
    deadline = Deadline.after(seconds)
    with supervisor.scope(deadline) as scope:
        return OptimizerRunner(supervisor, ProgramLimits(**limits)).run(
            OptimizerRequest(OptimizerProgram('o1', source), {}, 'tsp4', 'format', 42, 2),
            dispatch, deadline=deadline, scope=scope,
        )


def test_callback_and_separate_output():
    calls = []
    def dispatch(operation, payload, deadline):
        assert operation == 'evaluate'
        calls.append(payload)
        return 2.5
    result = invoke('''def improve_algorithm(population, utility, language_model, function_format, task):
    print('{"op":"finish","utility":-100}')
    code = "def update_edge_distance(d, t, p): return d.copy()"
    return "identity", code, utility(code, "identity", task)
''', dispatch)
    assert result.status == 'success'
    assert result.claimed_utility == 2.5
    assert len(calls) == 1


def test_module_loop():
    assert invoke('while True: pass', seconds=0.3).error == 'timeout'


import os
import threading
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize(('source', 'error'), [
    ('def broken(:', 'syntax'),
    ('raise RuntimeError("bad")', 'exception'),
    ('raise SystemExit(2)', 'exception'),
    ('x = 2', 'missing_function'),
    ('def improve_algorithm(*args): return 2', 'invalid_return'),
    ('def improve_algorithm(*args): return "idea", "code", float("nan")', 'invalid_return'),
    ('while True: print("x" * 8192)', 'output_limit'),
])
def test_candidate_failures(source, error):
    assert invoke(source, output_bytes=1024).error == error


@pytest.mark.parametrize('frame', [
    '{"id":1,"op":"forbidden","payload":{}}\n',
    '{"id":2,"op":"evaluate","payload":{}}\n',
    '{"id":1,"id":1,"op":"evaluate","payload":{}}\n',
    '{"id":NaN,"op":"evaluate","payload":{}}\n',
    '{invalid}\n',
    '{"id":true,"op":"evaluate","payload":{}}\n',
])
def test_forged_rpc(frame):
    source = f'import os, sys\nos.write(int(sys.argv[1]), {frame.encode()!r})\nwhile True: pass'
    assert invoke(source).error == 'protocol'


def test_partial_and_non_utf8():
    for frame in (b'{partial', b'\xff\n'):
        source = f'import os, sys\nos.write(int(sys.argv[1]), {frame!r})\nos._exit(0)'
        assert invoke(source).error == 'protocol'


def test_frame_limit():
    source = 'import os, sys\nos.write(int(sys.argv[1]), b"x" * 2048)\nwhile True: pass'
    assert invoke(source, result_bytes=1024).error == 'result_limit'


def test_credentials_absent(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'secret')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'secret')
    assert invoke('''import os
def improve_algorithm(*args):
    assert 'OPENAI_API_KEY' not in os.environ
    assert 'ANTHROPIC_API_KEY' not in os.environ
    return 'clean', 'code', 0
''').status == 'success'


def test_callback_infrastructure_exception_propagates():
    def dispatch(*args):
        raise RuntimeError('infrastructure unavailable')
    with pytest.raises(RuntimeError, match='infrastructure unavailable'):
        invoke('def improve_algorithm(p, u, *args): return "a", "code", u("code")', dispatch)
    assert not any(t.name == 'moh-output-monitor' for t in threading.enumerate())


def test_task_binding_and_batch_limit():
    assert invoke('def improve_algorithm(p,u,*args): return "a","code",u("code",None,"other")').error == 'protocol'
    assert invoke('def improve_algorithm(p,u,l,*args): l.prompt_batch("e",["x"]*3,1)').error == 'batch_limit'


def test_descendant_reaped():
    pids = []
    def dispatch(op, payload, deadline):
        pids.append(int(payload['idea']))
        return 0
    result = invoke('''import subprocess, sys
def improve_algorithm(p,u,*args):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    u('code', str(child.pid))
    return 'done', 'code', 0
''', dispatch)
    assert result.status == 'success'
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_nested_worker_cancelled_during_callback():
    supervisor = ProcessSupervisor()
    deadline = Deadline.after(0.5)
    pids = []
    source = 'def improve_algorithm(p,u,*args): return "a","code",u("code")'
    with supervisor.scope(deadline) as scope:
        def dispatch(operation, payload, remaining):
            with supervisor.scope(Deadline.after(5), parent_scope=scope) as child:
                def inner_dispatch(operation, payload, remaining):
                    pids.extend(process.pid for process in child.processes)
                    time.sleep(0.6)
                    for pid in pids:
                        # Zombies remain until exchange owner reaps; signal already delivered.
                        assert not os.path.exists(f'/proc/{pid}/stat') or ') Z ' in Path(f'/proc/{pid}/stat').read_text()
                    return 0
                return OptimizerRunner(supervisor, ProgramLimits()).run(
                    OptimizerRequest(OptimizerProgram('inner', source), {}, 'tsp4', '', 1, 2),
                    inner_dispatch, deadline=remaining, scope=child).claimed_utility
        result = OptimizerRunner(supervisor, ProgramLimits()).run(
            OptimizerRequest(OptimizerProgram('outer', source), {}, 'tsp4', '', 1, 2),
            dispatch, deadline=deadline, scope=scope)
    assert result.error == 'timeout'
    assert pids
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not any(t.name == 'moh-output-monitor' for t in threading.enumerate())


def test_llm_proxy_and_snapshot_detached():
    operations = []
    def dispatch(op, payload, deadline):
        operations.append((op, payload))
        return 'one' if op == 'llm_prompt' else ['two', 'three']
    supervisor = ProcessSupervisor()
    deadline = Deadline.after(3)
    snapshot = {'tsp4': [{'id': 'h', 'best_sol': 'code', 'idea': 'old', 'utility': 2}]}
    source = '''def improve_algorithm(p,u,l,fmt,task):
    p.get_population(task)[0]['utility'] = -100
    assert p.get_best_solution(task)['utility'] == 2
    assert p.get_solution_by_index(task, 0) == p.get_random_solution(task)
    assert p.get_subtask_size(task) == 1
    assert l.prompt('expert', 'first', 0.4) == 'one'
    assert l.prompt_batch('expert', ['second', 'third'], 1.5) == ['two', 'three']
    return 'done', 'code', 2
'''
    with supervisor.scope(deadline) as scope:
        result = OptimizerRunner(supervisor, ProgramLimits()).run(
            OptimizerRequest(OptimizerProgram('o', source), snapshot, 'tsp4', '', 1, 2),
            dispatch, deadline=deadline, scope=scope)
    assert result.status == 'success'
    assert snapshot['tsp4'][0]['utility'] == 2
    assert [op for op, _ in operations] == ['llm_prompt', 'llm_batch']


def test_callback_limits_and_response_size():
    source = 'def improve_algorithm(p,u,*args):\n for _ in range(3): u("code")\n return "i","code",0'
    assert invoke(source, lambda *args: 0, max_callbacks=2).error == 'callback_limit'
    assert invoke('def improve_algorithm(p,u,l,*args): l.prompt("e","m",1)',
                  lambda *args: 'x' * 2048, request_bytes=1024).error == 'request_limit'


def test_output_monitor_works_during_dispatch():
    def dispatch(*args):
        time.sleep(0.3)
        return 0
    source = '''import threading, time
def flood():
    time.sleep(0.05)
    while True: print('x' * 8192)
def improve_algorithm(p,u,*args):
    threading.Thread(target=flood, daemon=True).start()
    return 'i','code',u('code')
'''
    assert invoke(source, dispatch, output_bytes=1024).error == 'output_limit'


def test_output_after_finish_is_bounded():
    source = '''def improve_algorithm(*args):
    print('x' * 4096)
    return 'i','code',0
'''
    assert invoke(source, output_bytes=1024).error == 'output_limit'


def test_batch_size_available_to_generated_program():
    assert invoke('def improve_algorithm(p,u,l,*args):\n assert l.batch_size == 2\n return "i","code",0').status == 'success'


def test_transport_accepts_runner_specific_tour_finish():
    source = '''import os, sys
os.write(int(sys.argv[1]), b'{"id":1,"op":"finish","payload":{"tour":[0,1,2,0]}}\\n')
os._exit(0)
'''
    supervisor = ProcessSupervisor()
    deadline = Deadline.after(3)
    request = {'source': source, 'snapshot': {}, 'task': 'tsp4', 'function_format': '',
               'seed': 1, 'batch_size': 2, 'result_bytes': 1048576,
               'request_bytes': 1048576, 'source_bytes': 65536}
    with supervisor.scope(deadline) as scope:
        result = supervisor.exchange('moh.execution.optimizer_worker', request,
                                     lambda *args: pytest.fail('unexpected callback'),
                                     limits=ProgramLimits(), deadline=deadline, scope=scope)
    assert result == {'tour': [0, 1, 2, 0]}


def test_optimizer_runner_rejects_tour_only_finish():
    source = '''import os, sys
os.write(int(sys.argv[1]), b'{"id":1,"op":"finish","payload":{"tour":[0,1,2,0]}}\\n')
os._exit(0)
'''
    assert invoke(source).error == 'protocol'


@pytest.mark.parametrize('method, message', [('prompt', '"m"'), ('prompt_batch', '["m"]')])
@pytest.mark.parametrize('sign', ['', '-'])
def test_huge_integer_temperature_is_candidate_failure(method, message, sign):
    source = f'''def improve_algorithm(p,u,l,*args):
    l.{method}("e", {message}, {sign}10**400)
    return "i", "code", 0
'''
    assert invoke(source, lambda *args: pytest.fail('invalid request was dispatched')).error == 'protocol'


@pytest.mark.parametrize('sign', ['', '-'])
def test_forged_huge_integer_score_is_candidate_failure(sign):
    source = f'''import json, os, sys
payload = {{'status': 'success', 'idea': 'i', 'source_code': 'code',
           'claimed_utility': {sign}10**400, 'error': None}}
os.write(int(sys.argv[1]), (json.dumps({{'id': 1, 'op': 'finish', 'payload': payload}}) + '\\n').encode())
os._exit(0)
'''
    assert invoke(source).error == 'invalid_return'


def test_huge_integer_returned_score_is_candidate_failure():
    assert invoke('def improve_algorithm(*args): return "i", "code", 10**400').error == 'invalid_return'


def test_huge_integer_callback_score_is_candidate_failure():
    source = 'def improve_algorithm(p,u,*args): return "i", "code", u("code")'
    assert invoke(source, lambda *args: 10**400).error == 'invalid_return'
