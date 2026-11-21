"""Sequential 4 x 53 main run, with atomic records and no response resampling."""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import time
import traceback

import main_adapter_v1 as transport
from runtime import make_session

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'outputs/case_study_v5'
OUT = ASSETS / 'main_v1'
PROFILE = json.loads((ASSETS / 'execution_settings_v2.json').read_text())
TERMINAL = {'completed', 'request_timeout', 'case_timeout', 'llm_request_limit', 'tool_call_limit',
            'context_limit', 'output_limit', 'protocol_error', 'infrastructure_error', 'interrupted_uncertain'}

def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def classify_exception(exc, start, clock=time.monotonic):
    text = str(exc)
    if clock() - start >= PROFILE['case_timeout_seconds']:
        return 'case_timeout'
    for code in ['context_limit', 'output_limit']:
        if text.startswith(code + ':'):
            return code
    if isinstance(exc, (TimeoutError, socket.timeout)) or 'timed out' in text.lower():
        return 'request_timeout'
    if isinstance(exc, (KeyError, TypeError, ValueError)):
        return 'protocol_error'
    return 'infrastructure_error'

def execute_case(conversation, model, case_id, folder, adapter_factory=transport.OllamaAdapter, clock=time.monotonic):
    folder = Path(folder)
    result = {'model_id': model['id'], 'model_tag': model['tag'], 'case_id': case_id,
              'started_at': timestamp(), 'status': 'running', 'user_turns_presented': 1,
              'final_answers': [], 'model_requests_attempted': 0, 'scoring_status': 'pending_review'}
    start = clock()
    adapter = None
    def save():
        result['messages'] = deepcopy(conversation.messages)
        result['tool_events'] = deepcopy(conversation.environment.events)
        result['recovery_observations'] = conversation.environment.recovery_observations()
        result['elapsed_seconds'] = round(clock() - start, 3)
        if adapter is not None:
            result['model_requests_attempted'] = len(adapter.requests)
            result['generation_requests_sent'] = sum('generation_request_sent_at' in r or 'response' in r for r in adapter.requests)
            result['generation_responses_received'] = sum('response' in r for r in adapter.requests)
        write_json(folder / 'result.json', result)
    save()
    try:
        transport.CASE_DEADLINE = start + PROFILE['case_timeout_seconds']
        adapter = adapter_factory(model, folder / 'traces')
        while True:
            if clock() - start >= PROFILE['case_timeout_seconds']:
                result['status'] = 'case_timeout'
                break
            if len(adapter.requests) >= PROFILE['max_llm_requests_per_case']:
                result['status'] = 'llm_request_limit'
                break
            message = adapter.complete(conversation.messages)
            result['last_received_assistant_message'] = deepcopy(message)
            if clock() - start >= PROFILE['case_timeout_seconds']:
                result['status'] = 'case_timeout'
                break
            calls = message.get('tool_calls') or []
            if message.get('role') != 'assistant' or not isinstance(calls, list):
                raise ValueError('Invalid native assistant message')
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get('function'), dict) or not isinstance(call['function'].get('name'), str):
                    raise ValueError('Malformed native tool call')
            # Preserve the assistant proposal, execute at most the remaining
            # number of tool calls, and retain explicit budget-error returns.
            calls_before = len(conversation.environment.events)
            conversation.assistant(message)
            save()
            if calls_before + len(calls) > PROFILE['max_tool_calls_per_case']:
                result['status'] = 'tool_call_limit'
                break
            if calls:
                continue
            result['final_answers'].append({'user_turn': result['user_turns_presented'],
                                             'content': message.get('content', ''),
                                             'assistant_message_index': len(conversation.messages) - 1})
            if conversation._followup and not conversation._followup_sent:
                conversation.next_user_turn()
                result['user_turns_presented'] += 1
                save()
                continue
            result['status'] = 'completed'
            break
    except Exception as exc:
        result['status'] = classify_exception(exc, start, clock)
        result['error'] = repr(exc)
        result['error_traceback'] = traceback.format_exc()
    finally:
        transport.CASE_DEADLINE = None
        result['finished_at'] = timestamp()
        save()
    return result

def initialize(root=OUT):
    root.mkdir(parents=True, exist_ok=True)
    # All semantic inputs and executable code are hashed before the first run.
    paths = [ASSETS / name for name in ('execution_settings_v2.json', 'tasks.json', 'packets.json', 'design.json',
                                       'scoring_ledger_template.json', 'scenario_catalog.json', 'literature_basis.json')]
    paths += [ROOT / 'work/case_study_v5' / name for name in ('runtime.py', 'main_adapter_v1.py', 'run_main_v1.py')]
    freeze = {'version': 'four-model-main-1.0', 'created_at': timestamp(), 'profile': PROFILE,
              'planned_model_case_runs': 212, 'planned_user_turns': 228, 'repetitions': 1,
              'tool_schemas': transport.TOOLS,
              'hashes': {str(p.relative_to(ROOT)): sha(p) for p in paths},
              'resumption': 'Never regenerate a terminal case. A previously running case is marked interrupted_uncertain, retaining its partial evidence.',
              'grading': 'generation and grading are separate; every answer unit starts pending_review, with no guessed success scores'}
    target = root / 'freeze.json'
    if target.exists():
        old = json.loads(target.read_text())
        assert old['hashes'] == freeze['hashes'], 'Frozen main execution inputs changed'
        assert old['tool_schemas'] == freeze['tool_schemas']
    else:
        write_json(target, freeze)
    for entry in ['design_manifest.json', 'preflight_manifest.json']:
        manifest = json.loads((ASSETS / entry).read_text())
        for relative, expected in manifest['hashes'].items():
            path = ASSETS / relative if entry == 'design_manifest.json' else ROOT / relative
            assert sha(path) == expected, 'Prior evidence changed: ' + str(path)
    return freeze

def recover_or_skip(folder):
    path = folder / 'result.json'
    if not path.exists():
        return False
    old = json.loads(path.read_text())
    if old['status'] == 'running':
        old.update({'status': 'interrupted_uncertain', 'finished_at': timestamp(),
                    'interruption_note': 'Previous process ended during this case. Partial traces retained; no automatic resampling.'})
        write_json(path, old)
    assert old['status'] in TERMINAL
    return True

def update_status(root=OUT, current=None, phase='running'):
    results = []
    for model in PROFILE['models']:
        for case in PROFILE['case_order']:
            path = root / model['id'] / case / 'result.json'
            if path.exists():
                results.append(json.loads(path.read_text()))
    statuses = Counter(r['status'] for r in results)
    ledger = []
    template = {r['case_id']: r for r in json.loads((ASSETS / 'scoring_ledger_template.json').read_text())}
    lookup = {(r['model_id'], r['case_id']): r for r in results}
    for model in PROFILE['models']:
        for case in PROFILE['case_order']:
            entry = deepcopy(template[case])
            entry.update({'model_id': model['id'], 'model_tag': model['tag'], 'execution_status': 'not_run'})
            result = lookup.get((model['id'], case))
            if result:
                entry['execution_status'] = result['status']
                entry['whole_case'] = 'pending_review'
                for unit in entry['units']:
                    unit['result'] = 'pending_review'
                recovery = result.get('recovery_observations', {})
                entry.update({'tool_error_exposed': recovery.get('error_exposed'),
                              'tool_recovered': recovery.get('query_recovered'),
                              'prerequisite_fulfilled': recovery.get('prerequisite_fulfilled'),
                              'actual_tool_events': result.get('tool_events', []),
                              'evidence_file': f"{model['id']}/{case}/result.json"})
            ledger.append(entry)
    write_json(root / 'scoring_ledger.json', ledger)
    status = {'updated_at': timestamp(), 'phase': phase, 'planned_cases': 212,
              'terminal_cases': sum(r['status'] in TERMINAL for r in results),
              'status_counts': dict(statuses), 'current': current,
              'per_model': {m['id']: dict(Counter(r['status'] for r in results if r['model_id'] == m['id'])) for m in PROFILE['models']},
              'generation_requests_sent': sum(r.get('generation_requests_sent', 0) for r in results),
              'generation_responses_received': sum(r.get('generation_responses_received', 0) for r in results),
              'grading': 'pending_review; execution completion is not correctness'}
    write_json(root / 'status.json', status)
    return status

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize()
        version = transport.api('version')['version']
        assert version == PROFILE['required_server_version']
        inventory = {m['name']: m for m in transport.api('tags')['models']}
        assert all(inventory[m['tag']]['digest'] == m['expected_manifest_sha256'] for m in PROFILE['models'])
        update_status()
        infrastructure_failures = 0
        for model in PROFILE['models']:
            for ordinal, case_id in enumerate(PROFILE['case_order'], 1):
                folder = OUT / model['id'] / case_id
                if recover_or_skip(folder):
                    continue
                update_status(current={'model': model['id'], 'case_id': case_id, 'model_case_ordinal': ordinal})
                result = execute_case(make_session(case_id), model, case_id, folder)
                state = update_status()
                print(json.dumps({'model': model['id'], 'case_id': case_id, 'status': result['status'],
                                  'seconds': result['elapsed_seconds'], 'finished': state['terminal_cases'],
                                  'total': 212}, ensure_ascii=False), flush=True)
                infrastructure_failures = infrastructure_failures + 1 if result['status'] == 'infrastructure_error' else 0
                if infrastructure_failures >= 3 or 'Tokenization mismatch' in result.get('error', ''):
                    update_status(phase='stopped_for_infrastructure_check')
                    raise RuntimeError('Infrastructure problem; preserve evidence before further execution')
            try:
                transport.api('generate', {'model': model['tag'], 'keep_alive': 0, 'stream': False}, timeout=30)
            except Exception as exc:
                write_json(OUT / (model['id'] + '_unload_notice.json'), {'error': repr(exc), 'time': timestamp()})
        final = update_status(phase='generation_complete')
        assert final['terminal_cases'] == 212
        print(json.dumps(final, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        run()
    else:
        print('Run check_main_v1.py for the no-inference operational check; --execute starts the frozen main runs.')
