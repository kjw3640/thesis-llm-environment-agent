"""Pinned Ollama 0.33.3 adapter. Only named observation tools may execute.

The render-only and runner tokenization hooks are version-specific. They fail
closed when unavailable. No arbitrary function execution or response retries.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROFILE = json.loads((ROOT / 'outputs/case_study_v5/execution_settings_v2.json').read_text())

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'query_observations',
        'description': '현재 자료의 관측을 저장 단위 그대로 조회한다. records는 원행·결측·플래그를 보존한다. sum/mean은 메타데이터의 중복·품질 정책을 적용한다. 다중 출처는 source_id를 지정한다. strict에서 불완전하면 값 대신 오류를 반환하며 available_only는 관측분만 명시적으로 집계한다. 필터도 저장 단위를 쓴다. 날짜 양끝 포함. 빈 조건 결과와 자료 미제공을 구분한다.',
        'parameters': {'type': 'object', 'required': ['variable', 'start', 'end', 'operation'], 'additionalProperties': False,
            'properties': {
                'variable': {'type': 'string', 'description': '메타데이터 variables의 키'},
                'start': {'type': 'string', 'description': 'YYYY-MM-DD'},
                'end': {'type': 'string', 'description': 'YYYY-MM-DD'},
                'operation': {'type': 'string', 'enum': ['records', 'sum', 'mean']},
                'source_id': {'type': 'string'}, 'station': {'type': 'string'},
                'filters': {'type': 'array', 'items': {'type': 'object', 'required': ['variable', 'operator', 'value'],
                    'additionalProperties': False, 'properties': {
                        'variable': {'type': 'string'}, 'operator': {'type': 'string', 'enum': ['>', '>=', '<', '<=', '=', '==', '!=']},
                        'value': {'type': 'number'}}}},
                'sort_by': {'type': 'string', 'description': '변수 키 또는 date'},
                'descending': {'type': 'boolean'}, 'limit': {'type': 'integer', 'minimum': 1},
                'missing_policy': {'type': 'string', 'enum': ['strict', 'available_only']}}}}},
    {'type': 'function', 'function': {'name': 'subtract', 'description': 'left에서 right를 뺀다. 조회나 단위 변환은 하지 않는다.',
        'parameters': {'type': 'object', 'required': ['left', 'right'], 'additionalProperties': False,
            'properties': {'left': {'type': 'number'}, 'right': {'type': 'number'}}}}},
    {'type': 'function', 'function': {'name': 'select_dataset', 'description': '현재 제공된 dataset_id의 자료를 선택한다. 다른 자료에 접근하거나 서비스 장애를 해제하는 기능은 없다.',
        'parameters': {'type': 'object', 'required': ['dataset_id'], 'additionalProperties': False,
            'properties': {'dataset_id': {'type': 'string'}}}}}
]

CASE_DEADLINE = None

def http_json(url, data=None, timeout=180):
    if CASE_DEADLINE is not None:
        remaining = CASE_DEADLINE - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('case_timeout: total case budget exhausted')
        timeout = min(timeout, remaining)
    request = urllib.request.Request(url, data=None if data is None else json.dumps(data, ensure_ascii=False).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc
    if isinstance(value, dict) and value.get('error'):
        raise RuntimeError(value['error'])
    return value

def api(path, data=None, timeout=180):
    return http_json('http://127.0.0.1:11434/api/' + path, data, timeout)

class OllamaAdapter:
    def __init__(self, model, trace_dir):
        self.model = model
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.requests = []
        self.render_requests = 0
        self.tokenize_requests = 0
        assert api('version')['version'] == PROFILE['required_server_version']
        inventory = {m['name']: m for m in api('tags')['models']}
        self.installed = inventory[model['tag']]
        if model.get('expected_manifest_sha256'):
            assert self.installed['digest'] == model['expected_manifest_sha256']
        self.show = api('show', {'model': model['tag']})
        assert self.show['details']['format'] == 'gguf'
        assert self.show['details']['quantization_level'] == 'Q4_K_M'
        assert 'tools' in self.show['capabilities']
        if model['thinking_policy'].startswith('require_false'):
            assert 'thinking' in self.show['capabilities']
        else:
            assert 'instruct-2512' in model['tag']
            assert 'thinking' not in self.show['capabilities']
        self._save('metadata', {'installed': self.installed, 'show': self.show})

    def _save(self, name, value):
        target = self.trace_dir / (name + '.json')
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(target)

    def request(self, messages):
        request = {'model': self.model['tag'], 'messages': deepcopy(messages), 'tools': deepcopy(TOOLS),
                   'stream': False, 'keep_alive': '5m', 'options': deepcopy(PROFILE['common_options']),
                   'truncate': False, 'shift': False}
        if self.model['thinking_policy'].startswith('require_false'):
            request['think'] = False
        return request

    def render(self, request):
        response = api('chat', {**request, '_debug_render_only': True})
        self.render_requests += 1
        prompt = response['_debug_info']['rendered_template']
        assert prompt and response.get('eval_count', 0) == 0
        assert not response.get('message', {}).get('content')
        return prompt

    def _runner_url(self):
        # Read only runner launch lines from the local server log. Verify the
        # loaded weight path via /props; never send data to an unverified port.
        path = (Path.home() / '.ollama/logs/server.log')
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 3000000))
            lines = handle.read().decode(errors='replace').splitlines()
        matches = []
        for line in lines:
            if 'starting llama-server' not in line or '--port' not in line:
                continue
            match = re.search(r'--port[ =]+(\d+)', line)
            if match:
                matches.append(int(match.group(1)))
        manifest_path = (Path.home() / '.ollama/models/manifests/registry.ollama.ai/library') / self.model['tag'].replace(':', '/')
        manifest = json.loads(manifest_path.read_text())
        weights = {layer['digest'].replace(':', '-') for layer in manifest['layers'] if layer['mediaType'].endswith('.model')}
        for port in dict.fromkeys(reversed(matches)):
            url = 'http://127.0.0.1:' + str(port)
            try:
                props = http_json(url + '/props', timeout=2)
            except Exception:
                continue
            model_path = props.get('model_path', '')
            if Path(model_path).name in weights:
                return url
        raise RuntimeError('Verified runner tokenization endpoint unavailable; cannot verify token budget')

    def token_count(self, prompt):
        # Some templates include BOS; others rely on the tokenizer to add it.
        # Count the actual installed tokenizer's required BOS exactly once.
        url = self._runner_url() + '/tokenize'
        response = http_json(url,
                             {'content': prompt, 'add_special': False, 'parse_special': True}, timeout=30)
        self.tokenize_requests += 1
        tokens = response['tokens']
        info = self.show['model_info']
        bos = info.get('tokenizer.ggml.bos_token_id')
        needs_bos = info.get('tokenizer.ggml.add_bos_token', False) or self.show['details']['family'] == 'gemma4'
        if needs_bos and (not tokens or tokens[0] != bos):
            special = http_json(url, {'content': prompt, 'add_special': True, 'parse_special': True}, timeout=30)
            self.tokenize_requests += 1
            if bos is None or special['tokens'] != [bos, *tokens]:
                raise RuntimeError('Unexpected native special-token insertion; stop rather than estimate')
            tokens = special['tokens']
        return len(tokens)

    def complete(self, messages):
        request = self.request(messages)
        ordinal = len(self.requests) + 1
        record = {'ordinal': ordinal, 'request': request}
        self.requests.append(record)
        self._save(f'request_{ordinal:02d}', record)
        try:
            prompt = self.render(request)
            count = self.token_count(prompt)
            record['input_tokens_before_generation'] = count
            record['rendered_prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
            self._save(f'request_{ordinal:02d}', record)
            if count > PROFILE['context_guard']['input_token_limit_including_template_and_tools']:
                raise RuntimeError('context_limit: rendered input exceeds 12288 tokens')
            started = time.monotonic()
            record['generation_request_sent_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            self._save(f'request_{ordinal:02d}', record)
            response = api('chat', request, PROFILE['request_timeout_seconds'])
            record['wall_seconds'] = round(time.monotonic() - started, 3)
            record['response'] = response
            record['token_count_matches_actual'] = response.get('prompt_eval_count') == count
            self._save(f'request_{ordinal:02d}', record)
            if not record['token_count_matches_actual']:
                raise RuntimeError(f'Tokenization mismatch: counted {count}, actual {response.get("prompt_eval_count")}')
            if response.get('message', {}).get('thinking'):
                raise RuntimeError('Unexpected thinking output under non-thinking policy')
            if response.get('done_reason') == 'length':
                raise RuntimeError('output_limit: generation reached maximum length')
            return response['message']
        except Exception as exc:
            record['error'] = repr(exc)
            self._save(f'request_{ordinal:02d}', record)
            raise

    def unload(self):
        return api('generate', {'model': self.model['tag'], 'keep_alive': 0, 'stream': False})
