"""Run a bounded sandbox-backed survey: python research.py \"<topic>\"."""
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import redact
from langchain_core.callbacks import BaseCallbackHandler

ROOT = Path(__file__).parent
REPORTS = ROOT / 'reports'
VALIDATOR_SOURCE = ROOT / 'check_citations.py'
FINALIZER_SOURCE = ROOT / 'finalize_citations.py'


class Progress(BaseCallbackHandler):
    """Print tool names only, never arguments, fetched text or credentials."""
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[tool] {serialized.get('name', 'tool')}", flush=True)


def slugify(topic):
    return re.sub(r'[^\w]+', '-', topic.lower(), flags=re.UNICODE).strip('-')[:60].rstrip('-') or 'topic'


def build_prompt(topic):
    return (f'Create a deep research survey on: {topic.strip()}\n'
            f'Current date: {date.today().isoformat()}. Use the required sandbox workflow, >=3 parallel researcher tasks, '
            '>=3 source labels, foundational and recent papers, and an English thematic synthesis. '
            'Finish only after finalizer, validator OK and evidence spot-checks. Do not invent missing evidence.')


def summarize(messages, elapsed, model_name):
    calls = Counter()
    tokens = {'input': 0, 'output': 0}
    for message in messages:
        for call in getattr(message, 'tool_calls', None) or []:
            calls[call['name']] += 1
        usage = getattr(message, 'usage_metadata', None) or {}
        tokens['input'] += usage.get('input_tokens', 0)
        tokens['output'] += usage.get('output_tokens', 0)
    return {'model': model_name, 'elapsed_s': round(elapsed, 1), 'subagent_calls': calls['task'],
            'tool_calls': dict(calls), 'tokens': tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, sources_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip() or not sources_bytes:
        raise RuntimeError('sandbox report or sources is missing/empty; no outputs saved')
    try:
        report = report_bytes.decode('utf-8')
        sources = json.loads(sources_bytes)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError('invalid downloaded report or sources; no outputs saved') from exc
    problems = check(report, sources)
    if problems:
        raise RuntimeError('citation validation failed: ' + '; '.join(problems[:5]))
    families = sorted({s.get('source', '') for s in sources})
    if len(set(families) & {'arxiv', 'hf-daily', 'hf-search', 'web'}) < 3:
        raise RuntimeError('fewer than three source labels survived finalization')
    for source in sources:
        family, identifier, url = source.get('source'), source.get('id'), source.get('url')
        prefix = {'arxiv': 'https://arxiv.org/abs/', 'hf-daily': 'https://huggingface.co/papers/',
                  'hf-search': 'https://huggingface.co/papers/'}.get(family)
        if family not in {'arxiv', 'hf-daily', 'hf-search', 'web'} or (prefix and url != prefix + str(identifier)):
            raise RuntimeError(redact(f"source [{source.get('n')}] URL does not match family/identifier: "
                                      f"family={family!r}, id={identifier!r}, url={url!r}. "
                                      "Keep discovery provenance accurate; canonical arxiv/HF URL must end with its exact id."))
    metadata = dict(topic=topic, **summarize(messages, elapsed, model_name), n_sources=len(sources), source_families=families)
    if metadata['subagent_calls'] < 3:
        raise RuntimeError('lead did not delegate at least three tasks')
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    # Stage all three files before publication. Preserve the downloaded bytes exactly.
    with TemporaryDirectory(dir=reports_dir, prefix='.staging-') as staging:
        contents = {f'{slug}.md': report_bytes, f'{slug}.sources.json': sources_bytes,
                    f'{slug}.meta.json': (json.dumps(metadata, ensure_ascii=False, indent=2) + '\n').encode('utf-8')}
        for name, content in contents.items():
            (Path(staging) / name).write_bytes(content)
        previous = {name: (reports_dir / name).read_bytes() if (reports_dir / name).exists() else None
                    for name in contents}
        published = []
        try:
            for name in contents:
                (Path(staging) / name).replace(reports_dir / name)
                published.append(name)
        except OSError:
            for name in reversed(published):
                target = reports_dir / name
                if previous[name] is None:
                    # Only remove files this call just published, under the known output root.
                    if target.resolve().parent != reports_dir.resolve():
                        raise RuntimeError('unsafe output rollback path')
                    target.unlink()
                else:
                    target.write_bytes(previous[name])
            raise
    return reports_dir / f'{slug}.md'


def _execute_ok(backend, command):
    response = backend.execute(command)
    if response.exit_code != 0:
        raise RuntimeError(f'sandbox command failed: {redact(response.output)[:2000]}')
    return response.output


def main(topic):
    if not topic.strip():
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        start = time.monotonic()
        model_name = os.getenv('LAB_MODEL') or os.getenv('OPENAI_DEPLOYMENT_MODEL') or 'unknown'
        print(f'[research] {topic} | {model_name}', flush=True)
        with open_sandbox() as backend:
            _execute_ok(backend, f'mkdir -p {WORKDIR}/research/notes {WORKDIR}/report')
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            _execute_ok(backend, f'python3 -m py_compile {VALIDATOR_PATH} {FINALIZER_PATH}')
            agent = build_lead_agent(backend, model)
            result = agent.invoke({'messages': [{'role': 'user', 'content': build_prompt(topic)}]},
                                  config={'recursion_limit': 1000, 'callbacks': [Progress()]})
            # Bounded repair turns retain the same sandbox and evidence. No host-side rewriting.
            for repair in range(3):
                try:
                    print(_execute_ok(backend, f'python3 {FINALIZER_PATH}'), flush=True)
                    print(_execute_ok(backend, f'python3 {VALIDATOR_PATH}'), flush=True)
                    path = save_outputs(backend, topic, result['messages'], time.monotonic() - start, model_name)
                    break
                except RuntimeError as exc:
                    if repair == 2:
                        raise
                    diagnostic = redact(exc)
                    print(f'[repair {repair + 1}/2] {diagnostic}', flush=True)
                    result = agent.invoke({'messages': [*result['messages'], {'role': 'user', 'content':
                        'The final artifact gate rejected your output: ' + diagnostic +
                        '\nRepair the real files INSIDE the existing sandbox using retrieved evidence. '
                        'Do not invent sources or change scripts. Maintain >=3 source labels. '
                        'Run finalizer and validator again, and recheck any changed claims.'}]},
                        config={'recursion_limit': 1000, 'callbacks': [Progress()]})
        print(f'[saved] {path}', flush=True)
        return 0
    except Exception as exc:
        print(f'FAILED: {type(exc).__name__}: {redact(exc)}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main(' '.join(sys.argv[1:])))
