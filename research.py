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

from agents import FINALIZER_PATH, PREPARER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import redact
from langchain_core.callbacks import BaseCallbackHandler

ROOT = Path(__file__).parent
REPORTS = ROOT / 'reports'
VALIDATOR_SOURCE = ROOT / 'check_citations.py'
FINALIZER_SOURCE = ROOT / 'finalize_citations.py'
PREPARER_SOURCE = ROOT / 'prepare_report.py'


class Progress(BaseCallbackHandler):
    """Print tool names only, never arguments, fetched text or credentials."""
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[tool] {serialized.get('name', 'tool')}", flush=True)


def slugify(topic):
    return re.sub(r'[^\w]+', '-', topic.lower(), flags=re.UNICODE).strip('-')[:60].rstrip('-') or 'topic'


SEARCH_HINTS = {
    'survey about world model': 'World Models Ha Schmidhuber; DreamerV3; MuZero; Genie; action-conditioned video world models',
    'survey about reinforcement learning for llm reasoning': 'STaR; DeepSeek-R1; GRPO; RLVR; process reward models; reasoning reinforcement learning',
    'survey about llm agents and tool use': 'ReAct; Toolformer; WebArena; SWE-bench; agent planning tool use',
    'survey about video and multimodal generation': 'DDPM; latent diffusion; DiT; Sora; Movie Gen; video diffusion; multimodal generation',
    'survey about efficient inference and small language models': 'FlashAttention; vLLM PagedAttention; AWQ; GPTQ; speculative decoding; Phi-3; Qwen small models',
}


def build_prompt(topic):
    hints = SEARCH_HINTS.get(topic.strip().lower(), topic.strip())
    return (f'Create a deep research survey on: {topic.strip()}\n'
            f'Search seeds (queries only, NOT evidence or claims): {hints}. Verify all facts with tools. '
            'Cover the core technical methods, not tangential applications. Use short targeted queries. '
            'Find and cite at least one genuinely foundational source older than the last two years. '
            'Read its text, contrast it with recent work, and cite both. '
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


def check_report_quality(report, sources):
    """Structural guard, not a substitute for verifying semantic citation support."""
    body = report.split('## References', 1)[0]
    headings = re.findall(r'^## (.+)$', body, re.M)
    required = {'TL;DR', 'Background', 'Trends and open problems'}
    problems = []
    if not required.issubset(headings):
        problems.append('missing required headings: ' + ', '.join(sorted(required - set(headings))))
    if not 3 <= len([h for h in headings if h not in required]) <= 6:
        problems.append('survey must have 3-6 substantive thematic sections')
    words = len(re.findall(r'\b[\w-]+\b', body))
    if words < 1200:
        problems.append(f'only {words} body words; need >=1200 evidence-supported words, no padding')
    years = [int(str(s.get('date', ''))[:4]) for s in sources if re.match(r'^\d{4}', str(s.get('date', '')))]
    if not any(y < date.today().year - 2 for y in years):
        problems.append('missing cited foundational source older than the last two years')
    if not any(y >= date.today().year - 2 for y in years):
        problems.append('missing cited recent source from the last two years')
    return problems


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
    if not problems:
        problems += check_report_quality(report, sources)
    if problems:
        raise RuntimeError('citation validation failed: ' + '; '.join(problems[:5]))
    families = sorted({s.get('source', '') for s in sources})
    if len(set(families) & {'arxiv', 'hf-daily', 'hf-search', 'web'}) < 3:
        raise RuntimeError('fewer than three source labels survived finalization')
    source_problems = []
    for source in sources:
        family, identifier, url = source.get('source'), source.get('id'), source.get('url')
        prefix = {'arxiv': 'https://arxiv.org/abs/', 'hf-daily': 'https://huggingface.co/papers/',
                  'hf-search': 'https://huggingface.co/papers/'}.get(family)
        if family not in {'arxiv', 'hf-daily', 'hf-search', 'web'} or (prefix and url != prefix + str(identifier)):
            source_problems.append(f"source [{source.get('n')}] URL does not match family/identifier: "
                                   f"family={family!r}, id={identifier!r}, url={url!r}")
    if source_problems:
        raise RuntimeError(redact('; '.join(source_problems) +
                           '. Keep discovery provenance accurate; canonical arxiv/HF URL must end with its exact id.'))
    metadata = dict(topic=topic, **summarize(messages, elapsed, model_name), n_sources=len(sources), source_families=families)
    delegated = Counter(call.get('args', {}).get('subagent_type') for message in messages
                        for call in (getattr(message, 'tool_calls', None) or []) if call.get('name') == 'task')
    if delegated['researcher'] < 3:
        raise RuntimeError('lead must delegate at least three researcher tasks')
    if not delegated['citation-checker']:
        raise RuntimeError('lead must invoke citation-checker with concrete claim/URL pairs before completion')
    metadata['research_model'] = os.getenv('LAB_RESEARCH_MODEL') or model_name
    metadata['researcher_calls'] = delegated['researcher']
    metadata['citation_checker_calls'] = delegated['citation-checker']
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


PAPER_URL = re.compile(r'^https://(?:arxiv\.org/abs|huggingface\.co/papers)/(\d{4}\.\d{4,5})$')


def _norm_title(text):
    return set(re.findall(r'[a-z0-9]+', str(text).lower()))


_STOP = {'a', 'an', 'the', 'of', 'for', 'and', 'in', 'on', 'to', 'with', 'via', 'by', 'from', 'is', 'are',
         'arxiv', 'preprint', 'paper', 'official', 'foundational'}


def fetch_paper_title(paper_id):
    """Real title of an arXiv id via Hugging Face (no rate limit), falling back to arXiv. None if unknown."""
    import httpx
    try:
        r = httpx.get(f'https://huggingface.co/api/papers/{paper_id}', timeout=20, follow_redirects=True)
        if r.status_code == 200 and r.json().get('title'):
            return r.json()['title']
        time.sleep(3)
        r = httpx.get('https://export.arxiv.org/api/query', params={'id_list': paper_id}, timeout=30)
        m = re.search(r'<entry>.*?<title>(.*?)</title>', r.text, re.S)
        return re.sub(r'\s+', ' ', m.group(1)).strip() if m else None
    except Exception:
        return None


def audit_source_titles(sources, fetch=fetch_paper_title):
    """Deterministic check that each arXiv/HF paper id really is the paper the manifest says it is."""
    bad = []
    for s in sources:
        if s.get('source') in ('arxiv', 'hf-daily', 'hf-search') and not PAPER_URL.match(str(s.get('url', ''))):
            bad.append(f"source [{s.get('n')}] {s.get('url')} is not a real paper URL (needs a numeric arXiv id like 2501.12948)")
            continue
        m = PAPER_URL.match(str(s.get('url', '')))
        if not m:
            continue
        real = fetch(m.group(1))
        if not real:
            continue  # unverifiable (network); never fail on this
        a, b = _norm_title(s.get('title', '')) - _STOP, _norm_title(real) - _STOP
        shared = len(a & b)
        if not a or not b or (shared < 2 and shared < 0.5 * min(len(a), len(b))):
            bad.append(f"source [{s.get('n')}] {s.get('url')} is titled {real!r} but the manifest says {s.get('title')!r}")
    if bad:
        raise RuntimeError('wrong paper id/title: ' + '; '.join(bad[:6]) +
                           '. Take ids only from tool results; re-search the intended paper with web_search/arxiv_search, '
                           'fix sources.json AND every sentence citing it (drop the claim if no matching source is found).')


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
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                             PREPARER_PATH: PREPARER_SOURCE.read_bytes()})
            _execute_ok(backend, f'python3 -m py_compile {VALIDATOR_PATH} {FINALIZER_PATH} {PREPARER_PATH}')
            agent = build_lead_agent(backend, model)
            config = {'recursion_limit': 1000, 'callbacks': [Progress()]}
            print('[phase 1/3] Plan and gather evidence', flush=True)
            result = agent.invoke({'messages': [{'role': 'user', 'content': build_prompt(topic) +
                '\nPHASE 1: Only plan, delegate >=3 independent researcher tasks in parallel, '
                'and read/check their notes. Gather foundational plus recent directly relevant sources '
                'covering >=3 source labels. Stop with the notes paths; do NOT write report or manifest yet.'}]}, config=config)
            print('[phase 2/3] Synthesize and write', flush=True)
            result = agent.invoke({'messages': [*result['messages'], {'role': 'user', 'content':
                'PHASE 2: Read the notes and merge only relevant supported sources into sources.json. '
                'Assign each source a positive integer n (1,2,3,...). An inline citation uses that n, '
                'NEVER an arxiv id, URL or title. Canonical URL and discovery-source labels must match. '
                'Then WRITE the full 1800-2400 word report BODY (it MUST exceed 1500 words: write each of the 4-5 '
                'thematic sections as 3-4 substantial paragraphs comparing methods, and verify with `wc -w`) with cited TL;DR, Background, '
                '3-6 thematic sections and exact heading Trends and open problems. '
                'Support every technical claim with its actual source evidence. '
                'Do not write a References heading or list at all. Stop after saving the body; '
                'do NOT finalize or run checker yet.'}]}, config=config)
            print('[phase 3/3] Finalize and verify evidence', flush=True)
            result = agent.invoke({'messages': [*result['messages'], {'role': 'user', 'content':
                f'PHASE 3: Execute python3 {PREPARER_PATH}, then python3 {FINALIZER_PATH}, then python3 {VALIDATOR_PATH}. '
                'The finalizer, not you, generates References at the END. If it fails, read the actual '
                'report and manifest and fix the reported issue; stop and return diagnostics after '
                'two failed attempts rather than guessing. Send exactly one citation-checker task '
                'with five literal report claim/URL pairs, including a foundational claim and any '
                'numbers; require supporting quotes. Apply supported corrections in the sandbox, '
                'rerun finalizer and validator after edits, finish todos and stop.'}]}, config=config)
            # Bounded repair turns retain the same sandbox and evidence. No host-side rewriting.
            for repair in range(3):
                try:
                    # Restore trusted scripts before the final gate; never trust an agent-edited validator.
                    upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                                     PREPARER_PATH: PREPARER_SOURCE.read_bytes()})
                    print(_execute_ok(backend, f'python3 {PREPARER_PATH} && python3 {FINALIZER_PATH}'), flush=True)
                    print(_execute_ok(backend, f'python3 {VALIDATOR_PATH}'), flush=True)
                    try:
                        audit_source_titles(json.loads(download(backend, [SOURCES_PATH]).get(SOURCES_PATH) or b'[]'))
                    except (ValueError, TypeError, AttributeError):
                        pass  # unreadable manifest is reported by save_outputs below
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
