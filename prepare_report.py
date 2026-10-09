"""Deterministic citation preflight; runs INSIDE sandbox before the provided finalizer.

Only removes generated References blocks and converts unambiguous paper-ID citations
into manifest numbers. Never adds or edits factual prose. Standard library only.
"""
import json
import re
import sys

REPORT = '/tmp/work/report/report.md'
SOURCES = '/tmp/work/research/sources.json'
_CODE = re.compile(r'(```.*?```|~~~.*?~~~|`[^`\n]*`)', re.S)


_LABELS = {'arxiv_search': 'arxiv', 'hf_daily_papers': 'hf-daily', 'hf_search_papers': 'hf-search',
           'web_search': 'web', 'web_fetch': 'web'}
_PAPER_ID = re.compile(r'(?<!\d)(\d{4}\.\d{4,5})(?!\d)')
_PREFIX = {'arxiv': 'https://arxiv.org/abs/', 'hf-daily': 'https://huggingface.co/papers/',
           'hf-search': 'https://huggingface.co/papers/'}


def normalize_sources(sources):
    """Canonical form only: tool-name labels -> family labels; arXiv/HF id without vN and /abs|/papers URL.

    The discovery family is never changed, so provenance cannot be relabelled to meet a quota."""
    for source in sources:
        source['source'] = _LABELS.get(source.get('source'), source.get('source'))
        prefix = _PREFIX.get(source['source'])
        match = _PAPER_ID.search(str(source.get('id', ''))) or _PAPER_ID.search(str(source.get('url', '')))
        if prefix and match:
            source['id'], source['url'] = match.group(1), prefix + match.group(1)
    return sources


def prepare(report, sources):
    segments = _CODE.split(report)
    identifiers = {}
    for source in sources:
        identifier, number = str(source.get('id', '')), source.get('n')
        if identifier and not identifier.isdigit() and type(number) is int:
            identifiers.setdefault(identifier, set()).add(number)
    for i in range(0, len(segments), 2):
        segment = re.sub(r'(?ms)^##[ \t]+References[ \t]*\r?\n.*?(?=^##[ \t]+|\Z)', '', segments[i])
        for identifier, numbers in identifiers.items():
            if len(numbers) == 1:
                segment = re.sub(r'\[' + re.escape(identifier) + r'\](?!\()',
                                 f'[{next(iter(numbers))}]', segment)
        for heading in ('TL;DR', 'Background', 'Trends and open problems'):
            segment = re.sub(r'(?im)^##[ \t]+' + re.escape(heading) + r'[ \t]*\r?$',
                             '## ' + heading, segment)
        segments[i] = segment
    return ''.join(segments).rstrip() + '\n'


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding='utf-8') as handle:
            report = handle.read()
        with open(sources_path, encoding='utf-8') as handle:
            sources = json.load(handle)
        if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
            raise ValueError('manifest must be a list of objects')
        sources = normalize_sources(sources)
        body = prepare(report, sources)
        with open(report_path, 'w', encoding='utf-8') as handle:
            handle.write(body)
        with open(sources_path, 'w', encoding='utf-8') as handle:
            json.dump(sources, handle, ensure_ascii=False, indent=2)
    except (OSError, ValueError) as exc:
        print(f'PREPARATION FAILED: {exc}')
        return 1
    print('PREPARED: factual prose preserved; run finalize_citations.py next')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
