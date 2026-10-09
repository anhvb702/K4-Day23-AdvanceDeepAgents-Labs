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
        body = prepare(report, sources)
        with open(report_path, 'w', encoding='utf-8') as handle:
            handle.write(body)
    except (OSError, ValueError) as exc:
        print(f'PREPARATION FAILED: {exc}')
        return 1
    print('PREPARED: factual prose preserved; run finalize_citations.py next')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
