"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["sources.json must be a nonempty list"]
    by_n, urls = {}, set()
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append("source must be an object")
            continue
        n, url = entry.get("n"), entry.get("url")
        if type(n) is not int or n < 1:
            problems.append(f"invalid source number: {n!r}")
            continue
        if n in by_n:
            problems.append(f"duplicate source number [{n}]")
        by_n[n] = entry
        if not isinstance(url, str) or not re.fullmatch(r"https?://[^\s]+", url):
            problems.append(f"source [{n}] has invalid URL")
        elif url in urls:
            problems.append(f"duplicate URL: {url}")
        else:
            urls.add(url)
    # Mask code before recognizing headings or citations, retaining line boundaries.
    code = re.compile(r"(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[^\n]*(?:\n|\Z)|(?m:^(?: {4}|\t)[^\n]*(?:\n|\Z))|(`+)(?!`).*?(?<!`)\2(?!`)", re.S)
    report_text = code.sub(lambda m: '\n' * m.group(0).count('\n'), report_text)
    headings = list(re.finditer(r"(?m)^##[ \t]+References[ \t]*\r?$", report_text))
    if len(headings) != 1:
        return problems + ["report must have exactly one ## References heading"]
    body, references = report_text[:headings[0].start()], report_text[headings[0].end():]
    body = re.sub(r"```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`[^`\n]*`", "", body, flags=re.S)
    body = re.sub(r"\[[^\]\n]*\]\s*\([^\n]*?\)", "", body)
    cited = set()
    for match in re.finditer(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", body):
        for part in re.split(r"\s*,\s*", match.group(1)):
            span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
            if span:
                a, b = map(int, span.groups())
                if not 0 <= b - a <= 200:
                    problems.append(f"invalid citation range [{part}]")
                else:
                    cited.update(range(a, b + 1))
            elif part.isdigit():
                cited.add(int(part))
            else:
                problems.append(f"invalid citation group [{match.group(1)}]")
    for n in sorted(cited - by_n.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_n.keys() - cited):
        problems.append(f"source [{n}] never cited")
    counts = Counter()
    for line in references.splitlines():
        if not line.strip():
            continue
        match = re.match(r"^\[(\d+)\]\s+", line)
        if not match:
            problems.append("reference line must start with [n]")
            continue
        n = int(match.group(1))
        counts[n] += 1
        found_urls = re.findall(r"https?://[^\s<>]+", line)
        if n not in by_n:
            problems.append(f"reference [{n}] is not a source")
        elif len(found_urls) != 1 or found_urls[0] != by_n[n].get("url"):
            problems.append(f"reference [{n}] must contain exactly its source URL")
    for n in by_n:
        if counts[n] != 1:
            problems.append(f"source [{n}] needs exactly one reference line (found {counts[n]})")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
