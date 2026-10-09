"""Offline regression checks: python -m unittest test_lab -v."""
import unittest
from check_citations import check


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{"n": 1, "url": "https://example.org/a"},
                        {"n": 2, "url": "https://example.org/b"}]
        self.refs = "\n## References\n[1] A. https://example.org/a\n[2] B. https://example.org/b\n"

    def test_valid_and_grouped_citations(self):
        self.assertEqual(check("Evidence [1-2]." + self.refs, self.sources), [])

    def test_code_and_links_are_not_citations(self):
        self.assertTrue(check("`[1]`\n```\n[2]\n```\n[1](https://x.org)" + self.refs, self.sources))

    def test_unknown_missing_and_duplicate_references(self):
        for text in ["Claim [1][3]." + self.refs,
                     "Claim [1][2].\n## References\n[1] https://example.org/a\n",
                     "Claim [1][2]." + self.refs + "[1] https://example.org/a\n"]:
            with self.subTest(text=text):
                self.assertTrue(check(text, self.sources))

    def test_bad_urls_and_multiple_urls(self):
        self.assertTrue(check("Claims [1][2]." + self.refs.replace("https://example.org/a", "https://wrong.org"), self.sources))
        self.assertTrue(check("Claims [1][2]." + self.refs.replace("[1] A.", "[1] https://extra.org A."), self.sources))
        self.assertTrue(check("Claims [1][2]." + self.refs, [self.sources[0], dict(self.sources[1], url=self.sources[0]['url'])]))

    def test_code_heading_and_double_backticks_are_ignored(self):
        text = 'Evidence [1][2].\n```markdown\n## References\n[9]\n```\n``[8]``' + self.refs
        self.assertEqual(check(text, self.sources), [])

    def test_malformed_group_returns_issue(self):
        self.assertTrue(check('Claim [1-2-3].' + self.refs, self.sources))

    def test_literal_url_punctuation_is_preserved(self):
        sources = [{'n': 1, 'url': 'https://example.org/a_(b);'}]
        self.assertEqual(check('Claim [1].\n## References\n[1] A. https://example.org/a_(b); (n.d.)', sources), [])

    def test_invalid_source_schema(self):
        for sources in [[], {}, [None], [{"n": True, "url": "bad"}], [dict(self.sources[0]), dict(self.sources[0])]]:
            self.assertTrue(check("Claim [1].\n## References\n[1] https://example.org/a", sources))


class ToolTests(unittest.TestCase):
    def test_retry_backoff_cap_and_last_attempt(self):
        from unittest.mock import patch
        from tools import with_retry, RetryableError
        calls = []
        def fail():
            calls.append(1)
            raise RetryableError('busy')
        with patch('tools.time.sleep') as sleep, patch('tools.random.uniform', return_value=0):
            with self.assertRaises(RetryableError):
                with_retry(fail, attempts=4, base=2, cap=3)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 3, 3])
            self.assertEqual(len(calls), 4)

    def test_retry_after_and_nonretryable(self):
        from unittest.mock import patch, Mock
        from tools import with_retry, RetryableError
        with patch('tools.time.sleep') as sleep:
            fn = Mock(side_effect=[RetryableError('busy', 2), 'ok'])
            self.assertEqual(with_retry(fn), 'ok')
            sleep.assert_called_once_with(2)
            with self.assertRaises(ValueError):
                with_retry(Mock(side_effect=ValueError('invalid')))

    def test_exa_sse_rate_limit_and_jsonrpc(self):
        import httpx
        from unittest.mock import patch
        from tools import _exa
        limited = httpx.Response(200, text='data: {"result":{"_meta":{"rateLimitExceeded":true},"content":[{"type":"text","text":"rate limit"}]}}\n\n')
        success = httpx.Response(200, text='data: {"result":{"content":[{"type":"text","text":"real evidence"}]}}\n\n')
        with patch('tools._request', side_effect=[limited, success]), patch('tools.time.sleep'):
            self.assertEqual(_exa('web_search_exa', {}), 'real evidence')
        ordinary = httpx.Response(200, json={'result': {'_meta': {'rateLimitExceeded': False, 'rateLimit': {'remaining': 10}}, 'content': [{'type': 'text', 'text': 'A paper about rate limits'}]}})
        with patch('tools._request', return_value=ordinary):
            self.assertEqual(_exa('web_search_exa', {}), 'A paper about rate limits')
        bad = httpx.Response(200, json={'error': {'code': -1, 'message': 'invalid arguments'}})
        with patch('tools._request', return_value=bad):
            with self.assertRaises(ValueError):
                _exa('web_fetch_exa', {})

    def test_http_transient_classification(self):
        import httpx
        from unittest.mock import patch
        from tools import _request, RetryableError
        for status in (429, 500, 502, 503, 504):
            response = httpx.Response(status, headers={'Retry-After': '7'}, request=httpx.Request('GET', 'https://example.org'))
            with patch('tools.httpx.request', return_value=response), self.assertRaises(RetryableError) as caught:
                _request('GET', 'https://example.org')
            self.assertEqual(caught.exception.retry_after, 7)
        response = httpx.Response(401, request=httpx.Request('GET', 'https://example.org'))
        with patch('tools.httpx.request', return_value=response), self.assertRaises(httpx.HTTPStatusError):
            _request('GET', 'https://example.org')
        with patch('tools.httpx.request', side_effect=httpx.ConnectError('offline')), self.assertRaises(RetryableError):
            _request('GET', 'https://example.org')

    def test_hf_normalization(self):
        from tools import _hf_records
        result = _hf_records([{'paper': {'id': '2501.00001', 'title': ' A  paper ', 'ai_summary': 'short', 'summary': 'long', 'upvotes': 3}}, {'paper': {}}], True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['summary'], 'short')
        self.assertEqual(result[0]['url'], 'https://huggingface.co/papers/2501.00001')

    def test_tools_errors_redact_keys(self):
        from unittest.mock import patch
        from tools import web_search, arxiv_search
        with patch.dict('os.environ', {'EXA_API_KEY': 'private-test-value'}), patch('tools._exa', side_effect=ValueError('private-test-value')):
            text = web_search.invoke({'query': 'world models'})
            self.assertTrue(text.startswith('ERROR:'))
            self.assertNotIn('private-test-value', text)
        self.assertEqual(arxiv_search.invoke({'query': '\" : AND OR'}), 'NO RESULTS')


class ResearchTests(unittest.TestCase):
    def test_slug_and_usage_summary(self):
        from research import slugify, summarize
        from langchain_core.messages import AIMessage
        self.assertEqual(slugify('../../x'), 'x')
        self.assertEqual(slugify(''), 'topic')
        self.assertLessEqual(len(slugify('x' * 100)), 60)
        message = AIMessage(content='', tool_calls=[{'name': 'task', 'args': {}, 'id': 'a'}],
                            usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})
        result = summarize([message], 1.25, 'test')
        self.assertEqual(result['subagent_calls'], 1)
        self.assertEqual(result['tokens'], {'input': 10, 'output': 5})

    def test_successful_download_preserves_exact_bytes(self):
        import json
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from langchain_core.messages import AIMessage
        from research import save_outputs, REPORT_PATH, SOURCES_PATH
        sources = [dict(n=1, id='2501.00001', url='https://arxiv.org/abs/2501.00001', source='arxiv'),
                   dict(n=2, id='2501.00002', url='https://huggingface.co/papers/2501.00002', source='hf-search'),
                   dict(n=3, id='web', url='https://example.org/a', source='web')]
        report = ('Claim [1][2][3].\n\n## References\n' + '\n'.join(f"[{s['n']}] Title. {s['url']} (n.d.)" for s in sources) + '\n').encode()
        manifest = json.dumps(sources, indent=3).encode()
        message = AIMessage(content='', tool_calls=[{'name': 'task', 'args': {'subagent_type': 'researcher'}, 'id': str(i)} for i in range(3)])
        with TemporaryDirectory() as directory, patch('research.download', return_value={REPORT_PATH: report, SOURCES_PATH: manifest}):
            path = save_outputs(None, 'test', [message], 1, 'test', Path(directory))
            self.assertEqual(path.read_bytes(), report)
            self.assertEqual(path.with_suffix('.sources.json').read_bytes(), manifest)
            self.assertEqual(json.loads(path.with_suffix('.meta.json').read_text())['n_sources'], 3)
            original = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            replace = Path.replace
            calls = []
            def fail_second(source, target):
                calls.append(source)
                if len(calls) == 2:
                    raise OSError('simulated disk failure')
                return replace(source, target)
            with patch.object(Path, 'replace', fail_second), self.assertRaises(OSError):
                save_outputs(None, 'test', [message], 2, 'test', Path(directory))
            self.assertEqual({p.name: p.read_bytes() for p in Path(directory).iterdir()}, original)

    def test_failed_download_writes_nothing(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from research import save_outputs, REPORT_PATH, SOURCES_PATH
        with TemporaryDirectory() as directory, patch('research.download', return_value={REPORT_PATH: b'', SOURCES_PATH: b'[]'}):
            with self.assertRaises(RuntimeError):
                save_outputs(None, 'topic', [], 1, 'test', Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
