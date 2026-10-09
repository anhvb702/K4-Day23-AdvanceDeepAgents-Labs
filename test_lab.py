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

    def test_arxiv_atom_normalization_and_spacing(self):
        import json
        import httpx
        from unittest.mock import patch
        from tools import arxiv_search
        xml = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>http://arxiv.org/abs/2501.00001v3</id><published>2025-01-02T00:00:00Z</published>
          <title> A  paper\n title </title><summary> Evidence  summary </summary></entry></feed>'''
        with patch('tools._request', return_value=httpx.Response(200, text=xml)) as request, \
             patch('tools._last_arxiv', 100), patch('tools.time.monotonic', return_value=101), \
             patch('tools.time.sleep') as sleep:
            result = json.loads(arxiv_search.invoke({'query': 'all:world OR "model"', 'max_results': 100}))
            self.assertEqual(result[0]['id'], '2501.00001')
            self.assertEqual(result[0]['url'], 'https://arxiv.org/abs/2501.00001')
            self.assertEqual(result[0]['title'], 'A paper title')
            self.assertEqual(request.call_args.kwargs['params']['search_query'], 'all:world AND all:model')
            self.assertEqual(request.call_args.kwargs['params']['max_results'], 30)
            sleep.assert_called_once_with(2)

    def test_hf_daily_sort_filter_and_no_results(self):
        import json
        import httpx
        from unittest.mock import patch
        from tools import hf_daily_papers
        data = [{'paper': {'id': '1', 'title': 'World model A', 'upvotes': 2}},
                {'paper': {'id': '2', 'title': 'World model B', 'upvotes': 5}},
                {'paper': {'id': '3', 'title': 'Unrelated', 'upvotes': 10}}]
        with patch('tools._request', return_value=httpx.Response(200, json=data)):
            result = json.loads(hf_daily_papers.invoke({'keyword': 'world'}))
            self.assertEqual([r['id'] for r in result], ['2', '1'])
            self.assertEqual(hf_daily_papers.invoke({'keyword': 'missing'}), 'NO RESULTS')
            self.assertTrue(hf_daily_papers.invoke({'date': 'invalid'}).startswith('ERROR:'))

    def test_hf_normalization(self):
        from tools import _hf_records
        result = _hf_records([{'paper': {'id': '2501.00001', 'title': ' A  paper ', 'ai_summary': 'short', 'summary': 'long', 'upvotes': 3}}, {'paper': {}}], True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['summary'], 'short')
        self.assertEqual(result[0]['url'], 'https://huggingface.co/papers/2501.00001')

    def test_short_normalized_and_url_encoded_secrets_are_redacted(self):
        from unittest.mock import patch
        from tools import redact
        with patch.dict('os.environ', {'EXA_API_KEY': ' ab/+ '}):
            result = redact('key=ab/+&encoded=ab%2F%2B')
            self.assertNotIn('ab/+', result)
            self.assertNotIn('ab%2F%2B', result)

    def test_tools_errors_redact_keys(self):
        from unittest.mock import patch
        from tools import web_search, arxiv_search
        with patch.dict('os.environ', {'EXA_API_KEY': 'private-test-value'}), patch('tools._exa', side_effect=ValueError('private-test-value')):
            text = web_search.invoke({'query': 'world models'})
            self.assertTrue(text.startswith('ERROR:'))
            self.assertNotIn('private-test-value', text)
        self.assertEqual(arxiv_search.invoke({'query': '\" : AND OR'}), 'NO RESULTS')


class PreparationTests(unittest.TestCase):
    def test_required_heading_case_is_canonicalized(self):
        from prepare_report import prepare
        text = '# Survey\n## Trends and Open Problems\nSupported prose [1].\n'
        self.assertIn('## Trends and open problems', prepare(text, []))

    def test_premature_references_and_identifier_citations_are_normalized(self):
        from prepare_report import prepare
        from finalize_citations import finalize
        sources = [{'n': 1, 'id': '1803.10122', 'url': 'https://arxiv.org/abs/1803.10122', 'title': 'World Models', 'source': 'web'}]
        text = '# Survey\n## References\n[1] stale\n## TL;DR\nClaim [1803.10122]. `Example [1803.10122]`\n## Background\nClaim [1].\n'
        body = prepare(text, sources)
        self.assertNotIn('## References', body)
        self.assertIn('Claim [1].', body)
        self.assertIn('`Example [1803.10122]`', body)
        report, manifest, problems = finalize(body, sources)
        self.assertEqual(problems, [])
        self.assertEqual(check(report, manifest), [])


class AgentTests(unittest.TestCase):
    def test_optional_research_model_does_not_weaken_checker(self):
        from unittest.mock import patch
        from agents import build_subagents
        with patch.dict('os.environ', {'LAB_RESEARCH_MODEL': 'openai:gpt-4.1-mini'}):
            agents = build_subagents()
        self.assertEqual(agents[0]['model'], 'openai:gpt-4.1-mini')
        self.assertNotIn('model', agents[1])

    def test_only_bounded_named_subagents_and_no_implicit_general_purpose(self):
        from unittest.mock import patch
        from langchain_openai import ChatOpenAI
        from agents import build_lead_agent, build_subagents
        agents = build_subagents()
        self.assertEqual({a['name'] for a in agents}, {'researcher', 'citation-checker'})
        for agent in agents:
            self.assertEqual(len(agent['middleware']), 2)
        with patch('agents.create_deep_agent') as create, patch('agents.register_harness_profile') as register:
            build_lead_agent(None, ChatOpenAI(model='gpt-4o-mini', api_key='test'))
            self.assertFalse(register.call_args.args[1].general_purpose_subagent.enabled)
            self.assertEqual(len(create.call_args.kwargs['middleware']), 3)


class ResearchTests(unittest.TestCase):
    def test_report_quality_rejects_short_generic_survey(self):
        from research import check_report_quality
        self.assertTrue(check_report_quality('# Survey\n## TL;DR\nShort claim.\n## References', []))
        sources = [{'date': '2018-01-01'}, {'date': '2026-01-01'}]
        report = '# Survey\n## TL;DR\n## Background\n## Theme A\n## Theme B\n## Theme C\n## Trends and open problems\n' + 'evidence ' * 1250 + '\n## References\n'
        self.assertEqual(check_report_quality(report, sources), [])

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
        sources[0]['date'], sources[1]['date'] = '2018-01-01', '2026-01-01'
        report = ('# Survey\n## TL;DR\n## Background\n## Theme A\n## Theme B\n## Theme C\n## Trends and open problems\n' + 'evidence ' * 1250 + ' Claim [1][2][3].\n\n## References\n' + '\n'.join(f"[{s['n']}] Title. {s['url']} (n.d.)" for s in sources) + '\n').encode()
        manifest = json.dumps(sources, indent=3).encode()
        message = AIMessage(content='', tool_calls=[{'name': 'task', 'args': {'subagent_type': 'researcher'}, 'id': str(i)} for i in range(3)] + [{'name': 'task', 'args': {'subagent_type': 'citation-checker'}, 'id': 'check'}])
        with TemporaryDirectory() as directory, patch('research.download', return_value={REPORT_PATH: report, SOURCES_PATH: manifest}):
            path = save_outputs(None, 'test', [message], 1, 'test', Path(directory))
            self.assertEqual(path.read_bytes(), report)
            self.assertEqual(path.with_suffix('.sources.json').read_bytes(), manifest)
            self.assertEqual(json.loads(path.with_suffix('.meta.json').read_text())['n_sources'], 3)
            broken = [dict(sources[0], id='wrong-1'), dict(sources[1], id='wrong-2'), sources[2]]
            with patch('research.download', return_value={REPORT_PATH: report, SOURCES_PATH: json.dumps(broken).encode()}):
                with self.assertRaises(RuntimeError) as caught:
                    save_outputs(None, 'test', [message], 1, 'test', Path(directory))
                self.assertIn('source [1]', str(caught.exception))
                self.assertIn('source [2]', str(caught.exception))
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

    def test_main_repairs_invalid_artifacts_inside_same_sandbox(self):
        from contextlib import nullcontext
        from unittest.mock import patch, Mock
        from research import main
        agent = Mock()
        agent.invoke.return_value = {'messages': []}
        with patch('research.make_model'), patch('research.open_sandbox', return_value=nullcontext(Mock())), \
             patch('research.build_lead_agent', return_value=agent), patch('research.upload'), \
             patch('research._execute_ok', return_value='OK'), \
             patch('research.save_outputs', side_effect=[RuntimeError('source URL does not match'), 'reports/test.md']):
            self.assertEqual(main('test'), 0)
            self.assertEqual(agent.invoke.call_count, 4)
            for i in range(3):
                self.assertIn(f'PHASE {i + 1}', str(agent.invoke.call_args_list[i].args[0]))
            self.assertIn('source URL does not match', str(agent.invoke.call_args.args[0]))

    def test_main_stops_after_two_repairs_and_empty_topic_is_usage_error(self):
        from contextlib import nullcontext
        from unittest.mock import patch, Mock
        from research import main
        agent = Mock()
        agent.invoke.return_value = {'messages': []}
        with patch('research.make_model'), patch('research.open_sandbox', return_value=nullcontext(Mock())), \
             patch('research.build_lead_agent', return_value=agent), patch('research.upload'), \
             patch('research._execute_ok', return_value='OK'), \
             patch('research.save_outputs', side_effect=RuntimeError('bad artifacts')):
            self.assertEqual(main('test'), 1)
            self.assertEqual(agent.invoke.call_count, 5)
        self.assertEqual(main(''), 2)

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
