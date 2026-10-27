import io
import json
from unittest import mock
import unittest
from urllib import error

from proj_rsi.dashboard import publish_snapshot as publisher


class Response:
    status = 201
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, limit): return b'{"ok":true}'


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.values = {'url': 'https://dashboard.example', 'service_token': 'private-service-value', 'ingest_token': 'private-ingest-value-' * 3}
        self.credentials = publisher.load_credentials(stream=io.StringIO(json.dumps(self.values) + '\n'))
        self.snapshot = {'schema_version': 1, 'generated_at': 1800000000., 'summary': {}}

    def test_credentials_only_stdin_or_env_and_repr_hides_them(self):
        credentials = publisher.load_credentials(environ={'DASHBOARD_URL': self.values['url'], 'DASHBOARD_SERVICE_TOKEN': self.values['service_token'], 'DASHBOARD_INGEST_TOKEN': self.values['ingest_token']})
        self.assertEqual(credentials.url, self.values['url'])
        self.assertNotIn(self.values['service_token'], repr(credentials))
        for value in ('bad-json-private-service-value', json.dumps({**self.values, 'command': 'run'}), json.dumps({**self.values, 'url': 'http://dashboard.example'}), json.dumps({**self.values, 'url': 'https://dashboard.example?token=private'}), json.dumps({**self.values, 'service_token': 'a\r\nb'}), json.dumps({**self.values, 'ingest_token': 'short'}), 'x' * 32769):
            with self.assertRaises(publisher.PublishError) as caught:
                publisher.load_credentials(stream=io.StringIO(value))
            self.assertEqual(str(caught.exception), 'Invalid publisher credential configuration')

    def test_publishes_only_ingest_with_both_headers_and_no_remote_body(self):
        opener = mock.Mock(); opener.open.return_value = Response()
        result = publisher.publish_once(self.credentials, builder=lambda: self.snapshot, opener=opener)
        upload = opener.open.call_args.args[0]
        self.assertEqual(upload.full_url, 'https://dashboard.example/_ingest')
        self.assertEqual(upload.method, 'POST')
        self.assertEqual(upload.get_header('Oai-sites-authorization'), 'Bearer ' + self.values['service_token'])
        self.assertEqual(upload.get_header('Authorization'), 'Bearer ' + self.values['ingest_token'])
        self.assertEqual(json.loads(upload.data), self.snapshot)
        self.assertEqual(result, {'generated_at': 1800000000., 'bytes': len(upload.data)})

    def test_builder_options_are_readonly_pass_through(self):
        builder = mock.Mock(return_value=self.snapshot); opener = mock.Mock(); opener.open.return_value = Response()
        publisher.publish_once(self.credentials, builder=builder, opener=opener, builder_kwargs={'agents_path': 'observation.json'})
        builder.assert_called_once_with(agents_path='observation.json')

    def test_remote_errors_never_expose_token_url_response_body_or_exception(self):
        for failure in (RuntimeError('private-service-value https://internal.example'), error.HTTPError('https://internal.example', 403, 'private-service-value', {}, io.BytesIO(b'private-ingest-value'))):
            opener = mock.Mock(); opener.open.side_effect = failure
            with self.assertRaises(publisher.PublishError) as caught:
                publisher.publish_once(self.credentials, builder=lambda: self.snapshot, opener=opener)
            self.assertNotIn('private', str(caught.exception)); self.assertNotIn('internal', str(caught.exception))

    def test_no_network_on_bad_schema_nonfinite_or_oversized_snapshot(self):
        opener = mock.Mock()
        for value in ({'schema_version': 2, 'generated_at': 1}, {'schema_version': True, 'generated_at': 1}, {'schema_version': 1, 'generated_at': True}, {'schema_version': 1, 'generated_at': 0}, {'schema_version': 1, 'generated_at': float('nan')}, {'schema_version': 1, 'generated_at': 1, 'body': 'x' * publisher.MAX_BYTES}):
            with self.assertRaises(publisher.PublishError):
                publisher.publish_once(self.credentials, builder=lambda: value, opener=opener)
        opener.open.assert_not_called()

    def test_redirect_handler_never_forwards_credentials(self):
        self.assertIsNone(publisher.NoRedirect().redirect_request(None, None, 302, 'Found', {}, 'https://attacker.example'))

    def test_main_once_never_prints_credentials_on_failure(self):
        output = io.StringIO(); errors = io.StringIO()
        with mock.patch('sys.stdin', io.StringIO(json.dumps(self.values) + '\n')), mock.patch('sys.stdout', output), mock.patch('sys.stderr', errors), mock.patch.object(publisher, 'publish_once', side_effect=publisher.PublishError('Snapshot publish connection failed')):
            self.assertEqual(publisher.main(['--once', '--credentials-stdin']), 1)
        self.assertNotIn('private', output.getvalue() + errors.getvalue())
        self.assertEqual(output.getvalue(), '')

    def test_main_success_and_interval_validation(self):
        with mock.patch('sys.stdin', io.StringIO(json.dumps(self.values) + '\n')), mock.patch('sys.stdout', io.StringIO()) as out, mock.patch.object(publisher, 'publish_once', return_value={'bytes': 123, 'generated_at': 1}):
            self.assertEqual(publisher.main(['--once', '--credentials-stdin']), 0)
            self.assertEqual(out.getvalue(), 'Snapshot published (123 bytes)\n')
        with mock.patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit): publisher.main(['--interval', 'nan'])


if __name__ == '__main__': unittest.main()
