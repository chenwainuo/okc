"""Verify private access and live read-only data; read service credentials on stdin."""
import argparse
import json
from pathlib import Path
import sys
import time
from urllib import error, request

from proj_rsi.dashboard.publish_snapshot import NoRedirect, load_credentials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    credentials = load_credentials(stream=sys.stdin)
    opener = request.build_opener(NoRedirect())

    def fetch(path, authenticated=True, method='GET'):
        headers = {'OAI-Sites-Authorization': 'Bearer ' + credentials.service_token} if authenticated else {}
        req = request.Request(credentials.url + path, headers=headers, method=method)
        try:
            with opener.open(req, timeout=20) as response:
                return response.status, response.read(2 * 1024 * 1024)
        except error.HTTPError as exc:
            status, body = exc.code, exc.read(2 * 1024 * 1024)
            exc.close()
            return status, body

    public_status, _ = fetch('/api/snapshot', authenticated=False)
    assert public_status in (301, 302, 303, 307, 308, 401, 403), 'Private access boundary failed'
    html_status, html = fetch('/')
    assert html_status == 200 and b'/api/snapshot' in html and b'<title>' in html
    data_status, body = fetch('/api/snapshot')
    assert data_status == 200, 'Snapshot read failed'
    snapshot = json.loads(body)
    age = time.time() - snapshot['generated_at']
    assert -30 < age < 45, 'Snapshot publisher is stale'
    assert snapshot['summary']['submissions'] >= 10 and snapshot['summary']['checks'] >= 40
    write_status, _ = fetch('/api/snapshot', method='POST')
    ingest_status, _ = fetch('/_ingest', method='POST')
    assert write_status == 405 and ingest_status == 401, 'Read-only boundary failed'
    report = {'verified_at': time.time(), 'private_access_status': public_status, 'html_status': html_status,
                      'snapshot_status': data_status, 'snapshot_age_seconds': round(age, 2),
                      'queue_sequence': snapshot['queue']['sequence'], 'summary': snapshot['summary'],
                      'visitor_write_status': write_status, 'unauthorized_ingest_status': ingest_status}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
