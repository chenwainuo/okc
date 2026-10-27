"""Publish sanitized snapshots; credentials come only from runtime env or stdin.

python3 -m proj_rsi.dashboard.publish_snapshot --credentials-stdin --interval 10
The stdin mode accepts one JSON line with url/service_token/ingest_token, keeps
credentials in memory only, and never echoes them. Root owns daemon lifecycle.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib import error, parse, request

from proj_rsi.dashboard.snapshot import build_snapshot

MAX_BYTES = 2 * 1024 * 1024


class PublishError(Exception):
    """Contains only a fixed, non-sensitive diagnostic."""


@dataclass(repr=False, frozen=True)
class Credentials:
    url: str
    service_token: str
    ingest_token: str


def load_credentials(*, environ=None, stream=None):
    environ = os.environ if environ is None else environ
    if stream is not None:
        try:
            line = stream.readline(32769)
            if len(line) > 32768:
                raise ValueError()
            values = json.loads(line)
            if not isinstance(values, dict) or set(values) != {'url', 'service_token', 'ingest_token'}:
                raise ValueError()
        except Exception:
            raise PublishError('Invalid publisher credential configuration') from None
    else:
        values = {key: environ.get(name) for key, name in (
            ('url', 'DASHBOARD_URL'), ('service_token', 'DASHBOARD_SERVICE_TOKEN'),
            ('ingest_token', 'DASHBOARD_INGEST_TOKEN'))}
    if any(not isinstance(v, str) or not v or len(v) > 8192 or any(ord(c) < 33 or ord(c) > 126 for c in v)
           for v in values.values()):
        raise PublishError('Invalid publisher credential configuration')
    try:
        url = parse.urlsplit(values['url'])
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ('', '/'):
            raise ValueError()
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError()
        if len(values['ingest_token']) < 32 or len(values['ingest_token']) > 4096:
            raise ValueError()
    except Exception:
        raise PublishError('Invalid publisher credential configuration') from None
    return Credentials(values['url'].rstrip('/'), values['service_token'], values['ingest_token'])


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward either credential to a redirect target.
        return None


def publish_once(credentials, *, builder=build_snapshot, opener=None, builder_kwargs=None):
    try:
        snapshot = builder(**(builder_kwargs or {}))
        if type(snapshot.get('schema_version')) is not int or snapshot['schema_version'] != 1 or type(snapshot.get('generated_at')) not in (int, float) or not math.isfinite(snapshot['generated_at']) or snapshot['generated_at'] <= 0:
            raise ValueError()
        body = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        if len(body) > MAX_BYTES:
            raise PublishError('Snapshot exceeds upload limit')
    except PublishError:
        raise
    except Exception:
        raise PublishError('Snapshot construction failed') from None
    upload = request.Request(credentials.url + '/_ingest', data=body, method='POST', headers={
        'Content-Type': 'application/json',
        'OAI-Sites-Authorization': 'Bearer ' + credentials.service_token,
        'Authorization': 'Bearer ' + credentials.ingest_token,
        'Accept': 'application/json',
        'User-Agent': 'readonly-dashboard-publisher/1',
    })
    opener = request.build_opener(NoRedirect()) if opener is None else opener
    try:
        with opener.open(upload, timeout=15) as response:
            status = response.status
            # Success has a fixed tiny response. Never log or return remote body.
            response.read(4096)
            if status != 201:
                raise PublishError('Snapshot publish rejected')
    except PublishError:
        raise
    except error.HTTPError as exc:
        code = exc.code if isinstance(exc.code, int) and 100 <= exc.code <= 599 else 0
        exc.close()
        raise PublishError(f'Snapshot publish failed (HTTP {code})') from None
    except Exception:
        raise PublishError('Snapshot publish connection failed') from None
    return {'generated_at': snapshot['generated_at'], 'bytes': len(body)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--credentials-stdin', action='store_true')
    parser.add_argument('--scheduler-dir', type=Path)
    parser.add_argument('--agents-path', type=Path)
    parser.add_argument('--hardware-path', type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or not 1 <= args.interval <= 3600:
        parser.error('--interval must be between 1 and 3600 seconds')
    try:
        credentials = load_credentials(stream=sys.stdin if args.credentials_stdin else None)
    except PublishError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    options = {key: getattr(args, key) for key in ('scheduler_dir', 'agents_path', 'hardware_path') if getattr(args, key) is not None}
    while True:
        started = time.monotonic()
        try:
            result = publish_once(credentials, builder_kwargs=options)
            print(f"Snapshot published ({result['bytes']} bytes)", flush=True)
            succeeded = True
        except PublishError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            succeeded = False
        except Exception:
            print('Snapshot publisher failed; details withheld', file=sys.stderr, flush=True)
            succeeded = False
        if args.once:
            return 0 if succeeded else 1
        time.sleep(max(0.1, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
