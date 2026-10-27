"""Start the observation publisher detached, with credentials only on stdin."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from proj_rsi.dashboard.publish_snapshot import load_credentials

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.interval <= 3600:
        parser.error('interval must be between 1 and 3600 seconds')
    credentials = load_credentials(stream=sys.stdin)
    state = ROOT / 'results/proj-rsi/dashboard'
    state.mkdir(parents=True, exist_ok=True)
    pidfile = state / 'publisher.pid'
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'proj_rsi.dashboard.publish_snapshot' in command:
                raise SystemExit('Publisher already running')
        except (FileNotFoundError, ValueError):
            pass
    with (state / 'publisher.log').open('ab', buffering=0) as log:
        child = subprocess.Popen(
            [sys.executable, '-u', '-m', 'proj_rsi.dashboard.publish_snapshot',
             '--credentials-stdin', '--interval', str(args.interval)],
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=ROOT,
        )
        payload = json.dumps({'url': credentials.url,
                              'service_token': credentials.service_token,
                              'ingest_token': credentials.ingest_token}) + '\n'
        child.stdin.write(payload.encode())
        child.stdin.close()
    temporary = pidfile.with_suffix('.tmp')
    temporary.write_text(str(child.pid) + '\n')
    os.replace(temporary, pidfile)
    print(json.dumps({'publisher_pid': child.pid, 'interval_seconds': args.interval}))


if __name__ == '__main__':
    main()
