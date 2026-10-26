"""generate pw24h/slack.py (the P4 decode-slack curve) from submission.py"""
from pathlib import Path
d = Path(__file__).resolve().parents[1]
s = (d / "submission.py").read_text()
assert s.count("SLACK = False") == 1
(d / "slack.py").write_text(s.replace("SLACK = False", "SLACK = True", 1))
