"""Literal scheduler entrypoint for the independently frozen PW42 narrow lab."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    runpy.run_module('proj_rsi.candidates.pw42.lab',run_name='__main__')
