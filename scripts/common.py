# =============================================================================
#   ai4earthquake :: scripts
# =============================================================================
#   Module       :: common.py -- Shared argument parsing for the stage scripts
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Shared argument parsing for the stage scripts."""
import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))                 # works without pip install -e .

from labquake import io  # noqa: E402


def parse(description, extra=None):
    """--config <yaml> plus any extra arguments; returns (args, cfg)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="configs/<system>.yaml")
    if extra:
        extra(p)
    args = p.parse_args()
    return args, io.load_config(args.config)
