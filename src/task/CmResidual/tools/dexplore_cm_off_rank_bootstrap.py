"""CmResidual V1.21 Cm-off entrypoint for the unmodified DExplore runner.

The explicit zero coefficient is intentionally handled before importing the
existing DExplore bootstrap.  This provides a stable, testable identity for
the future Cm runner while making accidental Cm construction impossible in the
Cm-off parity branch.
"""
from __future__ import annotations

import argparse
import sys

import dexplore_ddp_rank_bootstrap


def parse_cm_off_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cm-distill-coef", type=float, required=True)
    args, passthrough = parser.parse_known_args(argv)
    if args.cm_distill_coef != 0.0:
        raise ValueError("Cm-off bootstrap only accepts --cm-distill-coef 0")
    return args, passthrough


def _assert_cm_not_imported() -> None:
    forbidden = tuple(name for name in sys.modules if "cmv2" in name.lower() or "cm_residual" in name.lower())
    if forbidden:
        raise RuntimeError(f"Cm-off bootstrap refuses pre-imported Cm modules: {forbidden}")


def main(argv=None) -> None:
    _, passthrough = parse_cm_off_args(argv)
    _assert_cm_not_imported()
    dexplore_ddp_rank_bootstrap.main(passthrough)


if __name__ == "__main__":
    main()
