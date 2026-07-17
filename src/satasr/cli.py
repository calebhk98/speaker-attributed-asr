"""``satasr`` command-line entry point — one command to run the whole pipeline.

    satasr run --config config.example.toml        # execute the configured stages
    satasr run --config config.example.toml --dry-run   # just show the plan

Installed as the ``satasr`` console script (see pyproject ``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence

from satasr.run.loader import load_config
from satasr.run.pipeline import PipelineError, run_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args and run; return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    parser.error("no command given")  # argparse exits; unreachable return below
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satasr", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_parser = subcommands.add_parser(
        "run", help="run the pipeline from a config file"
    )
    run_parser.add_argument(
        "--config", required=True, help="path to a TOML config file"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="print the stage plan without executing"
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        results = run_pipeline(config, dry_run=args.dry_run, on_stage=_announce)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {stage: _describe(value) for stage, value in results.items()}, indent=2
        )
    )
    return 0


def _announce(stage: str) -> None:
    print(f"==> {stage}", file=sys.stderr)


def _describe(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
