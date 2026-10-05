"""Command-line entry point: `ssve weekly`, `ssve ask`, `ssve compare`."""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np


def _cmd_weekly(args: argparse.Namespace) -> int:
    from .orchestrator import Orchestrator, load_config

    report = Orchestrator(load_config(args.config)).run_weekly()
    print(f"Ingested {report.ingested} docs, indexed {report.indexed}.")
    for p in report.outputs:
        print(f"  wrote {p}")
    for e in report.errors:
        print(f"  ERROR {e}", file=sys.stderr)
    return 1 if report.errors and not report.ingested else 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from .orchestrator import Orchestrator, load_config

    ans = Orchestrator(load_config(args.config)).pipeline.answer(args.question)
    print(ans.text)
    if ans.citations:
        print("\nSources:\n" + "\n".join(f"  https://doi.org/{d}" for d in ans.citations))
    return 0 if ans.grounded else 2


def _load_seq(path: str, args: argparse.Namespace) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    from .vision import PoseExtractor, clean_sequence

    seq, _ = PoseExtractor(weights=args.weights).extract(path, stride=args.stride)
    return clean_sequence(seq)


def _cmd_compare(args: argparse.Namespace) -> int:
    from .vision import compare_movements

    ref, ath = _load_seq(args.reference, args), _load_seq(args.athlete, args)
    has_conf = None if args.dims == "auto" else args.dims == "2d"
    print(compare_movements(ref, ath, has_conf=has_conf).to_markdown())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssve", description="Sports Science & Vision AI Engine")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("weekly", help="ingest -> index -> digest + memos (+ email)")
    w.add_argument("--config", default="config/manifest.yaml")
    w.set_defaults(func=_cmd_weekly)

    a = sub.add_parser("ask", help="DOI-grounded question answering over the index")
    a.add_argument("question")
    a.add_argument("--config", default="config/manifest.yaml")
    a.set_defaults(func=_cmd_ask)

    c = sub.add_parser("compare", help="compare athlete vs reference movement (video or .npy keypoints)")
    c.add_argument("reference")
    c.add_argument("athlete")
    c.add_argument("--weights", default="yolo11n-pose.pt")
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--dims", choices=["auto", "2d", "3d"], default="auto",
                   help="2d = (x, y, conf) keypoints, 3d = (X, Y, Z)")
    c.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
