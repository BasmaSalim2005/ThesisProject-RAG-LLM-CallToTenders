"""CLI: normalize requirements JSON and write formated_requirements.json next to this script.

Same logic as easy_tender.format_requirements.py (used by the web app).

Example (from repo root):
  python Step2/formating/export_formated_requirements.py path/to/extraction.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from easy_tender.format_requirements import format_requirements_payload

DEFAULT_OUTPUT = _HERE / "formated_requirements.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write normalized requirements JSON into Step2/formating/ (this folder)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSON: {'requirements': {...}} from extraction, or the three-category dict",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file (default: {DEFAULT_OUTPUT})",
    )
    og = parser.add_mutually_exclusive_group()
    og.add_argument(
        "--openrouter",
        action="store_true",
        help="Force Mistral/OpenRouter polish (needs OPENROUTER_API_KEY)",
    )
    og.add_argument(
        "--no-openrouter",
        action="store_true",
        help="Skip OpenRouter even if OPENROUTER_API_KEY is set",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input.resolve()}")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Input JSON must be an object at the top level.")

    raw_req = data.get("requirements")
    if not isinstance(raw_req, dict):
        raw_req = data

    refine = None
    if args.no_openrouter:
        refine = False
    elif args.openrouter:
        refine = True
    out = format_requirements_payload(raw_req, openrouter_refine=refine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n = sum(len(v) for v in out.values())
    print(f"Wrote {args.output.resolve()} ({n} requirements)")


if __name__ == "__main__":
    main()
