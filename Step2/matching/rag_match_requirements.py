"""Match tender requirements to admin documents using an existing Chroma store.

1) Loads requirements JSON (flat or nested).
2) Opens Chroma built with ``Step2/embedding/embed_admin_chroma`` (same --chroma-dir).
3) Retrieves best matching files per requirement (or marks rows as ``skipped`` when a scope
   is disabled, e.g. no groupement).
4) Writes ``{ "toggles": {...}, "matches": [...] }``; each item has ``matches`` or
   ``skipped`` + ``condition_tags``.

Use ``--toggles-file`` and/or ``--no-groupement``, ``--no-hors-maroc``, etc. to mirror checkboxes
for optional situations. See ``Step2/matching/condition_toggles`` for how lines are auto-tagged from French text.

For a PDF: ``python -m Step2.reporting.match_report_pdf`` (or ``python Step2/match_report_pdf.py``)

Build embeddings first (Chroma default folder is **project root** ``chroma_admin_summary`` when you run from the repo root):
  python Step2/indexing/index_admin_docs_summarized.py
  python -m Step2.embedding.embed_admin_chroma --admin-index data/admin/... --chroma-dir chroma_admin_summary --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Step2.matching.condition_toggles import add_condition_toggle_args, build_toggles_from_parsed_args, iter_toggles_file_lines
from Step2.matching.match_postprocess import apply_declaration_bundles, apply_one_unused_match_per_requirement
from Step2.matching.rag_chroma_utils import flatten_requirements, load_chroma_db, run_matching

_ST2 = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ST2.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match requirements to admin files via Chroma RAG (with optional condition toggles).",
    )
    parser.add_argument(
        "--requirements",
        default=str(_ST2 / "formating" / "formated_requirements.json"),
        help="Path to formated requirements JSON (default: Step2/formating/formated_requirements.json)",
    )
    parser.add_argument(
        "--chroma-dir",
        default=str(_REPO_ROOT / "chroma_admin_summary"),
        help="Chroma directory (default: chroma_admin_summary at project root, same as embed from repo root)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Nombre de resultats Chroma par exigence (ordre = score). Avec la repartition unique, "
        "augmentez si beaucoup d'exigences partagent le meme meilleur fichier (ex. 15-20).",
    )
    parser.add_argument(
        "--allow-duplicate-files",
        action="store_true",
        help="Desactive la regle « un fichier par exigence » : garde jusqu’a --top-k propositions et "
        "autorise le meme document pour plusieurs exigences.",
    )
    parser.add_argument(
        "--output",
        default=str(_ST2 / "requirements_matches.json"),
        help="Output match report JSON (toggles + matches array)",
    )
    parser.add_argument(
        "--toggles-file",
        type=Path,
        default=None,
        help="JSON object of booleans, e.g. {\"groupement\": false, ...}. See Step2/match_toggles.json",
    )
    add_condition_toggle_args(parser)
    args = parser.parse_args()

    requirements_path = Path(args.requirements)
    chroma_dir = Path(args.chroma_dir)
    output_path = Path(args.output)

    toggles = build_toggles_from_parsed_args(args, args.toggles_file)

    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    requirements_json = json.loads(requirements_path.read_text(encoding="utf-8"))
    requirements = flatten_requirements(requirements_json)
    db = load_chroma_db(chroma_dir)
    report = run_matching(requirements, db, top_k=args.top_k, toggles=toggles)
    report = apply_declaration_bundles(report, requirements)
    if not args.allow_duplicate_files:
        report = apply_one_unused_match_per_requirement(report)

    out_obj = {
        "toggles": toggles,
        "matches": report,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    n_done = sum(1 for r in report if not r.get("skipped"))
    n_skip = sum(1 for r in report if r.get("skipped"))
    print(f"Wrote {output_path.resolve()} (matched: {n_done}, ignored by toggles: {n_skip})")
    print("Toggles:")
    for line in iter_toggles_file_lines(toggles):
        print(line)


if __name__ == "__main__":
    main()
