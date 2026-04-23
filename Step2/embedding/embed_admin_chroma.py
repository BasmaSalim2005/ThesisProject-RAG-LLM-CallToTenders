"""Build / refresh Chroma embeddings from admin index + optional CV catalogue.

Run after the indexers under `Step2/indexing/`. By default embeds:
  - `data/admin/admin_docs_index_summarized.json` (or override with --admin-index)
  - `data/hr/cv.json` when that file exists (skip with --no-cv)

Then match: `Step2/formating/formated_requirements.json` and ``python -m Step2.matching.rag_match_requirements``.

Examples:
  python Step2/indexing/index_admin_docs_summarized.py
  python -m Step2.embedding.embed_admin_chroma --force
  python -m Step2.embedding.embed_admin_chroma --admin-index data/admin/admin_docs_index.json --cv-json data/hr/cv.json --chroma-dir chroma_admin_cv --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Step2.matching.rag_chroma_utils import build_chroma_from_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed admin index (and optional CV JSON) into one Chroma store."
    )
    parser.add_argument(
        "--admin-index",
        default="data/admin/admin_docs_index_summarized.json",
        help="Admin docs index JSON (array of objects from index_admin_docs*.py)",
    )
    parser.add_argument(
        "--cv-json",
        default="data/hr/cv.json",
        help="CV catalogue JSON with cv_database (ignored if missing unless you pass --require-cv)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Embed only the admin index, do not merge CVs",
    )
    parser.add_argument(
        "--require-cv",
        action="store_true",
        help="Fail if --cv-json is missing (default: warn and embed admin only)",
    )
    parser.add_argument(
        "--chroma-dir",
        default="chroma_admin_summary",
        help="Directory for persisted Chroma data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing chroma-dir and rebuild",
    )
    args = parser.parse_args()

    admin_path = Path(args.admin_index)
    chroma_dir = Path(args.chroma_dir)
    if not admin_path.exists():
        raise FileNotFoundError(f"Admin index not found: {admin_path}")

    paths: list[Path] = [admin_path]
    if not args.no_cv:
        cv_path = Path(args.cv_json)
        if cv_path.is_file():
            paths.append(cv_path)
        elif args.require_cv:
            raise FileNotFoundError(f"CV file not found (required): {cv_path}")
        else:
            print(f"Note: CV file not found, embedding admin only: {cv_path}")

    n = build_chroma_from_sources(paths, chroma_dir, force=args.force)
    labels = ", ".join(p.as_posix() for p in paths)
    print(f"Embedded {n} records from [{labels}] -> {chroma_dir.resolve()}")


if __name__ == "__main__":
    main()
