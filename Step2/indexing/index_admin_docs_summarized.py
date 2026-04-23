"""Build an admin index JSON with LLM summaries (Claude) from extracted text.

This script intentionally does NOT use OCR/Tesseract.

By default, each **PDF page** is summarized separately with the **full embedded text of that page**
(one Claude call per page). The combined `summary` in the index lists ``[Page N]`` blocks. Use
``--one-shot`` to revert to a single call on a truncated multi-page extract.

Compare with:
- `Step2/indexing/index_admin_docs.py` for an index built with embedded text plus Tesseract OCR when
  a PDF page has little or no text (same output shape: path, summary, issue_or_validity_date).

Output format:
[
  {
    "path": "data/admin/foo.pdf",
    "summary": "Quelques phrases en francais...",
    "issue_or_validity_date": "31/12/2025" | null,
    "extracted_text_preview": "Debut du texte utile..."
  }
]

Requirements:
- CLAUDE_API_KEY (env or ``Step2/.env``)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import anthropic
import fitz
from dotenv import load_dotenv

import index_admin_docs as idx  # noqa: E402

load_dotenv(_HERE.parent / ".env")

DEFAULT_MODEL = os.getenv("EASY_TENDER_MODEL", "claude-sonnet-4-6")

# One-shot (merged) mode: if --max-chars is 0, this cap applies (full merge can exceed API limits).
_ONESHOT_DEFAULT_MAX_CHARS = 12_000
_ONESHOT_DEFAULT_MAX_PAGES = 5


def _apply_char_cap(s: str, max_chars: int) -> str:
    if not max_chars or max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars]


def _load_text_file(path: Path, max_chars: int) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    raw = idx.strip_arabic_script(raw)
    return _apply_char_cap(raw, max_chars)


def extract_pdf_text_embedded(
    file_path: Path,
    *,
    max_pages: int,
    max_chars: int,
) -> str:
    try:
        doc = fitz.open(file_path)
    except Exception:
        # Some files can be plain text renamed with .pdf.
        return _load_text_file(file_path, max_chars)
    parts: list[str] = []
    try:
        n = len(doc) if max_pages <= 0 else min(len(doc), max(1, max_pages))
        for i in range(n):
            parts.append((doc[i].get_text() or "").strip())
    finally:
        doc.close()
    merged = "\n\n".join(p for p in parts if p)
    merged = idx.strip_arabic_script(merged)
    return _apply_char_cap(merged, max_chars)


def extract_pdf_page_texts(
    file_path: Path,
    *,
    max_pages: int,
) -> list[str]:
    """
    One string per page (UTF-8 embedded text), script stripped, order preserved.
    max_pages <= 0 means all pages in the file.
    """
    try:
        doc = fitz.open(file_path)
    except Exception:
        return []
    out: list[str] = []
    try:
        n = len(doc) if max_pages <= 0 else min(len(doc), max_pages)
        for i in range(n):
            t = (doc[i].get_text() or "").strip()
            t = idx.strip_arabic_script(t)
            out.append(t)
    finally:
        doc.close()
    return out


def extract_document_text(
    file_path: Path,
    *,
    max_pages: int,
    max_chars: int,
) -> str:
    suf = file_path.suffix.lower()
    if suf == ".pdf":
        return extract_pdf_text_embedded(
            file_path,
            max_pages=max_pages,
            max_chars=max_chars,
        )
    if suf in {".txt", ".md", ".json"}:
        return _load_text_file(file_path, max_chars)
    return ""


def summarize_with_claude(
    text: str,
    *,
    model: str,
    file_name: str = "",
    page_info: str = "",
) -> str:
    key = (os.getenv("CLAUDE_API_KEY") or "").strip()
    if not key:
        raise ValueError("CLAUDE_API_KEY is not set")

    client = anthropic.Anthropic(api_key=key)
    name_hint = (
        f"Nom du fichier (peut aider a identifier le type de piece, ex. CNSS, attestation fiscale): "
        f"{file_name}\n\n"
        if file_name
        else ""
    )
    if page_info:
        scope = (
            f"Ceci est le texte **complet** d'une seule page d'un PDF ({page_info}). "
            "L'extrait contient toute la page, sans troncature.\n"
            "Resume en francais **tout** le contenu utile de cette page (titres, mentions, tableaux, "
            "noms, dates, numeros) : 2 a 8 phrases si la page est riche, moins si elle est mince. "
            "Ne te limite pas a la premiere phrase du texte. Pas de JSON. Pas de markdown.\n\n"
        )
    else:
        scope = (
            "Voici le texte extrait d'un document administratif (souvent marocain). "
            "Ignore toute partie non francaise ou non latine si elle apparait encore.\n"
            "Redige un resume en francais (3 a 5 phrases maximum) qui permet de comprendre "
            "de quoi il s'agit (type de document, objet, beneficiaire ou organisme si visible, "
            "montants ou delais importants).\n"
            "Si le nom du fichier ci-dessus est informatif, tu peux l'integrer une fois dans le resume "
            "(sans chemin complet). Pas de JSON. Pas de markdown.\n\n"
        )
    prompt = f"{scope}{name_hint}Texte:\n{text}"
    msg = client.messages.create(
        model=model,
        max_tokens=1024 if page_info else 512,
        system="Tu es un assistant specialise en documents administratifs. Sois factuel.",
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            return (block.text or "").strip()
    return ""


def _merged_for_date(pages: list[str], cap: int = 100_000) -> str:
    s = "\n\n".join(p for p in pages if p and p.strip())
    return s[:cap] if len(s) > cap else s


def _summarize_per_pdf_pages(
    path: Path,
    pages: list[str],
    *,
    model: str,
    max_chars_per_page: int,
) -> str:
    """One Claude call per non-empty page; each call receives the full page text (optionally capped)."""
    nonempty: list[tuple[int, str]] = []
    for i, t in enumerate(pages, start=1):
        if not t.strip():
            continue
        chunk = _apply_char_cap(t, max_chars_per_page) if max_chars_per_page and max_chars_per_page > 0 else t
        nonempty.append((i, chunk))
    if not nonempty:
        return (
            "Aucun texte extractible (pages vides ou PDF scanne sans couche texte). "
            "Utilisez Step2/indexing/index_admin_docs.py (OCR) pour indexer ces fichiers."
        )
    n_pages = len(pages)
    parts: list[str] = []
    for i, chunk in nonempty:
        s = summarize_with_claude(
            chunk,
            model=model,
            file_name=path.name,
            page_info=f"page {i} sur {n_pages}",
        )
        parts.append(f"[Page {i}]\n{s}")
    return "\n\n".join(parts)


def build_index(
    admin_dir: Path,
    *,
    model: str,
    max_pages: int,
    max_chars: int,
    preview_chars: int,
    one_shot: bool,
) -> list[dict]:
    out: list[dict] = []
    for path in sorted(admin_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in idx.SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf" and not one_shot:
            # Full page: one LLM call per page, full embedded text of that page
            max_p = max_pages if max_pages > 0 else 0
            page_texts = extract_pdf_page_texts(path, max_pages=max_p)
            merged = _merged_for_date(page_texts)
            date = idx.extract_issue_or_validity_date(merged)
            per_cap = max_chars if max_chars > 0 else 0
            if not any(t.strip() for t in page_texts):
                summary = (
                    "Aucun texte extractible (PDF scanne sans couche texte). "
                    "Utilisez Step2/indexing/index_admin_docs.py (OCR) pour indexer ces fichiers."
                )
            else:
                summary = _summarize_per_pdf_pages(
                    path, page_texts, model=model, max_chars_per_page=per_cap
                )
            text_preview_source = merged
        else:
            # Merged: ``--one-shot`` on PDF, or any non-PDF: single LLM call
            if path.suffix.lower() == ".pdf":
                eff_pages = max_pages if max_pages > 0 else _ONESHOT_DEFAULT_MAX_PAGES
                eff_chars = max_chars if max_chars > 0 else _ONESHOT_DEFAULT_MAX_CHARS
            else:
                eff_pages = 0
                eff_chars = max_chars if max_chars > 0 else 0
            text = extract_document_text(
                path,
                max_pages=eff_pages,
                max_chars=eff_chars,
            )
            date = idx.extract_issue_or_validity_date(text)
            if not text.strip():
                summary = (
                    "Aucun texte extractible (PDF scanne sans couche texte). "
                    "Utilisez Step2/indexing/index_admin_docs.py (OCR) pour indexer ces fichiers."
                )
            else:
                summary = summarize_with_claude(text, model=model, file_name=path.name)
            text_preview_source = text

        preview = text_preview_source[:preview_chars] if preview_chars > 0 else ""

        out.append(
            {
                "path": str(path.as_posix()),
                "summary": summary,
                "issue_or_validity_date": date,
                "extracted_text_preview": preview,
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Index admin docs with Claude summaries (no OCR). "
        "Default: per PDF page, full page text, one LLM call per page (synthesis couvre toute la page). "
        "Use --one-shot for a single call on a truncated merge (legacy, fewer API calls).",
    )
    p.add_argument("--admin-dir", default="data/admin")
    p.add_argument("--output", default="data/admin/admin_docs_index_summarized.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--one-shot",
        action="store_true",
        help=f"Un seul appel par PDF: fusion des pages tronquée (defaut: --max-pages {_ONESHOT_DEFAULT_MAX_PAGES}, "
        f"--max-chars {_ONESHOT_DEFAULT_MAX_CHARS} si 0). Sans ce flag, resume page par page (texte integral par page).",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="PDF: 0 = toutes les pages. Per-page: nombre max de pages a resumer. "
        f"One-shot: idem, ou defaut {_ONESHOT_DEFAULT_MAX_PAGES} si 0.",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Per-page: troncature par page (0 = texte de la page en entier). "
        f"One-shot: plafond sur le texte fusionne (0 = defaut {_ONESHOT_DEFAULT_MAX_CHARS}).",
    )
    p.add_argument("--preview-chars", type=int, default=600, help="Chars stored in JSON preview")
    args = p.parse_args()

    admin_dir = Path(args.admin_dir)
    if not admin_dir.exists():
        raise FileNotFoundError(admin_dir)

    rows = build_index(
        admin_dir,
        model=args.model,
        max_pages=args.max_pages,
        max_chars=args.max_chars,
        preview_chars=args.preview_chars,
        one_shot=bool(args.one_shot),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} entries -> {out_path}")


if __name__ == "__main__":
    main()
