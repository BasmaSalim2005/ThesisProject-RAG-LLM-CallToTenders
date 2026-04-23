"""Build an index JSON for documents in data/admin.

Output format (one object per file):
[
  {
    "path": "data/admin/example.pdf",
    "summary": "...",
    "issue_or_validity_date": "31/12/2025" | null
  }
]

Dates are inferred from the first page (PDF) or start of text files using
French/common patterns (valable, validité, délivré, etc.). If none found, null.
"""

from __future__ import annotations

import argparse
import json
import io
import os
import re
from pathlib import Path

from dotenv import load_dotenv
import fitz

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

try:
    import pytesseract
    from PIL import Image
except ImportError:  # optional OCR stack
    pytesseract = None
    Image = None

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json"}

OCR_LANG_DEFAULT = os.getenv("TESSERACT_LANG", "fra+eng")


def _windows_tesseract_exe_candidates() -> list[Path]:
    """Typical install locations when tesseract.exe is not on PATH."""
    if os.name != "nt":
        return []
    seen: set[str] = set()
    out: list[Path] = []
    for key in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(key)
        if not root:
            continue
        p = Path(root) / "Tesseract-OCR" / "tesseract.exe"
        key_norm = str(p.resolve()).lower()
        if p.is_file() and key_norm not in seen:
            seen.add(key_norm)
            out.append(p)
    return out


def configure_tesseract() -> str | None:
    """
    Configure pytesseract executable path from env if needed.
    Returns an error message if OCR deps exist but Tesseract is not callable.
    """
    if pytesseract is None:
        return "pytesseract is not installed (pip install pytesseract)."

    attempts: list[str] = []
    env_cmd = (os.getenv("TESSERACT_CMD") or "").strip()
    if env_cmd:
        attempts.append(env_cmd)
    attempts.extend(str(p) for p in _windows_tesseract_exe_candidates())
    attempts.append("tesseract")

    last_err: Exception | None = None
    for cmd in attempts:
        pytesseract.pytesseract.tesseract_cmd = cmd
        try:
            _ = pytesseract.get_tesseract_version()
            return None
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    hint = (
        "Tesseract OCR is not installed or tesseract.exe is not reachable. "
        "On Windows, PATH must include the folder that contains tesseract.exe "
        "(usually C:\\\\Program Files\\\\Tesseract-OCR\\\\), not the tessdata subfolder. "
        "The tessdata folder only holds language .traineddata files; if Tesseract cannot "
        "find them, set TESSDATA_PREFIX to that folder (e.g. ...\\\\Tesseract-OCR\\\\tessdata). "
        "You can also set TESSERACT_CMD to the full path of tesseract.exe."
    )
    return f"{hint} Details: {last_err}"


def ocr_pdf_page(doc: "fitz.Document", page_index: int, *, zoom: float = 2.0) -> str:
    """
    OCR a single PDF page using Tesseract.
    Uses French+English by default (TESSERACT_LANG), strips Arabic afterward.
    """
    if pytesseract is None or Image is None:
        return ""
    if page_index < 0 or page_index >= len(doc):
        return ""

    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))

    # Try multiple languages + page segmentation modes. Windows installs sometimes ship
    # only English packs unless you add French training data.
    lang_candidates: list[str] = []
    primary = (OCR_LANG_DEFAULT or "fra+eng").strip()
    if primary:
        lang_candidates.append(primary)
    for extra in ("fra", "eng"):
        if extra not in lang_candidates:
            lang_candidates.append(extra)

    last_err: Exception | None = None
    for lang in lang_candidates:
        for psm in ("6", "4", "1"):
            try:
                text = pytesseract.image_to_string(
                    image,
                    lang=lang,
                    config=f"--oem 3 --psm {psm}",
                )
                cleaned = strip_arabic_script(text)
                if cleaned.strip():
                    return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue

    return ""


# Arabic script blocks (letters only — keep digits/punctuation from other ranges).
_ARABIC_LETTERS_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+")


def strip_arabic_script(text: str) -> str:
    """Remove Arabic-script words; keep Latin (French), digits, punctuation."""
    if not text:
        return ""
    text = _ARABIC_LETTERS_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _latin_letter_ratio(text: str) -> float:
    letters = sum(1 for c in text if ("A" <= c <= "Z") or ("a" <= c <= "z") or c in "àâäéèêëïîôùûüçœæÀÂÄÉÈÊËÏÎÔÙÛÜÇŒÆ")
    denom = max(1, len(text.replace(" ", "")))
    return letters / denom


def _needs_ocr(pdf_text: str) -> bool:
    t = (pdf_text or "").strip()
    if len(t) < 40:
        return True
    if _latin_letter_ratio(t) < 0.12:
        return True
    return False


def ocr_first_page_pdf(file_path: Path, *, zoom: float = 2.0) -> str:
    """
    OCR first page using Tesseract (French). Requires:
    - pip: pytesseract, pillow
    - system: Tesseract-OCR installed and on PATH
    """
    if pytesseract is None or Image is None:
        return ""
    doc = fitz.open(file_path)
    try:
        if len(doc) == 0:
            return ""
        return ocr_pdf_page(doc, 0, zoom=zoom)
    finally:
        doc.close()


def extract_first_page_text_pdf(file_path: Path, *, use_ocr: bool, ocr_zoom: float) -> str:
    """
    Prefer embedded PDF text. For scans, OCR the first pages until we get usable text.
    Falls back to plain-text read if a .pdf is not a valid PDF binary.
    """
    try:
        doc = fitz.open(file_path)
    except Exception:
        raw = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        return raw[:3500]
    try:
        if len(doc) == 0:
            return ""

        best_embedded = ""
        scan_pages = min(len(doc), 3)

        for i in range(scan_pages):
            embedded = (doc[i].get_text() or "").strip()
            if len(strip_arabic_script(embedded)) > len(strip_arabic_script(best_embedded)):
                best_embedded = embedded

        if not use_ocr:
            return best_embedded

        for i in range(scan_pages):
            embedded = (doc[i].get_text() or "").strip()
            if not _needs_ocr(embedded):
                continue
            ocr_text = ocr_pdf_page(doc, i, zoom=ocr_zoom).strip()
            if len(strip_arabic_script(ocr_text)) > len(strip_arabic_script(embedded)):
                return ocr_text

        return best_embedded
    finally:
        doc.close()

# Keywords suggesting issue or validity / expiration nearby (French / admin docs).
_DATE_CONTEXT = re.compile(
    r"(valable|validit[ée]|jusqu['']?\s*au|date\s+d['']?expiration|expiration|"
    r"expire|établi|etabli|délivr[ée]|delivr[ée]|date\s+d['']?émission|date\s+d['']?emission|"
    r"émis|emis|fait\s+à|delivre\s+le|délivré\s+le|"
    r"attestation.+valable|moins\s+d['']?un\s+an)",
    re.IGNORECASE,
)

# Numeric dates: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
_DATE_DMY = re.compile(
    r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b",
)
# ISO-style YYYY-MM-DD
_DATE_ISO = re.compile(r"\b(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})\b")

# e.g. "15 janvier 2025"
_MONTHS_FR = (
    "janvier|février|fevrier|mars|avril|mai|juin|"
    "juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
)
_DATE_FR = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTHS_FR})\s+(\d{{4}})\b",
    re.IGNORECASE,
)


def read_first_page_text(
    file_path: Path,
    *,
    use_ocr: bool,
    ocr_zoom: float,
) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        raw = extract_first_page_text_pdf(file_path, use_ocr=use_ocr, ocr_zoom=ocr_zoom)
        return strip_arabic_script(raw)

    if suffix in {".txt", ".md", ".json"}:
        raw = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        # First "page" ~ 3500 chars for long text exports
        return strip_arabic_script(raw[:3500])

    return ""


def simple_summary(text: str, max_chars: int = 360) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "No readable French/Latin text on first page (try OCR: install Tesseract + pytesseract)."
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _first_date_in_window(window: str) -> str | None:
    for pattern in (_DATE_DMY, _DATE_ISO, _DATE_FR):
        m = pattern.search(window)
        if m:
            return m.group(0).strip()
    return None


def extract_issue_or_validity_date(text: str) -> str | None:
    if not text.strip():
        return None

    best: str | None = None
    search_text = text[:4000]

    for ctx in _DATE_CONTEXT.finditer(search_text):
        start = max(0, ctx.start() - 40)
        end = min(len(search_text), ctx.end() + 140)
        window = search_text[start:end]
        found = _first_date_in_window(window)
        if found:
            best = found
            break

    if not best:
        head = search_text[:1500]
        found = _first_date_in_window(head)
        if found:
            best = f"{found} (date seule sur la premiere partie)"

    return best


def build_index(admin_dir: Path, *, use_ocr: bool, ocr_zoom: float) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(admin_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = read_first_page_text(path, use_ocr=use_ocr, ocr_zoom=ocr_zoom)
        docs.append(
            {
                "path": str(path.as_posix()),
                "summary": simple_summary(text),
                "issue_or_validity_date": extract_issue_or_validity_date(text),
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Index data/admin documents.")
    parser.add_argument("--admin-dir", default="data/admin", help="Admin documents folder")
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback for scanned PDFs (embedded text only).",
    )
    parser.add_argument(
        "--ocr-zoom",
        type=float,
        default=2.0,
        help="Render scale for OCR (higher = sharper, slower). Default 2.0.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default="",
        help="Optional full path to tesseract.exe (or set TESSERACT_CMD in .env).",
    )
    parser.add_argument(
        "--output",
        default="data/admin/admin_docs_index.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    admin_dir = Path(args.admin_dir)
    if not admin_dir.exists():
        raise FileNotFoundError(f"Admin folder not found: {admin_dir}")

    use_ocr = not args.no_ocr
    if args.tesseract_cmd.strip():
        os.environ["TESSERACT_CMD"] = args.tesseract_cmd.strip()
    if use_ocr:
        ocr_err = configure_tesseract()
        if ocr_err:
            print(f"WARNING: OCR disabled/failed to initialize: {ocr_err}")
            use_ocr = False

    index_data = build_index(admin_dir, use_ocr=use_ocr, ocr_zoom=args.ocr_zoom)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(index_data)} files -> {output}")


if __name__ == "__main__":
    main()
