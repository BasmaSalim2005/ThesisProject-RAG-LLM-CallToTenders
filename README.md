# Tender RAG assistant

This repository helps **prepare a bid** for a public tender: it **normalizes requirements** from Step 1 extraction, **indexes** administrative documents (and optional CVs) in **Chroma**, **retrieves** the best-matching files per requirement, and produces a **human-readable PDF report** (optionally followed by a **project context** appendix from the specifications export).

## What is where

| Path | Role |
|------|------|
| `step1/` | Extraction and intermediate JSON (e.g. `v2/specifications_extracted.json`, per-lot `*_extracted.json`) |
| `Step2/matching/` | RAG: requirements flattening, Chroma search, condition toggles, post-processing (bundles, one-file rule) |
| `Step2/reporting/` | PDF report from `requirements_matches.json` + labels |
| `Step2/embedding/` | Build / refresh the Chroma vector store from the admin index and optional `data/hr/cv.json` |
| `Step2/indexing/` | Build **JSON indexes** of PDFs (text + LLM summary or OCR variant) |
| `Step2/formating/` | Export **normalized** requirements (`formated_requirements.json`) for matching |
| `data/` | Source PDFs, admin index, CV catalogue |
| `easy_tender/` | Small app / shared **format** logic for requirements |

## Environment (`.env` per folder)

Scripts load **the `.env` next to the code they belong to** (not a single root file):

| File | Purpose (typical keys) |
|------|------------------------|
| `step1/.env` | e.g. `GROQ_API_KEY`, `CLAUDE_API_KEY` for `extract.py` and `lngextct.py` |
| `Step2/.env` | e.g. `CLAUDE_API_KEY`, `EASY_TENDER_MODEL`, `TESSERACT_CMD` for `Step2/indexing/…` |
| `easy_tender/.env` | e.g. `OPENROUTER_API_KEY`, `CLAUDE_API_KEY` for the web app and `format_requirements` |

`.env` is gitignored. Copy the keys you use into each file as needed.

## End-to-end (from the repo root)

Prerequisites: Python 3, dependencies for `Step2` (e.g. `langchain_chroma`, `fastembed`, `fpdf2`, and others as used by the scripts).

1. **Normalize requirements** (from an extraction JSON):

   `python Step2/formating/export_formated_requirements.py path/to/extraction.json`

2. **Index PDFs** (produces a summarized admin index, example):

   `python Step2/indexing/index_admin_docs_summarized.py`

3. **Embed** into Chroma (default store at repo root: `chroma_admin_summary` if you use defaults from the repo root):

   `python -m Step2.embedding.embed_admin_chroma --force`

4. **Match** requirements to files:

   `python -m Step2.matching.rag_match_requirements`

5. **PDF report** (match section + optional **project** appendix):

   `python -m Step2.reporting.match_report_pdf`

   Shims still work, e.g. `python Step2/match_report_pdf.py`.

## PDF report: matching + project context

- The first part of the report is the **RAG layout**: suggested documents per requirement (admin + CVs), with toggles, validity window, and notes.
- If `step1/v2/specifications_extracted.json` exists, a final section **“Contexte du marche”** is added with a **synthesis** of:
  - `PERIMETRE_DU_PROJET`
  - `DEROULEMENT_DU_PROJET`
  - `CONSISTANCE_DES_PRESTATIONS`  
  (and any other top-level string fields in the same file.)

- To **disable** that appendix: `--no-project-spec`
- To use **another** JSON: `--project-spec path/to/specifications.json`

## Configuration snippets

- **Condition checkboxes (groupement, hors Maroc, etc.):** `Step2/match_toggles.json` and CLI flags; see `Step2/matching/condition_toggles.py`.
- **Chroma path:** use `--chroma-dir` (matching and embedding) consistently from the same working directory, or use absolute paths.
- **CV matching:** `data/hr/cv.json` is embedded with the admin index when the file is present; CV requirements in the formated JSON are detected via `requirement_type: cv` or the “CV” wording in the technical lot.

## Development note

The `Step2` tree is a Python package. Prefer running from the **repository root** so `import Step2` resolves; entry scripts at `Step2/rag_match_requirements.py`, `Step2/match_report_pdf.py`, and `Step2/embed_admin_chroma.py` are thin shims for the same behavior.

---

*Internal / project tool — not a public API guarantee.*
