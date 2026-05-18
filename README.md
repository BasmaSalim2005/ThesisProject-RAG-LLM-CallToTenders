# Tender RAG Assistant

This project helps prepare a response to a Moroccan public tender. It extracts tender requirements, formats them into a clean checklist, indexes company administrative documents and CV profiles, matches each requirement with the best available file or person, then generates a readable PDF report.

The matching step uses Chroma/RAG for document search. CV/profile requirements are handled separately so requested roles, study level, and experience can be compared against the available CV dataset.

## Main Files And Folders

| Path | Purpose |
| --- | --- |
| `easy_tender/` | Web app and shared extraction/formatting logic. `app.py` runs the UI, `extract_service.py` extracts tender text, and `format_requirements.py` normalizes requirements. |
| `extraction/` | Experimental/CLI extraction scripts and extracted JSON examples. |
| `Step2/formating/` | CLI formatter that writes `formated_requirements.json` from extracted requirements. |
| `Step2/indexing/` | Builds JSON indexes from admin PDFs before embedding. |
| `Step2/embedding/` | Builds the Chroma vector store from admin document indexes and the CV catalogue. |
| `Step2/matching/` | Requirement matching logic, CV ranking, condition toggles, and post-processing. |
| `Step2/reporting/` | Generates the final PDF report from matches and formatted requirements. |
| `data/admin/` | Admin document index and source admin files. |
| `data/hr/cv_docs_index.json` | CV/profile dataset used for HR matching. |
| `example1/`, `example2/`, `example3/` | Example inputs, formatted requirements, matches, and reports. |

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Add API keys where needed. The app and scripts load `.env` files near the code:

```text
easy_tender/.env
Step2/.env
extraction/.env
```

Typical keys are:

```text
CLAUDE_API_KEY=...
OPENROUTER_API_KEY=...
```

## Run The Web App

Use this if you want to upload a tender PDF/TXT and get extracted/formatted requirements through the UI:

```bash
python -m easy_tender.app
```

Then open:

```text
http://127.0.0.1:5050
```

## Run The Pipeline Manually

### 1. Format Extracted Requirements

Use an extracted requirements JSON, for example one of the example files:

```bash
python Step2/formating/export_formated_requirements.py example2/requirements_extracted.json --output Step2/formating/formated_requirements.json
```

This creates the normalized requirements file used by matching.

### 2. Build Or Refresh The Admin Index

If the admin PDF index already exists at `data/admin/admin_docs_index_summarized.json`, you can skip this step.

```bash
python Step2/indexing/index_admin_docs_summarized.py
```

### 3. Build The Chroma Store

This embeds admin documents and CV profiles:

```bash
python -m Step2.embedding.embed_admin_chroma --force
```

By default it uses:

```text
data/admin/admin_docs_index_summarized.json
data/hr/cv_docs_index.json
```

### 4. Match Requirements

```bash
python -m Step2.matching.rag_match_requirements --requirements Step2/formating/formated_requirements.json --output Step2/requirements_matches.json
```

Optional situation toggles can be passed with CLI flags or a toggles file:

```bash
python -m Step2.matching.rag_match_requirements --toggles-file Step2/match_toggles.json
```

### 5. Generate The PDF Report

```bash
python -m Step2.reporting.match_report_pdf --matches Step2/requirements_matches.json --requirements Step2/formating/formated_requirements.json --output Step2/requirements_matches_report.pdf
```

To remove the project-context appendix:

```bash
python -m Step2.reporting.match_report_pdf --no-project-spec
```

