"""Easy Tender — web UI for tender document extraction."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, jsonify, render_template, request

from easy_tender.extract_service import (
    DEFAULT_MODEL,
    extract_text_from_pdf_bytes,
    run_extraction,
)
from easy_tender.format_requirements import format_requirements_payload

ROOT = Path(__file__).resolve().parent

app = Flask(
    "easy_tender",
    root_path=str(ROOT),
    template_folder="templates",
    static_folder="static",
)


def _format_results_for_ui(results):
    formatted = dict(results)
    formatted["requirements"] = format_requirements_payload(results.get("requirements"))
    return formatted


def _build_checklist(formatted_results):
    reqs = formatted_results.get("requirements", {})
    checklist = {
        "dossier_administratif": [],
        "offre_technique": [],
        "offre_financiere": [],
    }
    for category in checklist:
        entries = reqs.get(category, [])
        for item in entries:
            description = str(item.get("description", "")).strip()
            legacy = str(item.get("doc_name", "")).strip()
            criteria = str(item.get("criteria", "")).strip()
            line = description or legacy
            if criteria and criteria not in line:
                line = f"{line} — {criteria}".strip(" —") if line else criteria
            checklist[category].append(line or "Exigence sans description.")
    return checklist


@app.route("/")
def index():
    return render_template("index.html", default_model=DEFAULT_MODEL)


@app.post("/api/extract")
def api_extract():
    model = (request.form.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify(
            {
                "ok": False,
                "error": "Veuillez televerser un document d'appel d'offres (PDF ou TXT).",
            }
        ), 400

    f = request.files["file"]
    raw = f.read()
    if not raw:
        return jsonify({"ok": False, "error": "Le fichier est vide."}), 400

    name = (f.filename or "").lower()
    if name.endswith(".pdf"):
        try:
            text = extract_text_from_pdf_bytes(raw)
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"Impossible de lire le PDF: {e}"}), 400
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")

    if not text.strip():
        return jsonify({"ok": False, "error": "Le document ne contient pas de texte exploitable."}), 400

    results, errors = run_extraction(text, model_id=model)
    formatted_results = _format_results_for_ui(results)
    checklist = _build_checklist(formatted_results)
    return jsonify(
        {
            "ok": True,
            "model": model,
            "results": formatted_results,
            "checklist": checklist,
            "errors": errors,
        }
    )


def main():
    app.run(host="127.0.0.1", port=5050, debug=True)


if __name__ == "__main__":
    main()
