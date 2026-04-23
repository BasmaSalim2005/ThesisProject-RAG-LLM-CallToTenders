"""Claude API-based tender extraction — shared by Easy Tender UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic
import fitz
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

SYSTEM_PROMPT = (
    "You are a precise document analysis assistant for Moroccan tender calls. "
    "Respond with valid JSON only."
)

DEFAULT_MODEL = os.getenv(
    "EASY_TENDER_MODEL",
    "claude-sonnet-4-6",
)


def extract_text_from_pdf_bytes(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text())
        return "\n".join(parts)
    finally:
        doc.close()


def build_user_prompts(tender_call_text: str) -> dict[str, str]:
    return {
        "requirements": (
            "From the tender document below, extract ONLY the submission document "
            "requirements include special cases. Return JSON: "
            '{ "dossier_administratif": [], "offre_technique": [], "offre_financiere": {} } '
            "\n\nDocument: "
        )
        + tender_call_text,
    }


def _extract_text_block(message: anthropic.types.Message) -> str:
    for block in message.content:
        if getattr(block, "type", "") == "text":
            return block.text or ""
    return ""


def _parse_json_response(raw_text: str) -> dict[str, Any] | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def call_claude(
    model_id: str,
    user_prompt: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    key = (api_key or os.getenv("CLAUDE_API_KEY") or "").strip()
    if not key:
        raise ValueError("CLAUDE_API_KEY is not set")

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return _parse_json_response(_extract_text_block(response))


def run_extraction(
    tender_call_text: str,
    model_id: str | None = None,
    *,
    api_key: str | None = None,
) -> tuple[dict[str, Any | None], list[str]]:
    """
    Run requirements extraction. Returns (results_by_category, errors).
    """
    model = model_id or DEFAULT_MODEL
    prompts = build_user_prompts(tender_call_text.strip())
    results: dict[str, Any | None] = {}
    errors: list[str] = []

    for category, user_prompt in prompts.items():
        try:
            results[category] = call_claude(model, user_prompt, api_key=api_key)
        except Exception as e:  # noqa: BLE001 — surface to UI
            results[category] = None
            errors.append(f"{category}: {e}")

    return results, errors
