"""Normalize tender requirements for the Easy Tender UI and for Step2 RAG."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

_VALIDITY_RE = re.compile(
    r"(moins\s+d['']?un\s+an|<\s*1\s*an|valable|validit[ée]|jusqu['']?au|expiration|expire|déliv[rR]ée|delivr[rR]ée)",
    re.IGNORECASE,
)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_OPENROUTER_FORMAT_MODEL = "mistralai/mistral-7b-instruct:free"


def _detect_requirement_type(category: str, *parts: str) -> str:
    """
    Disambiguate items inside "offre_technique" (CV requirements vs other documents).
    """
    if category != "offre_technique":
        return ""
    hay = " ".join(p for p in parts if p).strip()
    if re.search(r"\bCVs?\b", hay, re.IGNORECASE):
        return "cv"
    return ""


def _split_on_first_colon(item: str) -> tuple[str, str]:
    if ":" in item:
        left, right = item.split(":", 1)
        return left.strip(), right.strip()
    return item.strip(), ""


def _extract_title_and_extra(raw_text: str) -> tuple[str, str]:
    """
    Split free-text into a short title-ish prefix and the rest (constraints / details).
    Output is used only to build a single "description" string for matching.
    """
    raw_text = raw_text.strip()
    if ":" in raw_text:
        return _split_on_first_colon(raw_text)

    if "(" in raw_text:
        cut = raw_text.find("(")
        title = raw_text[:cut].strip()
        extra = raw_text[cut:].strip()
        return (title or raw_text), extra

    m = _VALIDITY_RE.search(raw_text)
    if m:
        title = raw_text[: m.start()].strip().rstrip("-:;,. ")
        return (title or raw_text), raw_text

    return raw_text, ""


def _slugify(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:64] or "requirement"


# Stable id for the row that encodes the global publication / freshness window (RAG + PDF).
REGLE_FRAICHEUR_PIECES_ID = "regle_fraicheur_pieces"


def _strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _choose_text_for_id(desc: str, criteria: str) -> str:
    """When the model repeats the long « Pour le concurrent… » preambule, name from criteria."""
    d = (desc or "").strip()
    c = (criteria or "").strip()
    if c and re.match(r"^pour\s+le\s+concurrent", d, re.IGNORECASE):
        return c
    if c and re.match(r"^pour\s+concurrents", d, re.IGNORECASE):
        return c
    if c and 12 < len(c) < len(d) and len(c) < 0.92 * max(len(d), 1):
        return c
    return d


def _is_validity_window_row(desc: str) -> bool:
    """The « documents valables 1 an / date de publication » policy line (dossier common case)."""
    d = (desc or "").lower()
    if "valid" in d and (
        "1 ans" in d
        or "un an" in d
        or " 1an" in d.replace(" ", "")
    ) and ("publication" in d or "annonce" in d or "appel" in d or "offre" in d):
        return True
    if d.startswith("les document") and "valide" in d and "1" in d:
        return True
    return False


def _file_like_id_from_description(
    desc: str,
    *,
    category: str,
    line_index: int,
    used: set[str],
) -> str:
    """
    Short, readable id similar to a document filename (no long hash-like slugs).
    Uses a few title words in snake_case, ASCII, unique within a category.
    """
    if _is_validity_window_row(desc):
        base = REGLE_FRAICHEUR_PIECES_ID
        c = base
        n = 0
        while c in used:
            n += 1
            c = f"{base}_{n}"
        used.add(c)
        return c

    raw = (desc or "").split("\n", 1)[0].strip()
    if "(" in raw:
        raw = raw[: raw.find("(")].strip()
    if "." in raw and len(raw) > 12:
        first_sentence = raw.split(".", 1)[0].strip()
        if len(first_sentence) >= 8:
            raw = first_sentence
    raw = _strip_accents(raw)
    words = re.findall(r"[A-Za-z0-9]+", raw, flags=re.IGNORECASE)
    # Skip typical tender boilerplate at the start so the id reads like a document name.
    stop = {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "d",
        "et",
        "ou",
        "pour",
        "par",
        "au",
        "aux",
        "en",
        "sur",
        "il",
        "est",
        "a",
        "lequel",
        "laquelle",
        "lesquels",
        "auquel",
        "auxquels",
        "auxquelles",
        "cette",
        "ces",
        "ce",
        "cette",
        "concurrent",
        "concurrents",
        "candidat",
        "candidature",
        "marche",
        "envisage",
        "attribue",
        "attribuer",
    }
    kept: list[str] = []
    # Drop leading stopwords and administrative prefixes (first up to 14 words scanned).
    skip_budget = 14
    for w in words:
        low = w.lower()
        if skip_budget > 0 and low in stop:
            skip_budget -= 1
            continue
        if low in stop and len(kept) < 1:
            continue
        kept.append(low)
        if len(kept) >= 9:
            break
    if not kept:
        base = f"exigence_{category.split('_', 1)[-1] if category else 'lot'}_{line_index}"
    else:
        base = "_".join(kept)
    base = re.sub(r"_+", "_", base).strip("_")[:72]
    if len(base) < 4:
        base = f"piece_{line_index}"
    n = 0
    candidate = base
    while candidate in used:
        n += 1
        candidate = f"{base}_{n}"
    used.add(candidate)
    return candidate


def _build_description(title: str, extra: str) -> str:
    title = title.strip()
    extra = extra.strip()
    if title and extra and extra != title:
        return f"{title}. {extra}".strip()
    return title or extra


def _normalize_list(category: str, values: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for idx, value in enumerate(values, start=1):
        if isinstance(value, dict):
            desc = str(
                value.get("description", "")
                or value.get("summary", "")
                or value.get("matching_text", "")
            ).strip()
            legacy_title = str(value.get("doc_name", "")).strip()
            criteria = str(value.get("criteria", "")).strip()
            if not desc and legacy_title:
                desc = _build_description(legacy_title, criteria)
            elif desc and criteria and criteria not in desc:
                desc = _build_description(desc, criteria)
            elif not desc:
                desc = criteria or f"Requirement {idx}"

            name_source = _choose_text_for_id(desc, criteria)
            req_id = _file_like_id_from_description(
                name_source, category=category, line_index=idx, used=used_ids
            )
            requirement_type = str(value.get("requirement_type", "")).strip().lower()
            if not requirement_type:
                requirement_type = _detect_requirement_type(category, desc, criteria)

            normalized.append(
                {
                    "id": req_id,
                    "description": desc,
                    "criteria": criteria,
                    "requirement_type": requirement_type,
                }
            )
            continue

        raw_text = str(value).strip()
        title, extra = _extract_title_and_extra(raw_text)
        desc = _build_description(title, extra)
        criteria = extra if extra and extra != title else ""

        requirement_type = _detect_requirement_type(category, raw_text, desc, criteria)

        normalized.append(
            {
                "id": _file_like_id_from_description(
                    desc or raw_text, category=category, line_index=idx, used=used_ids
                ),
                "description": desc or raw_text,
                "criteria": criteria,
                "requirement_type": requirement_type,
            }
        )
    return normalized


def _format_requirements_deterministic(
    raw_requirements: dict[str, Any] | None,
) -> dict[str, list[dict[str, str]]]:
    if not isinstance(raw_requirements, dict):
        return {
            "dossier_administratif": [],
            "offre_technique": [],
            "offre_financiere": [],
        }

    formatted: dict[str, list[dict[str, str]]] = {}
    for category in ("dossier_administratif", "offre_technique", "offre_financiere"):
        value = raw_requirements.get(category, [])
        if isinstance(value, dict):
            value = [f"{k}: {v}" for k, v in value.items()]
        if not isinstance(value, list):
            value = [str(value)]
        formatted[category] = _normalize_list(category, value)

    return formatted


def _parse_json_from_llm(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip("\n").strip()
    return json.loads(text)


def _merge_openrouter_refine(
    base: dict[str, list[dict[str, str]]],
    refined: Any,
) -> dict[str, list[dict[str, str]]]:
    if not isinstance(refined, dict):
        return base
    out: dict[str, list[dict[str, str]]] = {}
    for cat, base_items in base.items():
        ref_items = refined.get(cat)
        if not isinstance(ref_items, list) or len(ref_items) != len(base_items):
            out[cat] = base_items
            continue
        merged: list[dict[str, str]] = []
        for bi, ri in zip(base_items, ref_items):
            item = dict(bi)
            if isinstance(ri, dict):
                d = str(ri.get("description", "")).strip()
                if d:
                    item["description"] = d
                if "criteria" in ri:
                    item["criteria"] = str(ri.get("criteria", "")).strip()
                if "requirement_type" in ri:
                    item["requirement_type"] = str(ri.get("requirement_type", "")).strip().lower()
            merged.append(item)
        out[cat] = merged
    return out


def _should_refine_with_openrouter() -> bool:
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        return False
    flag = (os.getenv("OPENROUTER_REFINE_FORMAT") or "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def _refine_with_openrouter(base: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    from openai import OpenAI

    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        return base

    model = (os.getenv("OPENROUTER_FORMAT_MODEL") or _DEFAULT_OPENROUTER_FORMAT_MODEL).strip()
    referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "http://localhost").strip()
    title = (os.getenv("OPENROUTER_APP_TITLE") or "Easy Tender").strip()

    client = OpenAI(
        base_url=_OPENROUTER_BASE,
        api_key=key,
        default_headers={
            "HTTP-Referer": referer,
            "X-Title": title,
        },
    )

    user_content = (
        "JSON d'exigences d'un appel d'offres marocain (déjà structuré). "
        "Améliore le français de description et criteria: orthographe, clarté, "
        "sans changer le sens juridique. requirement_type: chaîne vide ou 'cv' pour les lignes CV.\n\n"
        "Règles: réponds avec UN SEUL objet JSON, mêmes clés (dossier_administratif, offre_technique, "
        "offre_financiere), même nombre d'éléments par liste, même ordre, mêmes id. "
        "Ne pas ajouter/supprimer d'objets ni de clés (id, description, criteria, requirement_type).\n\n"
        f"{json.dumps(base, ensure_ascii=False)}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=8192,
        messages=[
            {
                "role": "system",
                "content": "Tu réponds uniquement avec du JSON valide, sans markdown ni texte autour.",
            },
            {"role": "user", "content": user_content},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    parsed = _parse_json_from_llm(raw)
    return _merge_openrouter_refine(base, parsed)


def format_requirements_payload(
    raw_requirements: dict[str, Any] | None,
    *,
    openrouter_refine: bool | None = None,
) -> dict[str, list[dict[str, str]]]:
    """
    Normalize extraction output to a stable schema.

    When ``OPENROUTER_API_KEY`` is set (and ``OPENROUTER_REFINE_FORMAT`` is not disabled),
    runs an extra pass via OpenRouter (default: free Mistral instruct) to polish French text.
    Override model with ``OPENROUTER_FORMAT_MODEL``. On API/parse errors, returns the deterministic result.
    """
    base = _format_requirements_deterministic(raw_requirements)
    do_refine = openrouter_refine if openrouter_refine is not None else _should_refine_with_openrouter()
    if not do_refine:
        return base
    try:
        return _refine_with_openrouter(base)
    except Exception:
        return base

