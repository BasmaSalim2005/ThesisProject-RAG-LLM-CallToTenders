"""Post-processing after vector matching: date-policy skips (already in matcher) and
bundled coverage for « déclaration sur l’honneur » when the RC or registre de commerce
requirement already matches a document (modèle typiquement joint à ces pièces).
"""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any

# Row that only encodes the global publication / freshness window (PDF report uses it; no RAG).
# Legacy (slug) and new file-like id from ``format_requirements``:
_DATE_VALIDITY_IDS = frozenset({"Date de validite", "regle_fraicheur_pieces"})


def _is_date_validity_id_or_text(req: dict[str, Any]) -> bool:
    rid = str(req.get("id", "")).strip()
    if rid in _DATE_VALIDITY_IDS:
        return True
    t = f"{req.get('text', '')} {req.get('id', '')}"
    t_low = t.lower()
    if "valid" in t_low and ("1 ans" in t_low or "un an" in t_low) and (
        "publication" in t_low or "annonce" in t_low or "appel" in t_low
    ):
        return True
    if t_low.startswith("les document") and "valide" in t_low and "1" in t_low:
        return True
    return False

_DECL_RE = re.compile(
    r"d[ée]claration\s+sur\s+l['\u2019]?honneur|honneur.*d[ée]claration",
    re.IGNORECASE,
)
_RC_RE = re.compile(
    r"r[èe]glement\s+de\s+consultation|r[eè]glement.*consultation|\(rc\)|\brc\b.*sign",
    re.IGNORECASE,
)
# Cibler la piece « certificat / immatriculation (modele 9) », pas les mentions generiques « registre de commerce ».
_REG_RE = re.compile(
    r"certificat\s+d['\u2019]?immatriculation|immatriculation\s+au\s+registre|"
    r"registre\s+de\s+commerce\s*\(mod[èe]le\s*9\)|mod[èe]le\s*9|modele\s*9",
    re.IGNORECASE,
)


def _fold_id(s: str) -> str:
    t = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def is_date_validity_policy_requirement(req: dict[str, Any]) -> bool:
    return _is_date_validity_id_or_text(req)


def is_declaration_sur_honneur_row(req: dict[str, Any]) -> bool:
    i = _fold_id(str(req.get("id", "")))
    if "honneur" in i and "declaration" in i.replace("é", "e"):
        return True
    blob = f"{req.get('text', '')} {req.get('id', '')}"
    return bool(_DECL_RE.search(blob))


def is_rc_reglement_row(req: dict[str, Any]) -> bool:
    if is_declaration_sur_honneur_row(req):
        return False
    blob = f"{req.get('text', '')} {req.get('id', '')}"
    return bool(_RC_RE.search(blob))


def is_registre_commerce_row(req: dict[str, Any]) -> bool:
    blob = f"{req.get('text', '')} {req.get('id', '')}"
    if is_declaration_sur_honneur_row(req) or is_rc_reglement_row(req):
        return False
    return bool(_REG_RE.search(blob))


def _top_matches(block: dict[str, Any], n: int) -> list[dict[str, Any]]:
    if block.get("skipped"):
        return []
    raw = block.get("matches") or []
    out: list[dict[str, Any]] = []
    for m in raw[:n]:
        if isinstance(m, dict) and (m.get("path") or m.get("match")):
            out.append(m)
    return out


def _dedupe_path_key(m: dict[str, Any]) -> str:
    p = (m.get("path") or m.get("match") or "").strip()
    return p or id(m)  # fallback


def apply_declaration_bundles(
    report: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    *,
    top_from_each: int = 1,
) -> list[dict[str, Any]]:
    """
    If « déclaration sur l’honneur » is present, add inherited matches (copies, tagged) from
    the best RC and/or registre de commerce rows when those rows have non-empty matches.
    Does not remove the original search results for the déclaration line.
    """
    if len(report) != len(requirements):
        return report
    out = [deepcopy(b) for b in report]
    decl_idx = [i for i, r in enumerate(requirements) if is_declaration_sur_honneur_row(r)]
    if not decl_idx:
        return out
    rc_idx = [i for i, r in enumerate(requirements) if is_rc_reglement_row(r)]
    reg_idx = [i for i, r in enumerate(requirements) if is_registre_commerce_row(r)]

    for d in decl_idx:
        block = out[d]
        if block.get("skipped"):
            continue
        had_paths: set[str] = set()
        for m in block.get("matches") or []:
            if isinstance(m, dict):
                had_paths.add(_dedupe_path_key(m))

        extras: list[dict[str, Any]] = []
        for label, idx_list, source_note in (
            ("RC", rc_idx, "Modèle annexé au règlement de consultation (RC) : la même pièce couvre souvent la déclaration sur l’honneur."),
            (
                "Registre de commerce",
                reg_idx,
                "Déclaration / formulaires souvent regroupés avec le certificat d’immatriculation : proposition héritée de l’exigence « registre de commerce ».",
            ),
        ):
            for src_i in idx_list:
                for m in _top_matches(out[src_i], top_from_each):
                    k = _dedupe_path_key(m)
                    if not k or k in had_paths:
                        continue
                    had_paths.add(k)
                    m2 = deepcopy(m)
                    m2["bundled"] = True
                    m2["bundle_source"] = label
                    m2["bundle_source_requirement_id"] = str(requirements[src_i].get("id", ""))
                    m2["bundle_note"] = source_note
                    extras.append(m2)
        if extras:
            block["matches"] = list(block.get("matches") or []) + extras
    return out


def _global_doc_key(m: dict[str, Any]) -> str:
    """Stable key for “this file / this CV” across requirements (paths normalized)."""
    if not isinstance(m, dict):
        return ""
    p = (m.get("path") or "").strip()
    if p:
        return p.replace("\\", "/").lower()
    if str(m.get("doc_type", "")) == "cv" or m.get("cv_id"):
        cid = str(m.get("cv_id") or "").strip()
        if cid:
            return f"cv:{cid}"
    lab = str(m.get("match", "")).strip()
    if lab:
        return f"l:{lab[:200]}"
    return ""


def apply_one_unused_match_per_requirement(
    report: list[dict[str, Any]],
    *,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """
    One output match per requirement, in report order: pick the first candidate in score order
    whose file was not already chosen for a previous requirement; if all are taken, keep the best.
    """
    if not enabled:
        return report
    used: set[str] = set()
    out: list[dict[str, Any]] = []
    for block in report:
        b = deepcopy(block)
        if b.get("skipped") or not b.get("matches"):
            out.append(b)
            continue
        raw = [m for m in (b.get("matches") or []) if isinstance(m, dict)]
        pick: dict[str, Any] | None = None
        for m in raw:
            k = _global_doc_key(m)
            if not k:
                pick = deepcopy(m)
                break
            if k not in used:
                used.add(k)
                pick = deepcopy(m)
                break
        if pick is None and raw:
            pick = deepcopy(raw[0])
        b["matches"] = [pick] if pick is not None else []
        out.append(b)
    return out
