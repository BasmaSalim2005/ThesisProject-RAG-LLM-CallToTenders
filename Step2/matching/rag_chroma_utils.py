"""Shared helpers for admin-doc Chroma embedding and requirement matching."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document

from .condition_toggles import infer_condition_tags, tags_allowed
from .match_postprocess import is_date_validity_policy_requirement

CV_REQ_MARK = "cv requirement:"

_STOP = frozenset(
    """
    des les une un pour avec dans sur aux son sa ses leur le la ce cet cette ces
    que qui dont et ou mais minimum minime diplome certifie conforme original copie
    modele annexe signe electroniquement attestation reference
    """.split()
)


def flatten_requirements(payload: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    def push(category: str, req_id: str, text: str) -> None:
        text = " ".join(str(text).split())
        if not text:
            return
        tags = infer_condition_tags(text)
        items.append(
            {
                "category": category,
                "id": req_id,
                "text": text,
                "condition_tags": sorted(tags),
            }
        )

    for category, value in payload.items():
        if isinstance(value, list):
            for idx, entry in enumerate(value):
                if isinstance(entry, dict):
                    req_id = str(entry.get("id", f"{category}_{idx+1}"))
                    description = str(
                        entry.get("description", "")
                        or entry.get("summary", "")
                        or entry.get("matching_text", "")
                    ).strip()
                    legacy_title = str(entry.get("doc_name", "")).strip()
                    criteria = str(entry.get("criteria", "")).strip()

                    if not description and legacy_title:
                        text = legacy_title if not criteria else f"{legacy_title}. {criteria}".strip()
                    else:
                        text = description
                        if criteria and criteria not in text:
                            text = f"{text}. {criteria}".strip()

                    req_type = str(entry.get("requirement_type", "")).strip().lower()
                    if not req_type and category == "offre_technique":
                        if re.search(r"\bCVs?\b", text, flags=re.IGNORECASE):
                            req_type = "cv"
                    if req_type == "cv":
                        text = f"CV requirement: {text}".strip()
                    push(category, req_id, text)
                else:
                    raw = str(entry).strip()
                    if category == "offre_technique" and re.search(r"\bCVs?\b", raw, flags=re.IGNORECASE):
                        raw = f"CV requirement: {raw}"
                    push(category, f"{category}_{idx+1}", raw)
        elif isinstance(value, dict):
            for key, sub in value.items():
                push(category, str(key), str(sub))
        else:
            push(category, category, str(value))

    return items


def _fold_lower(s: str) -> str:
    s = (s or "").lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def parse_study_level_to_int(study: str) -> int | None:
    m = re.search(r"bac\s*\+\s*(\d+)", study or "", re.I)
    return int(m.group(1)) if m else None


def parse_min_bac_level(text: str) -> int | None:
    found: list[int] = []
    for m in re.finditer(r"bac\s*\+\s*(\d+)", text, re.I):
        found.append(int(m.group(1)))
    return max(found) if found else None


def parse_min_experience_years(text: str) -> int | None:
    """Heuristic: years of professional experience (ignore bac+N fragments)."""
    t = re.sub(r"bac\s*\+\s*\d+", " ", text, flags=re.I)
    t = re.sub(r"\(\s*0*(\d+)\s*\)", r" \1 ", t)
    vals: list[int] = []
    for m in re.finditer(
        r"\b(\d{1,2})\s*(?:ans|années)\b(?:\s*(?:d['’])?(?:expérience|experience|management|projets))?",
        t,
        re.I,
    ):
        span = m.group(0).lower()
        if "projet" in span and "exp" not in span and "experience" not in span and "expérience" not in span:
            continue
        n = int(m.group(1))
        if 1 <= n <= 45:
            vals.append(n)
    return max(vals) if vals else None


def _safe_int_exp(meta: dict[str, Any]) -> int | None:
    raw = meta.get("exp_years")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        v = int(str(raw).strip())
    except ValueError:
        return None
    return v if v >= 0 else None


def _cv_bac_plus(meta: dict[str, Any]) -> int | None:
    """Bac+N from metadata (numeric bac_plus or parsed study_level)."""
    raw = meta.get("bac_plus")
    if raw is not None and str(raw).strip().isdigit():
        return int(str(raw).strip())
    return parse_study_level_to_int(str(meta.get("study_level") or ""))


def role_keyword_overlap(requirement: str, title: str) -> float:
    """How well the CV title aligns with role words in the requirement (0–1)."""
    req = _fold_lower(requirement)
    if CV_REQ_MARK in req:
        req = req.split(CV_REQ_MARK, 1)[-1].strip()
    words = re.findall(r"[a-zàâäéèêëïîôùûç]{4,}", req)
    words = [w for w in words if w not in _STOP]
    if not words:
        return 0.45
    ttl = _fold_lower(title)
    hits = sum(1 for w in words if w in ttl)
    # Directeur / directrice / chef + projet (same role family for tenders)
    role_ask = bool(re.search(r"direct(eur|rice)\b", req) or re.search(r"\bchef\b", req))
    role_ttl = bool(re.search(r"direct(eur|rice)\b", ttl) or re.search(r"\bchef\b", ttl))
    proj_req = "projet" in req or "projets" in req
    proj_ttl = "projet" in ttl or "projets" in ttl
    if role_ask and role_ttl and proj_req and proj_ttl:
        hits += 3
    elif re.search(r"direct(eur|rice)\b", req) and re.search(r"direct(eur|rice)\b", ttl):
        hits += 1
    elif "chef" in req and "chef" in ttl and proj_req and proj_ttl:
        hits += 2
    # Penalize pure consulting titles when the ask is a project lead / director
    if role_ask and proj_req and re.search(r"consult", ttl) and not re.search(r"consult", req):
        hits -= 2
    denom = max(5, int(0.55 * len(words)) + 1)
    return max(0.12, min(1.0, hits / denom))


def cv_rank_key(requirement_text: str, doc: Document, embed_score: float) -> tuple:
    """
    How to order CVs for a requirement: (1) job title vs exigence, (2) more experience,
    (3) higher study level, (4) Chroma score. Titles are compared first; exp/Bac only break
    ties among similar title fits, including when no CV meets the minimum years (still
    return the best available).
    """
    meta = doc.metadata or {}
    title = str(meta.get("title") or "")
    ov = role_keyword_overlap(requirement_text, title)
    exp_i = _safe_int_exp(meta)
    exp_sort = exp_i if exp_i is not None else -1
    cand_bac = _cv_bac_plus(meta)
    bac_sort = cand_bac if cand_bac is not None else -1
    return (ov, exp_sort, bac_sort, float(embed_score))


def cv_composite_score(requirement_text: str, embed_score: float, meta: dict[str, Any]) -> float:
    """
    Report score: mirrors title-first logic without harsh multipliers (ranking is done by
    cv_rank_key). embed_score: higher = better (Chroma relevance).
    """
    exp_i = _safe_int_exp(meta)
    title = str(meta.get("title") or "")
    min_exp = parse_min_experience_years(requirement_text)
    min_bac = parse_min_bac_level(requirement_text)
    cand_bac = _cv_bac_plus(meta)
    overlap = role_keyword_overlap(requirement_text, title)

    score = float(embed_score) * (0.25 + 0.75 * overlap)

    if min_exp is not None and exp_i is not None and exp_i < min_exp:
        score *= 0.82
    if min_bac is not None and cand_bac is not None and cand_bac < min_bac:
        score *= 0.88
    if min_exp is not None and exp_i is not None and exp_i >= min_exp:
        score += min(0.2, (exp_i - min_exp) * 0.02)
    return score


def is_cv_requirement(text: str) -> bool:
    return _fold_lower(text).startswith(_fold_lower(CV_REQ_MARK))


def _admin_rows_to_documents(rows: list[Any]) -> list[Document]:
    docs: list[Document] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = row.get("path", "")
        fname = Path(str(path)).name if path else ""
        summary = row.get("summary") or row.get("summary_first_page", "")
        preview = row.get("extracted_text_preview") or row.get("first_page_excerpt", "")
        validity = row.get("issue_or_validity_date")
        parts = []
        if fname:
            parts.append(f"Fichier: {fname}")
        parts.append(f"Summary: {summary}")
        if preview:
            parts.append(f"Text preview: {preview}")
        if validity:
            parts.append(f"Validity/issue date: {validity}")
        page_content = "\n".join(parts).strip()
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": path,
                    "issue_or_validity_date": validity or "",
                    "doc_type": "admin",
                },
            )
        )
    return docs


def load_admin_documents(index_path: Path) -> list[Document]:
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Admin index must be a JSON array: {index_path}")
    return _admin_rows_to_documents(rows)


def _documents_from_cv_object(data: dict[str, Any]) -> list[Document]:
    """Build documents from {\"cv_database\": [{id, name, title, ...}, ...]}."""
    rows = data.get("cv_database", [])
    if not isinstance(rows, list):
        raise ValueError("cv_database must be a list")
    docs: list[Document] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        path_s = str(row.get("path", ""))
        name = str(row.get("name", "")).strip()
        title = str(row.get("title", "")).strip()
        study = str(row.get("study_level", "")).strip()
        years = row.get("exp_years", "")
        projects = str(row.get("projects", "")).strip()
        parts = [
            f"CV candidat: {name}",
            f"Intitulé / rôle: {title}",
            f"Formation: {study}",
            f"Années d'expérience: {years}",
            f"Projets / compétences: {projects}",
        ]
        page_content = "\n".join(parts).strip()
        exp_raw = row.get("exp_years", "")
        try:
            exp_years = int(exp_raw) if exp_raw is not None and str(exp_raw).strip() != "" else -1
        except (TypeError, ValueError):
            exp_years = -1
        bac_n = parse_study_level_to_int(study)
        docs.append(
            Document(
                page_content=page_content or f"CV {row.get('id', '')}",
                metadata={
                    "source": path_s,
                    "issue_or_validity_date": "",
                    "doc_type": "cv",
                    "cv_id": str(row.get("id", "")),
                    "exp_years": str(exp_years) if exp_years >= 0 else "",
                    "study_level": study,
                    "title": title,
                    "cv_name": name,
                    "bac_plus": str(bac_n) if bac_n is not None else "",
                },
            )
        )
    return docs


def load_cv_database(cv_path: Path) -> list[Document]:
    """Load data/hr/cv.json shape: {\"cv_database\": [{id, name, title, ...}, ...]}."""
    data = json.loads(cv_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"CV file must be a JSON object: {cv_path}")
    return _documents_from_cv_object(data)


def load_documents_from_file(path: Path) -> list[Document]:
    """Dispatch: admin index (JSON array) or CV catalogue (object with cv_database)."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return _admin_rows_to_documents(raw)
    if isinstance(raw, dict) and "cv_database" in raw:
        return _documents_from_cv_object(raw)
    raise ValueError(
        f"Unsupported JSON in {path}: expected a list (admin index) or an object with 'cv_database'."
    )


def _embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings()


def build_chroma_from_sources(
    source_paths: list[Path],
    chroma_dir: Path,
    *,
    force: bool = False,
) -> int:
    """
    Embed one or more JSON sources into a single Chroma store (admin index, cv.json, etc.).
    If chroma_dir already has data and force is False, exits with SystemExit.
    Returns number of documents embedded.
    """
    if not source_paths:
        raise ValueError("At least one source JSON path is required.")
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        if not force:
            raise SystemExit(
                f"Chroma directory is not empty: {chroma_dir}\n"
                "Use --force to delete it and rebuild, or pick a different --chroma-dir."
            )
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    for p in source_paths:
        if not p.exists():
            raise FileNotFoundError(f"Source not found: {p}")
        documents.extend(load_documents_from_file(p))
    embeddings = _embeddings()
    Chroma.from_documents(documents, embedding=embeddings, persist_directory=str(chroma_dir))
    return len(documents)


def build_chroma_from_index(
    index_path: Path,
    chroma_dir: Path,
    *,
    force: bool = False,
) -> int:
    """Backward-compatible single-file embed (admin array index)."""
    return build_chroma_from_sources([index_path], chroma_dir, force=force)


def load_chroma_db(chroma_dir: Path) -> Chroma:
    """Open an existing Chroma store (same embedding model as build)."""
    if not chroma_dir.is_dir() or not any(chroma_dir.iterdir()):
        raise FileNotFoundError(
            f"Chroma store missing or empty: {chroma_dir}. "
            "Run: python -m Step2.embedding.embed_admin_chroma --chroma-dir "
            f"{chroma_dir} --force"
        )
    return Chroma(persist_directory=str(chroma_dir), embedding_function=_embeddings())


def match_display_label(doc: Document) -> str:
    """Single human-readable line for a match (path, or CV name + id)."""
    md = doc.metadata or {}
    if md.get("doc_type") == "cv":
        name = str(md.get("cv_name") or "").strip()
        cid = str(md.get("cv_id") or "").strip()
        path = str(md.get("source") or "").strip()
        if name and cid:
            return f"{name} [{cid}]"
        if path:
            return path
        return cid or "CV"
    path = str(md.get("source") or "").strip()
    return path or "document"


def run_matching(
    requirements: list[dict[str, Any]],
    db: Chroma,
    top_k: int = 3,
    *,
    toggles: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for req in requirements:
        raw_tags = req.get("condition_tags") or ()
        ctags = frozenset(str(t) for t in raw_tags)
        if not tags_allowed(ctags, toggles):
            output.append(
                {
                    "id": req.get("id", ""),
                    "skipped": True,
                    "skip_reason": "condition_toggles",
                    "condition_tags": sorted(ctags),
                    "matches": [],
                }
            )
            continue

        if is_date_validity_policy_requirement(req):
            output.append(
                {
                    "id": req.get("id", ""),
                    "skipped": True,
                    "skip_reason": "date_validation_context",
                    "matches": [],
                }
            )
            continue

        query = str(req.get("text", ""))
        if is_cv_requirement(query):
            retrieve_k = min(120, max(40, top_k * 12))
            pool = db.similarity_search_with_relevance_scores(query, k=retrieve_k)
            cv_hits = [(d, s) for d, s in pool if (d.metadata or {}).get("doc_type") == "cv"]
            if cv_hits:
                ranked = sorted(
                    cv_hits,
                    key=lambda ds: cv_rank_key(query, ds[0], ds[1]),
                    reverse=True,
                )
                results = ranked[:top_k]
            else:
                results = pool[:top_k]
        else:
            results = db.similarity_search_with_relevance_scores(query, k=top_k)

        matches_out: list[dict[str, Any]] = []
        for doc, chroma_s in results:
            matches_out.append(_match_row(query, doc, float(chroma_s)))
        output.append({"id": req["id"], "matches": matches_out})
    return output


def _match_row(query: str, doc: Document, chroma_s: float) -> dict[str, Any]:
    """Per-hit fields for the report: scores, type, path, light CV/admin extras."""
    md = doc.metadata or {}
    label = match_display_label(doc)
    ctype = str(md.get("doc_type") or "admin")
    path = str(md.get("source") or "").strip()
    chroma_r = round(float(chroma_s), 4)
    if is_cv_requirement(query) and ctype == "cv":
        final_r = round(cv_composite_score(query, chroma_s, md), 4)
    else:
        final_r = chroma_r

    row: dict[str, Any] = {
        "match": label,
        "score": final_r,
        "chroma_score": chroma_r,
        "doc_type": ctype,
        "path": path or None,
    }
    if ctype == "cv":
        exp = _safe_int_exp(md)
        if exp is not None:
            row["exp_years"] = exp
        bac_raw = md.get("bac_plus")
        if bac_raw is not None and str(bac_raw).strip().isdigit():
            row["bac_plus"] = int(str(bac_raw))
        cid = str(md.get("cv_id") or "").strip()
        if cid:
            row["cv_id"] = cid
        ttl = str(md.get("title") or "").strip()
        if ttl:
            row["title"] = ttl
    else:
        val = md.get("issue_or_validity_date")
        if val:
            row["validity"] = val
    return row
