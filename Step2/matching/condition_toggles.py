"""Optional RAG match scopes (groupement, cas particuliers, etc.).

Toggles default to True: every scope is included until you set one to false.
A requirement is skipped if any of its detected ``condition_tags`` is disabled
(all listed tags for that row must be active for the row to run).
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

# --- Single normalization path for pattern matching (no import cycle with rag_chroma_utils) ---


def _fold(s: str) -> str:
    s = (s or "").lower()
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _any_pattern(patterns: tuple[re.Pattern[str], ...], text_f: str) -> bool:
    return any(p.search(text_f) for p in patterns)


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """One user-toggleable situation detected from requirement text (French tenders)."""

    id: str
    label_fr: str
    patterns: tuple[re.Pattern[str], ...]

    def matches(self, text_folded: str) -> bool:
        return _any_pattern(self.patterns, text_folded)


def _pats(*regex_parts: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in regex_parts)


# Order matters: more specific “situations” first; generic `cas_particulier` is last.
_SPECS: tuple[ConditionSpec, ...] = (
    ConditionSpec(
        id="groupement",
        label_fr="Groupement / U.T.E. / mandataire (exigences spécifiques à un groupement)",
        patterns=_pats(
            r"\bgroupement\b",
            r"u\.?\s*\.?\s*t\.?\s*\.?\s*e\.?\b",
            r"concurrent\s+en\s+groupement",
            r"prestations\s+realisees?\s+en\s+groupement",
            r"membre\s+du\s+groupe",
            r"mandataire.*(groupe|procur|membre)|membre.*mandataire|par\s+le\s+mandataire",
            r"membres?\s+du\s+groupement",
            r"cas_groupement",
        ),
    ),
    ConditionSpec(
        id="hors_maroc",
        label_fr="Concurrent non installé au Maroc / international (équivalents, devises, etc.)",
        patterns=_pats(
            r"non\s+installe?s?\s+au\s+maroc",
            r"concurrente?s?\s+non\s+installe",
            r"hors\s+maroc",
            r"non[\s-]+installe?s?\s+au\s+maroc",
            r"\bpays\s+d['\u2019]origine\b",
            r"autorites?\s+competentes?\s+du\s+pays",
            r"etablis(sement)?s?\s+a\s+l?etranger",
        ),
    ),
    ConditionSpec(
        id="sous_traitance",
        label_fr="Sous-traitance / co-traitance",
        patterns=_pats(
            r"sous[-\s]traitance",
            r"sous[-\s]traitant",
            r"co[-\s]?trait",
        ),
    ),
    ConditionSpec(
        id="cas_particulier",
        label_fr='Autres formulations « en cas de… », « le cas échéant », etc.',
        patterns=_pats(
            r"\ben\s+cas\s+d['\u2019]?\w+",
            r"\ben\s+cas\s+de\s+",
            r"\ble\s+cas\s+echeant\b",
            r"\ble\s+cas\s+ou\b",
        ),
    ),
)

SPEC_BY_ID: dict[str, ConditionSpec] = {s.id: s for s in _SPECS}


def condition_specs() -> tuple[ConditionSpec, ...]:
    return _SPECS


def all_condition_ids() -> tuple[str, ...]:
    return tuple(s.id for s in _SPECS)


def default_toggles() -> dict[str, bool]:
    return {s.id: True for s in _SPECS}


def _dedupe_cas_if_specific(tags: set[str]) -> set[str]:
    if tags & ({"groupement", "hors_maroc", "sous_traitance"}):
        tags.discard("cas_particulier")
    return tags


def infer_condition_tags(text: str) -> frozenset[str]:
    """
    Infer which optional situations a requirement line refers to, from the full match text
    (description + criteria + CV mark as produced by ``flatten_requirements``).
    """
    t = _fold(text)
    found: set[str] = set()
    for spec in _SPECS:
        if spec.matches(t):
            found.add(spec.id)
    return frozenset(_dedupe_cas_if_specific(found))


def tags_allowed(
    tags: frozenset[str] | set[str] | list[str] | None,
    toggles: Mapping[str, bool] | None,
) -> bool:
    """
    If ``toggles`` is None, all requirements are included (previous behaviour).
    Otherwise each tag on the item must be True in ``toggles`` (missing key defaults to True).
    """
    if not tags:
        return True
    if toggles is None:
        return True
    for t in tags:
        if not bool(toggles.get(t, True)):
            return False
    return True


def parse_toggles_dict(raw: Any) -> dict[str, bool]:
    """Merge a JSON object with boolean values into a full toggles map (unknown keys dropped)."""
    out = default_toggles()
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        ks = str(k).strip()
        if ks in out:
            out[ks] = bool(v)
    return out


def merge_toggles(
    base: Mapping[str, bool] | None,
    overrides: Mapping[str, bool | None] | None,
) -> dict[str, bool]:
    out = {**default_toggles(), **(dict(base) if base else {})}
    if not overrides:
        return out
    for k, v in overrides.items():
        if v is not None and k in out:
            out[k] = v
    return out


def iter_toggles_file_lines(toggles: dict[str, bool]) -> Iterator[str]:
    """One-line description per toggle for help text or logs."""
    for tid in all_condition_ids():
        spec = SPEC_BY_ID[tid]
        st = "oui" if toggles.get(tid) else "non"
        yield f"  {tid}: {st} — {spec.label_fr}"


def add_condition_toggle_args(parser: argparse.ArgumentParser) -> None:
    """``--<id> / --no-<id>`` for each known scope (e.g. ``--groupement`` / ``--no-groupement``)."""
    for spec in _SPECS:
        flag = spec.id.replace("_", "-")
        parser.add_argument(
            f"--{flag}",
            action=argparse.BooleanOptionalAction,
            dest=spec.id,
            default=None,
            help=(
                f"Inclure les exigences détectées pour « {spec.id} ». "
                f"Par défaut: oui. Utiliser --no-{flag} pour les exclure du RAG. "
                f"({spec.label_fr})"
            ),
        )


def build_toggles_from_parsed_args(
    args: Any,
    toggles_path: Path | None,
) -> dict[str, bool]:
    """JSON file (optional) is applied first, then per-flag CLI overrides."""
    if toggles_path and toggles_path.is_file():
        base = parse_toggles_dict(json.loads(toggles_path.read_text(encoding="utf-8")))
    else:
        base = default_toggles()
    for spec in _SPECS:
        v = getattr(args, spec.id, None)
        if v is not None:
            base[spec.id] = v
    return base
