"""Build a human-readable PDF from requirements_matches.json + formated_requirements.json."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Step2.matching.rag_chroma_utils import (
    flatten_requirements,
    is_cv_requirement,
    parse_min_bac_level,
    parse_min_experience_years,
    role_keyword_overlap,
)

# SegoeUI embedded subset used by fpdf2 can lack arrows / bullets / some punctuation in Bold.
# This avoids "missing glyph" and "MERG NOT subset" when printing.
def _t(s: str | Any) -> str:
    if s is None:
        return ""
    t = str(s)
    if not t:
        return t
    out: list[str] = []
    for ch in t:
        o = ord(ch)
        if o == 0x21B3:  # ↳ (arrow often missing in TTF subset)
            out.append("->")
        elif 0x2190 <= o <= 0x21FF:
            out.append("->")
        elif o == 0x00B7:  # middle dot
            out.append("-")
        elif o in (0x2014, 0x2013, 0x2015):
            out.append(" - ")
        elif o in (0x00AB, 0x00BB):
            out.append('"')
        elif o in (0x2018, 0x2019, 0x201C, 0x201D):
            out.append("'")
        elif o == 0x00A0:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


# Formated-requirements entry that defines the global publication / freshness rule
# (legacy id or new file-like id from format_requirements).
VALIDITY_POLICY_ENTRY_ID = "Date de validite"
_REGLE_FRAICHEUR_ID = "regle_fraicheur_pieces"
_VALIDITY_POLICY_IDS = frozenset({VALIDITY_POLICY_ENTRY_ID, _REGLE_FRAICHEUR_ID})

_SECT_COLORS: dict[str, tuple[int, int, int]] = {
    "dossier_administratif": (25, 55, 109),
    "offre_technique": (0, 100, 75),
    "offre_financiere": (140, 75, 0),
}
_SECT_TITLES: dict[str, str] = {
    "dossier_administratif": "Dossier administratif",
    "offre_technique": "Offre technique",
    "offre_financiere": "Offre financiere",
}  # accents limitees pour compat. polices; possible d’enrichir avec TTF

_MONTHS_FR: dict[str, int] = {
    "janvier": 1,
    "fevrier": 2,
    "février": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "août": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
    "décembre": 12,
}


@dataclass(frozen=True)
class ValidityPolicy:
    """Documents should carry a date inside [window_start, window_end] inclusive (issue / validity)."""

    window_start: date
    window_end: date
    summary: str


def _parse_iso_date(s: str) -> date | None:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _entry_looks_like_validity_policy(entry: dict[str, Any]) -> bool:
    desc = f"{entry.get('description', '')} {entry.get('criteria', '')}"
    d = desc.lower()
    if "valid" in d and ("1 ans" in d or "un an" in d) and (
        "publication" in d or "annonce" in d or "appel" in d
    ):
        return True
    if d.strip().startswith("les document") and "valide" in d and "1" in d:
        return True
    return False


def _find_policy_entry(formated: dict[str, Any]) -> dict[str, Any] | None:
    for cat in ("dossier_administratif", "offre_technique", "offre_financiere"):
        for entry in formated.get(cat, []) or []:
            if not isinstance(entry, dict):
                continue
            rid = str(entry.get("id", "")).strip()
            if rid in _VALIDITY_POLICY_IDS or _entry_looks_like_validity_policy(entry):
                return entry
    return None


def _add_calendar_years(d: date, delta_years: int) -> date:
    y = d.year + delta_years
    try:
        return d.replace(year=y)
    except ValueError:
        return d.replace(year=y, month=2, day=28)


def _parse_window_years_from_text(text: str) -> int | None:
    m = re.search(r"\b(\d{1,2})\s*ans?\b", text, re.IGNORECASE)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 30 else None


def load_validity_policy(formated_path: Path) -> ValidityPolicy | None:
    """
    Reads the rule from the date / fraicheur row (id ``regle_fraicheur_pieces`` or legacy
    ``Date de validite``), optional structured fields:
      - `validity_reference_date` (ISO or DD/MM/YYYY)
      - `validity_window_years` (int, default 1)
    If missing, parses French prose in `description` (e.g. '25 mars 2023' + '1 ans').
    Window = [reference_date - N years, reference_date] inclusive.
    """
    data = json.loads(formated_path.read_text(encoding="utf-8"))
    entry = _find_policy_entry(data)
    if not entry:
        return None

    ref: date | None = None
    years = 1
    raw_ref = str(entry.get("validity_reference_date", "")).strip()
    if raw_ref:
        ref = _parse_iso_date(raw_ref)

    desc = f"{entry.get('description', '')} {entry.get('criteria', '')}"
    if "validity_window_years" in entry and str(entry.get("validity_window_years", "")).strip() != "":
        try:
            years = int(entry["validity_window_years"])
        except (TypeError, ValueError):
            years = 1
    else:
        parsed_y = _parse_window_years_from_text(desc)
        if parsed_y is not None:
            years = parsed_y
    years = max(1, min(years, 30))

    if ref is None:
        ref = _parse_french_day_month_year(desc)
    if ref is None:
        return None

    window_end = ref
    window_start = _add_calendar_years(ref, -years)

    summ = (
        f"Regle de fraicheur (ligne d'id {str(entry.get('id', '')).strip() or 'regle_fraicheur'}) : une date extraite de "
        f"l'index pour chaque piece administrative est comparee a la fenetre "
        f"[{window_start.isoformat()} ; {window_end.isoformat()}] "
        f"({years} an(s) avant la date de reference du {ref.isoformat()})."
    )
    return ValidityPolicy(window_start=window_start, window_end=window_end, summary=_t(summ))


def _parse_french_day_month_year(text: str) -> date | None:
    m = re.search(
        r"\b(\d{1,2})\s+([a-zA-Zàâäéèêëïîôùûç]+)\s+(\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    d, mon, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
    month = _MONTHS_FR.get(mon)
    if not month:
        return None
    try:
        return date(y, month, d)
    except ValueError:
        return None


def _parse_dates_dmy(text: str) -> list[date]:
    out: list[date] = []
    for m in re.finditer(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b", text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y < 70 else 1900
        try:
            out.append(date(y, mo, d))
        except ValueError:
            continue
    return out


def document_validity_assessment(policy: ValidityPolicy, validity_index_field: str | Any) -> list[str]:
    """Compare indexed validity string to the policy window."""
    raw = str(validity_index_field or "").strip()
    if not raw:
        return [
            "Aucune date de validite/emission dans l'index pour ce document — impossible de verifier automatiquement la fraicheur."
        ]
    dates = _parse_dates_dmy(raw)
    if not dates:
        return [
            f"Date dans l'index ({raw[:80]}...) non lisible automatiquement (format attendu JJ/MM/AAAA) — "
            "comparer a la main a la regle de publication."
        ]
    doc_ref = max(dates)
    notes: list[str] = []
    if doc_ref < policy.window_start:
        notes.append(
            f"Document retrouvé, mais la date extraite ({doc_ref.isoformat()}) est anterieure au debut "
            f"de la fenetre autorisée ({policy.window_start.isoformat()}) — probablement trop ancien "
            f"pour l'exigence de fraicheur (reference {policy.window_end.isoformat()})."
        )
    elif doc_ref > policy.window_end:
        notes.append(
            f"Date extraite ({doc_ref.isoformat()}) posterieure a la date de reference du dossier "
            f"({policy.window_end.isoformat()}) — verifier si la piece est correcte."
        )
    else:
        notes.append(
            f"Date extraite ({doc_ref.isoformat()}) dans la fenetre [{policy.window_start.isoformat()} ; "
            f"{policy.window_end.isoformat()}] — OK au regard automatique de la regle saisie."
        )
    return notes


def _load_requirement_labels(formated_path: Path) -> dict[str, str]:
    data = json.loads(formated_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for cat in ("dossier_administratif", "offre_technique", "offre_financiere"):
        for entry in data.get(cat, []) or []:
            if not isinstance(entry, dict):
                continue
            rid = str(entry.get("id", "")).strip()
            if not rid:
                continue
            desc = str(entry.get("description", "")).strip()
            crit = str(entry.get("criteria", "")).strip()
            txt = desc
            if crit and crit not in txt:
                txt = f"{txt} ({crit})".strip() if txt else crit
            out[rid] = txt or rid
    return out


def _windows_fonts() -> tuple[str | None, str, str | None]:
    """(regular TTF, family name, optional bold TTF) for Latin + general Unicode."""
    root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name, bold_name in (("segoeui.ttf", "segoeuib.ttf"), ("arial.ttf", "arialbd.ttf"), ("calibri.ttf", "calibrib.ttf")):
        pr = root / name
        pb = root / bold_name
        if pr.is_file():
            return (str(pr), "ReportUnicode", str(pb) if pb.is_file() else None)
    return (None, "Helvetica", None)


def _admin_notes(m: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    sc = float(m.get("chroma_score") or m.get("score") or 0)
    if sc < 0.42:
        notes.append(
            "Pertinence modérée selon le score automatique — vérification manuelle recommandée."
        )
    return notes


def _cv_notes(requirement_text: str, m: dict[str, Any]) -> list[str]:
    if m.get("doc_type") != "cv" or not is_cv_requirement(requirement_text):
        return []
    notes: list[str] = []
    min_exp = parse_min_experience_years(requirement_text)
    exp = m.get("exp_years")
    if min_exp is not None and isinstance(exp, int) and exp < min_exp:
        notes.append(
            f"Exigence d'expérience : au moins {min_exp} an(s) mentionné(s) ; ce CV indique {exp} an(s). "
            f"Classement par intitulé de poste ; ce profil n'est pas pleinement conforme. "
            f"Lors de l'évaluation de l'offre, l'écart peut conduire à pénaliser la note (selon le règlement de consultation) ; "
            f"c'est en tout cas la meilleure entrée du catalogue pour cette exigence."
        )
    min_bac = parse_min_bac_level(requirement_text)
    bac = m.get("bac_plus")
    if min_bac is not None and isinstance(bac, int) and bac < min_bac:
        notes.append(
            f"Exigence de formation : au moins Bac+{min_bac} ; ce profil correspond à Bac+{bac}. "
            f"Classement par intitulé puis par niveau d'études : correspondance partielle. "
            f"L'offre pourra être pénalisée au barème d'évaluation, selon le RC — à valider en interne."
        )
    title = str(m.get("title") or "")
    if title:
        ov = role_keyword_overlap(requirement_text, title)
        if ov < 0.38:
            notes.append(
                "L'intitulé du poste ne correspond qu'approximativement au libellé de l'exigence "
                "(même famille de métiers ou mots-clés proches) — proposition la plus proche en base."
            )
    if notes:
        notes.insert(
            0,
            "Lecture automatique : ce rapport ne remplace pas une analyse humaine ; "
            "il classe les documents disponibles.",
        )
    return notes


def _match_notes(
    requirement_id: str,
    requirement_text: str,
    m: dict[str, Any],
    *,
    validity_policy: ValidityPolicy | None = None,
) -> list[str]:
    if m.get("doc_type") == "cv":
        return [_t(n) for n in _cv_notes(requirement_text, m)]
    notes = _admin_notes(m)
    if m.get("bundled"):
        notes.insert(
            0,
            "Proposition reprise d'une autre exigence (RC ou registre) : souvent la meme piece couvre aussi la declaration sur l'honneur - controle visuel recommande.",
        )
        return [_t(n) for n in notes]
    if (
        validity_policy
        and str(requirement_id).strip() not in _VALIDITY_POLICY_IDS
        and m.get("doc_type") != "cv"
    ):
        notes.extend(document_validity_assessment(validity_policy, m.get("validity")))
    return [_t(n) for n in notes]


def _colored_bar_header(
    pdf: FPDF, family: str, has_b: bool, w: float, title: str, subtitle: str, rgb: tuple[int, int, int]
) -> None:
    # multi_cell() defaults to new_x=RIGHT; the next full-width cell would otherwise start at the
    # right margin and look like a second column of clipped text.
    pdf.set_x(pdf.l_margin)
    pdf.set_fill_color(*rgb)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(family, "B" if has_b else "", 12)
    pdf.cell(w, 7, _t(f"  {title}"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font(family, "", 9)
    pdf.cell(0, 5, _t(f"  {subtitle}"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


class _ReportPDF(FPDF):
    def __init__(
        self,
        font_file: str | None,
        font_bold: str | None,
        font_family: str,
        *,
        title: str = "Rapport de correspondance",
    ) -> None:
        super().__init__()
        self._font_file = font_file
        self._font_bold = font_bold
        self._font_family = font_family
        self._doc_title = title
        self._gen_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    def setup_fonts(self) -> None:
        if self._font_file:
            self.add_font(self._font_family, "", self._font_file)
        if self._font_bold:
            self.add_font(self._font_family, "B", self._font_bold)
        self.set_font(self._font_family, "B" if self._font_bold else "", 11)

    def header(self) -> None:  # noqa: D401
        if self.page_no() == 1:
            self.set_fill_color(25, 55, 109)
            self.set_text_color(255, 255, 255)
            self.set_y(6)
            self.set_font(self._font_family, "B" if self._font_bold else "", 16)
            self.cell(0, 9, _t(self._doc_title), align="C", new_x="LMARGIN", new_y="NEXT", fill=True)
            self.set_font(self._font_family, "", 9)
            self.set_fill_color(35, 65, 120)
            self.cell(
                0,
                6,
                _t(
                    f"Génère le  {self._gen_at}  -  Aucun JSON dans ce document : texte de synthèse uniquement"
                ),
                align="C",
                new_x="LMARGIN",
                new_y="NEXT",
                fill=True,
            )
            self.set_text_color(0, 0, 0)
            self.ln(4)
        # Pages 2+ : pas de texte a droite ; le pied de page porte "Page n".
        else:
            self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(self._font_family, "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _unwrap_matches_file_payload(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "matches" in raw:
        m = raw["matches"]
        if not isinstance(m, list):
            raise ValueError("matches file: 'matches' must be a list")
        return m
    raise ValueError("Expected a JSON list or an object with a 'matches' array")


def _append_specifications_appendix(
    pdf: FPDF,
    family: str,
    has_b: bool,
    w: float,
    spec_path: Path | None,
) -> None:
    """
    One extra section after the RAG table: business context from a specifications JSON
    (e.g. step1/v2/specifications_extracted.json). Omitted if path is None, missing, or empty.
    """
    if not spec_path or not spec_path.is_file():
        return
    try:
        raw: Any = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(raw, dict) or not raw:
        return

    ordered: list[tuple[str, str]] = [
        ("PERIMETRE_DU_PROJET", "Perimetre du projet (synthese)"),
        ("DEROULEMENT_DU_PROJET", "Deroulement du projet"),
        ("CONSISTANCE_DES_PRESTATIONS", "Consistance des prestations (synthese)"),
    ]
    parts: list[tuple[str, str]] = []
    for key, sub_title in ordered:
        val = raw.get(key)
        if val and str(val).strip():
            parts.append((sub_title, str(val).strip()))
    for key, val in raw.items():
        if key in {k for k, _ in ordered}:
            continue
        if val and str(val).strip():
            parts.append((str(key).replace("_", " ").strip(), str(val).strip()))
    if not parts:
        return

    pdf.add_page()
    _colored_bar_header(
        pdf,
        family,
        has_b,
        w,
        "Contexte du marche",
        "Synthese des specifications (annexe) - perimetre, deroulement et prestations",
        (35, 65, 120),
    )
    pdf.set_font(family, "", 9.5)
    pdf.set_text_color(55, 55, 55)
    pdf.multi_cell(
        w,
        4.5,
        _t(
            "Ce volet n'est pas produit par la correspondance documentaire (RAG) ci-avant : il rappelle "
            "le cadre de l'appel d'offres a partir d'un export des specifications, pour situer l'offre "
            "par rapport aux exigences de la reponse et au projet du maitre d'ouvrage."
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(2)
    pdf.set_text_color(0, 0, 0)
    for sub_title, body in parts:
        pdf.set_font(family, "B" if has_b else "", 10)
        pdf.multi_cell(
            w,
            5,
            _t(sub_title),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_font(family, "", 8.5)
        pdf.multi_cell(
            w,
            3.8,
            _t(body),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.ln(1.5)


def write_match_report_pdf(
    matches_report: list[dict[str, Any]],
    formated_requirements_path: Path,
    output_pdf: Path,
    *,
    requirement_texts: dict[str, str] | None = None,
    requirements_flat: list[dict[str, Any]] | None = None,
    append_specifications_path: Path | None = None,
) -> None:
    """
    matches_report: same structure as rag_match_requirements JSON output.
    requirement_texts: optional id -> full query text (incl. CV requirement: prefix). If omitted,
    only labels from formated JSON are shown and CV-specific notes are skipped.
    requirements_flat: optional list aligned with ``matches_report`` (``category``, ``id``) for
    section headings; if missing, a single list is used without colored blocks per lot.
    append_specifications_path: if set and the file exists (JSON object with e.g. PERIMETRE_DU_PROJET),
    a final appendix page summarizes the project after the RAG part.
    """
    labels = {k: _t(v) for k, v in _load_requirement_labels(formated_requirements_path).items()}
    req_text = {k: _t(v) for k, v in (requirement_texts or {}).items()}
    validity_policy = load_validity_policy(formated_requirements_path)

    font_path, family, font_bold = _windows_fonts()

    pdf = _ReportPDF(
        font_path,
        font_bold,
        family,
    )
    has_b = bool(font_bold)
    pdf.set_margins(14, 32, 14)
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.setup_fonts()

    pdf.add_page()
    w = pdf.epw
    pdf.set_font(family, "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(
        w,
        5.5,
        _t(
            "Ce rapport regroupe, par lot, les propositions de documents (index administratif + CV) "
            "les plus proches de chaque exigence. Les scores sont indicatifs. Les zones colorees en "
            "tete de section aident a parcourir le dossier administratif, l'offre technique et "
            "l'offre financiere."
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_text_color(0, 0, 0)
    if validity_policy:
        pdf.ln(1)
        pdf.set_fill_color(240, 248, 255)
        pdf.set_font(family, "B" if has_b else "", 10)
        pdf.multi_cell(
            w,
            5,
            _t("Fenetre de validite (reference pour comparer les dates d'emission / validite en index)"),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_font(family, "", 9)
        pdf.multi_cell(
            w,
            4.5,
            _t(validity_policy.summary),
            fill=True,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.ln(2)

    n_req = 0
    last_cat: str | None = None
    n_flat = len(requirements_flat or [])

    for idx, block in enumerate(matches_report):
        rid = str(block.get("id", ""))
        category = None
        if requirements_flat and idx < n_flat and isinstance(requirements_flat[idx], dict):
            category = str(requirements_flat[idx].get("category", "") or "").strip() or None
        if category and category != last_cat:
            rgb = _SECT_COLORS.get(category, (25, 55, 109))
            title = _SECT_TITLES.get(category, category)
            _colored_bar_header(
                pdf,
                family,
                has_b,
                w,
                title,
                "Exigences et propositions (resume)",
                rgb,
            )
            last_cat = category

        n_req += 1
        main_label = (labels.get(rid) or "").strip() or _t(rid)
        pdf.set_font(family, "B" if has_b else "", 10.5)
        pdf.multi_cell(
            w,
            5,
            _t(f"Exigence {n_req} - {main_label}"),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_text_color(110, 110, 110)
        pdf.set_font(family, "", 7.5)
        pdf.multi_cell(
            w,
            3.5,
            _t(f"Ref. fiche (comme un nom de fichier) : {rid}"),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(0.5)

        qtext = req_text.get(rid, "")

        if block.get("skipped"):
            reason = str(block.get("skip_reason", "") or "")
            if reason == "date_validation_context":
                msg = _t(
                    "Ligne de politique de dates : sert a definir la fenetre de fraicheur (ex. moins d'un an par "
                    "rapport a l'appel d'offres), pas a retrouver un fichier. Utilisez le cadre \"validite\" "
                    "en tete de rapport et les champs de date de l'index."
                )
            elif reason == "condition_toggles":
                ctags = block.get("condition_tags") or []
                ctags_s = _t(", ".join(str(t) for t in ctags) if ctags else "-")
                msg = _t(f"Non retenu pour le RAG (coche desactivee pour : {ctags_s}).")
            else:
                msg = _t(f"Non retenu ({reason})." if reason else "Non retenu.")
            pdf.set_font(family, "", 9)
            pdf.set_text_color(90, 90, 90)
            pdf.multi_cell(w, 4, msg, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            continue

        matches = block.get("matches") or []
        if not matches:
            pdf.set_font(family, "", 9)
            pdf.set_text_color(100, 50, 50)
            pdf.multi_cell(
                w,
                4,
                "Aucun document propose automatiquement pour cette exigence.",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            continue

        for j, m in enumerate(matches, start=1):
            if not isinstance(m, dict):
                continue
            if m.get("bundled"):
                pdf.set_text_color(0, 100, 55)
                pdf.set_font(family, "B" if has_b else "", 9)
                src = _t(str(m.get("bundle_source") or "autre exigence"))
                pdf.multi_cell(
                    w,
                    4,
                    _t(f"    -> Suggestion regroupee (meme pratique de dossier : {src})"),
                    new_x=XPos.LMARGIN,
                    new_y=YPos.NEXT,
                )
                if m.get("bundle_note"):
                    pdf.set_font(family, "", 8.5)
                    pdf.multi_cell(
                        w,
                        3.5,
                        _t(f"    {m.get('bundle_note')}"),
                        new_x=XPos.LMARGIN,
                        new_y=YPos.NEXT,
                    )
                pdf.set_text_color(0, 0, 0)
            label = _t(str(m.get("match", "")))
            score = m.get("score")
            cscore = m.get("chroma_score")
            dtype = _t(str(m.get("doc_type", "")))
            path = _t(str(m.get("path") or ""))
            line = f"    {j}. {label}  |  {dtype}"
            if path and path != label:
                line += f"\n       Fichier : {path}"
            if score is not None and cscore is not None:
                line += f"\n       Scores  final : {score}  -  vectoriel : {cscore}"
            pdf.set_font(family, "", 9.5)
            pdf.multi_cell(
                w,
                4.2,
                _t(line),
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            for note in _match_notes(rid, qtext, m, validity_policy=validity_policy):
                pdf.set_font(family, "", 8.5)
                pdf.set_text_color(60, 60, 100)
                pdf.multi_cell(
                    w,
                    3.5,
                    _t(f"    - {note}"),
                    new_x=XPos.LMARGIN,
                    new_y=YPos.NEXT,
                )
            pdf.set_text_color(0, 0, 0)
            if m.get("validity") and not m.get("bundled"):
                pdf.set_font(family, "", 8.5)
                pdf.multi_cell(
                    w,
                    3.5,
                    _t(
                        f"    Date (index) : {m.get('validity')} - a comparer a la fenetre de validite en tete de rapport"
                    ),
                    new_x=XPos.LMARGIN,
                    new_y=YPos.NEXT,
                )
            pdf.ln(0.5)
        pdf.ln(2.5)

    _append_specifications_appendix(pdf, family, has_b, w, append_specifications_path)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_pdf))


def main() -> None:
    import argparse

    _st2 = Path(__file__).resolve().parent.parent
    _repo = _st2.parent
    _default_spec = _repo / "step1" / "v2" / "specifications_extracted.json"
    p = argparse.ArgumentParser(description="Generate PDF report from requirements_matches.json")
    p.add_argument("--matches", default=str(_st2 / "requirements_matches.json"))
    p.add_argument(
        "--requirements",
        default=str(_st2 / "formating" / "formated_requirements.json"),
        help="Formated requirements JSON (for human labels)",
    )
    p.add_argument("--output", default=str(_st2 / "requirements_matches_report.pdf"))
    p.add_argument(
        "--project-spec",
        type=Path,
        default=None,
        metavar="JSON",
        help="Specifications JSON (PERIMETRE_..., DEROULEMENT_..., CONSISTANCE_...). "
        f"Default: {_default_spec.as_posix()} if that file exists.",
    )
    p.add_argument(
        "--no-project-spec",
        action="store_true",
        help="Omit the 'Contexte du marche' appendix (no step1/v2 spec summary).",
    )
    args = p.parse_args()

    matches_path = Path(args.matches)
    form_path = Path(args.requirements)
    out_path = Path(args.output)
    if args.no_project_spec:
        spec_path: Path | None = None
    elif args.project_spec is not None:
        spec_path = Path(args.project_spec)
        if not spec_path.is_file():
            raise SystemExit(f"Project specifications file not found: {spec_path}")
    else:
        spec_path = _default_spec if _default_spec.is_file() else None
    if not matches_path.is_file():
        raise SystemExit(f"Matches file not found: {matches_path}")
    if not form_path.is_file():
        raise SystemExit(f"Requirements file not found: {form_path}")

    report = _unwrap_matches_file_payload(
        json.loads(matches_path.read_text(encoding="utf-8"))
    )
    # Rebuild query texts for CV notes (same flattening as RAG)
    req_payload = json.loads(form_path.read_text(encoding="utf-8"))
    flat = flatten_requirements(req_payload)
    req_texts = {r["id"]: r["text"] for r in flat}

    write_match_report_pdf(
        report,
        form_path,
        out_path,
        requirement_texts=req_texts,
        requirements_flat=flat,
        append_specifications_path=spec_path,
    )
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
