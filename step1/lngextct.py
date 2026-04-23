import os
import json
import textwrap
from pathlib import Path
import langextract as lx
from dotenv import load_dotenv

import fitz

load_dotenv(Path(__file__).resolve().parent / ".env")

from langextract.providers.openai import OpenAILanguageModel

# 1. Setup Groq Configuration (set GROQ_API_KEY in .env — never commit keys)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in the environment or in step1/.env .")
# Example Groq models: "llama-3.3-70b-versatile" or "llama3-8b-8192"
MODEL_ID = "llama-3.3-70b-versatile"

# 2. Instantiate the Groq model using the OpenAI provider class
groq_model = OpenAILanguageModel(
    model_id=MODEL_ID,
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    temperature=0.1,
)

# 1. CLEAN TEXT (Fixes encoding/character issues)
def get_pdf_text(path):
    doc = fitz.open(path)
    # Using 'utf-8' is vital for French tenders to avoid the '' error
    text = "\n".join([page.get_text() for page in doc])
    return text.encode("utf-8", "ignore").decode("utf-8")


def normalize_extraction_text(value):
    """Ensure extraction_text is string/int/float to satisfy langextract validation."""
    if isinstance(value, (str, int, float)):
        return value
    if value is None:
        return ""
    return json.dumps(value)  # Convert dict/list to JSON string


full_text = get_pdf_text("Appel-doffre-N30-ANP-2025.pdf")

# Save the extracted PDF text to a file for inspection
with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

# 2. EXACT MATCH EXAMPLES
# Ensure extraction_text appears exactly in text
examples = [
    lx.data.ExampleData(
        text="Le candidat doit fournir un cautionnement provisoire de 50.000 DH.",
        extractions=[
            lx.data.Extraction(
                extraction_class="FIN",
                extraction_text="cautionnement provisoire de 50.000 DH"
            )
        ]
    )
]

prompt=textwrap.dedent("""\
                       À partir du document d’appel d’offres fourni, extrayez LES EXPLOITATIONS suivantes en français:

1) Ne retournez que le JSON ci-dessous (aucune explication, aucun texte libre en dehors du JSON).
2) Le format attendu doit être EXACTEMENT:
{
  "extractions": [
    {
      "extraction_class": "DOSSIER_ADMINISTRATIF|OFFRE_TECHNIC|OFFRE_FINANCIERE",
      "extraction_text": "texte exact à partir du PDF (chaîne)"
    }
  ]
}

3) Ne pas extraire les titres, numéros d'articles, en-têtes, noms d'organismes sauf si ils font partie d'une exigence administrative ou technique réelle.
4) N'incluez pas de liste dans extraction_text, c'est une chaîne unique.
5) Si la valeur n’est pas trouvée, ne l’ajoutez pas.
6) Concentrez-vous sur:
   - documents administratifs obligatoires (CNSS, RC, IJ, etc.)
   - documents techniques obligatoires (CV, références, organigramme, etc.)
   - pièces financières principales (caution provisoire, prix, bordereau)
   - conditions d’admission (validité, délais, attestation de situation fiscale, etc.)

Répondez STRICTEMENT au format JSON ci-dessus.
""" )
# 3. RUN EXTRACTION
try:
    result = lx.extract(
        text_or_documents=full_text,
        prompt_description=prompt,
        examples=examples,
        model=groq_model,
        fence_output=True,
        # use_schema_constraints=True
    )

    print(f"Extractions found: {len(result.extractions)}")

    # Save to file
    with open("thesis_extraction.json", "w", encoding="utf-8") as f:
        data_to_save = {}
        for e in result.extractions:
            cls = e.extraction_class
            text = normalize_extraction_text(getattr(e, "extraction_text", None))
            if cls not in data_to_save:
                data_to_save[cls] = []
            if text not in data_to_save[cls]:
                data_to_save[cls].append(text)
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print("Saved to thesis_extraction.json")
        
except Exception as e:
    import traceback
    print(f"Extraction failed: {e}")
    traceback.print_exc()