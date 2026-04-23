import anthropic
import base64
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

# The stable 2026 model
MODEL_ID = "claude-sonnet-4-6"
FILE_PATH = "step1/Appel-doffre-N30-ANP-2025.pdf"

def extract_all_at_once(file_path):
    with open(file_path, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

    # This prompt tells Claude exactly how to structure the "Master JSON"
    master_prompt = """
    Analyze the attached Moroccan tender PDF and extract three distinct sections.
    Return the result ONLY as a single valid JSON object with exactly these three keys:
    
    "requirements": From the tender document below, extract ONLY the submission document requirements and the publication date of the announcement which is the refference for validiy date of the documents (as a checklist of documents to use for a matching with my db documents) include special cases. Return JSON: { "dossier_administratif": [], "offre_technique": [], "offre_financiere": {} } \n\nDocument:  ,
    "specifications": From the tender document below, extract ONLY project specifications. Return JSON: { "PERIMETRE_DU_PROJET": "", "DEROULEMENT_DU_PROJET": "", "CONSISTANCE_DES_PRESTATIONS": "" } \n\nDocument: ,
    "evaluation": From the tender document below, extract ONLY evaluation criteria. Return JSON: { "EVALUATION_DES_OFFRES_DES_CONCURRENTS": { "evaluation_technique": [], "evaluation_financiere": [] } } \n\nDocument: 
    
    Do not include any conversational text or markdown explanation.
    """

    try:
        print("Sending PDF for full analysis (One-shot)...")
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=8192, # Increased to ensure enough room for all 3 sections
            system="You are a Moroccan tender expert. Respond with valid JSON only.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64,
                            },
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": master_prompt
                        }
                    ],
                }
            ],
        )

        # Parse the master response
        raw_text = response.content[0].text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        
        master_data = json.loads(raw_text)

        # --- SEPARATION LOGIC ---
        # Now we save them into 3 different files
        for key in ["requirements", "specifications", "evaluation"]:
            filename = f"{key}_extracted.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(master_data.get(key, {}), f, ensure_ascii=False, indent=2)
            print(f"Successfully saved: {filename}")

        return True

    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

# Run the extraction
extract_all_at_once(FILE_PATH)