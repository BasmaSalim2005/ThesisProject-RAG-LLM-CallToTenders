import os

import json

import sys

from pathlib import Path



# Allow imports from project root

_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:

    sys.path.insert(0, str(_ROOT))



from easy_tender.extract_service import (  # noqa: E402

    DEFAULT_MODEL,

    extract_text_from_pdf_bytes,

    run_extraction,

)



# --- CONFIGURATION ---

FILE_PATH = "step1/Appel-doffre-N30-ANP-2025.pdf"

MODELS_TO_TEST = [

    "anthropic/claude-3-7-sonnet-20250219",

]



with open(FILE_PATH, "rb") as f:

    tender_call_text = extract_text_from_pdf_bytes(f.read())



# --- EXECUTION LOOP ---

for model_name in MODELS_TO_TEST:

    folder_name = model_name.replace("/", "_").replace(":", "_")

    os.makedirs(folder_name, exist_ok=True)



    print(f"\n--- Testing Model: {model_name} ---")



    results, errors = run_extraction(tender_call_text, model_id=model_name)



    for category, result in results.items():

        if result is not None:

            filename = f"{folder_name}/{category}.json"

            with open(filename, "w", encoding="utf-8") as out:

                json.dump(result, out, ensure_ascii=False, indent=2)

            print(f"Saved {filename}")

        else:

            print(f"Skipped {category} (no result)")



    if errors:

        for err in errors:

            print(f"Error: {err}")



print("\nDone! Check the folders for comparison.")


