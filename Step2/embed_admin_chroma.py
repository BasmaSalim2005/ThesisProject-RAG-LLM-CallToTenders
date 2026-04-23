"""Shim: run ``python -m Step2.embedding.embed_admin_chroma`` from the repo root (preferred)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    from Step2.embedding import embed_admin_chroma as m

    m.main()
