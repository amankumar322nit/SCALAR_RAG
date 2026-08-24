#!/usr/bin/env python3
"""
Scaler Learner Support RAG CLI & Service Entrypoint.
Thin executable wrapper delegating to backend/src/cli.py.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if BACKEND_DIR.exists():
    sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))

from src.cli import main

if __name__ == "__main__":
    main()
