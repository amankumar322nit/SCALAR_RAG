#!/usr/bin/env python3
"""
Scaler Learner Support RAG CLI & Service Entrypoint.
Thin executable wrapper delegating to src/cli.py.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))

from src.cli import main

if __name__ == "__main__":
    main()
