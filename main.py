"""
Точка входа RAG-ассистента из корня проекта.

Запуск:
    python main.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSISTANT_DIR = ROOT / "assistant_api"

# UTF-8 вывод в Windows-консоли
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Корень — для config.py; assistant_api — для модулей ассистента
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ASSISTANT_DIR))

from app import main  # noqa: E402


if __name__ == "__main__":
    main()
