"""
Единая конфигурация проекта RAG-ассистента.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(key: str, default: Path) -> Path:
    value = os.getenv(key)
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


# Настройки Proxy API
PROXY_API_URL = _env("PROXY_API_URL", "https://api.proxyapi.ru/openai/v1").rstrip("/")
PROXY_API_KEY = _env("PROXY_API_KEY") or _env("OPENAI_API_KEY")
if not PROXY_API_KEY:
    raise ValueError("PROXY_API_KEY не установлен в .env")

REQUEST_TIMEOUT = _env_int("REQUEST_TIMEOUT", 60)

# Совместимость с библиотеками, которые ожидают OpenAI env vars
OPENAI_API_KEY = PROXY_API_KEY
OPENAI_BASE_URL = PROXY_API_URL


def apply_proxy_env() -> None:
    os.environ["OPENAI_API_KEY"] = PROXY_API_KEY
    os.environ["OPENAI_BASE_URL"] = PROXY_API_URL
    os.environ["OPENAI_API_BASE"] = PROXY_API_URL


apply_proxy_env()

# Модели OpenAI-совместимого Proxy API
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = _env("CHAT_MODEL", "gpt-4o-mini")

# Параметры клиентов OpenAI SDK
OPENAI_CLIENT_KWARGS = {
    "api_key": PROXY_API_KEY,
    "base_url": PROXY_API_URL,
    "timeout": REQUEST_TIMEOUT,
}

# Пути (относительно assistant_api при запуске из этой папки)
ASSISTANT_DIR = BASE_DIR / "assistant_api"
CHROMA_DB_PATH = _env_path("CHROMA_DB_PATH", ASSISTANT_DIR / "chroma_db")
CACHE_DB_PATH = _env_path("CACHE_DB_PATH", ASSISTANT_DIR / "api_rag_cache.db")
DATA_FILE = _env_path("DATA_FILE", ASSISTANT_DIR / "data" / "docs.txt")

# Настройки ChromaDB
CHROMA_SETTINGS = {"anonymized_telemetry": _env_bool("CHROMA_TELEMETRY", False)}
COLLECTION_NAME = _env("COLLECTION_NAME", "api_rag_collection")
COLLECTION_METADATA = {"description": "Документы для RAG-ассистента", "hnsw:space": "cosine"}

# Параметры чанкинга
CHUNK_SIZE = _env_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 100)

# Параметры поиска и генерации
TOP_K = _env_int("TOP_K", 3)
MAX_TOKENS = _env_int("MAX_TOKENS", 500)
ANSWER_TEMPERATURE = _env_float("ANSWER_TEMPERATURE", 0.3)

_DEFAULT_SYSTEM_PROMPT = (
    "Ты - полезный AI ассистент, который отвечает на вопросы "
    "на основе предоставленного контекста."
)
SYSTEM_PROMPT = _env("SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

_DEFAULT_USER_PROMPT_TEMPLATE = """Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Если в контексте нет информации для ответа, скажи об этом
- Будь точным и кратким
- Отвечай на русском языке

Ответ:"""
USER_PROMPT_TEMPLATE = _env("USER_PROMPT_TEMPLATE", _DEFAULT_USER_PROMPT_TEMPLATE)

# CLI-настройки
EXIT_COMMANDS = ("exit", "quit", "выход", "q")
