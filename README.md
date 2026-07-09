# RAG-ассистент (ProxyAPI / OpenAI)

Готовый RAG-ассистент: поиск по базе знаний (ChromaDB) + генерация ответа через OpenAI-совместимый ProxyAPI.

## Возможности

- Retrieval-Augmented Generation (поиск релевантных фрагментов + ответ LLM)
- Embeddings и chat через [ProxyAPI](https://proxyapi.ru) (`gpt-4o-mini`, `text-embedding-3-small`)
- Векторное хранилище ChromaDB
- Кеш ответов (SQLite)
- Оценка качества через RAGAS (опционально)

## Быстрый старт

### 1. Окружение

Рекомендуется Python 3.11+ (проект также запускался на 3.14).

```powershell
py -3.11 -m venv venv_py311
.\venv_py311\Scripts\activate
pip install -r .\requirements.txt
```

### 2. Настройки

Скопируйте шаблон и укажите ключ:

```powershell
copy env.example .env
```

В `.env` задайте:

```
PROXY_API_KEY=your_proxy_api_key_here
```

Файл `.env` не коммитится в Git (см. `.gitignore`).

### 3. Запуск

Из корня проекта:

```powershell
python main.py
```

Альтернатива:

```powershell
cd assistant_api
python app.py
```

Команды в консоли: `stats`, `clear`, `exit` / `quit`.

### 4. Оценка RAGAS (опционально)

```powershell
cd assistant_api
python evaluate_ragas.py
```

## Структура

```
prompt-5_8/
├── main.py                 # запуск из корня
├── config.py               # все настройки проекта
├── env.example             # шаблон переменных окружения
├── requirements.txt
├── README.md
└── assistant_api/
    ├── app.py              # консольный UI
    ├── rag_pipeline.py     # RAG: кеш → поиск → LLM
    ├── vector_store.py     # ChromaDB + embeddings
    ├── cache.py            # SQLite-кеш
    ├── evaluate_ragas.py   # метрики качества
    └── data/docs.txt       # база знаний
```

## Настройки

Основные параметры — в `config.py` и `.env` (см. `env.example`):

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `PROXY_API_URL` | Endpoint ProxyAPI | `https://api.proxyapi.ru/openai/v1` |
| `PROXY_API_KEY` | API-ключ | — |
| `CHAT_MODEL` | Модель генерации | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Модель embeddings | `text-embedding-3-small` |
| `TOP_K` | Число документов в контексте | `3` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Чанкинг | `500` / `100` |
