"""
Основной RAG pipeline.
Управляет потоком: запрос -> кеш -> vector search -> LLM (ProxyAPI) -> ответ -> кеш.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from vector_store import VectorStore
from cache import RAGCache


class RAGPipeline:
    """Основной pipeline для RAG системы."""

    def __init__(
        self,
        collection_name: str = None,
        cache_db_path: str = None,
        data_file: str = None,
        model: str = None,
    ):
        """
        Инициализация RAG pipeline.

        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_file: путь к файлу с документами
            model: модель для генерации ответов
        """
        self.model = model or config.CHAT_MODEL
        self.openai_client = OpenAI(**config.OPENAI_CLIENT_KWARGS)

        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(collection_name=collection_name)

        data_path = Path(data_file) if data_file else config.DATA_FILE
        if self.vector_store.collection.count() == 0:
            print(f"Загрузка документов из {data_path}...")
            self.vector_store.load_documents(str(data_path))

        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=str(cache_db_path or config.CACHE_DB_PATH))

        print(f"RAG Pipeline инициализирован (ProxyAPI: {config.PROXY_API_URL})")

    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Создание промпта для LLM с контекстом."""
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"Документ {i}:\n{doc['text']}\n")

        context = "\n".join(context_parts)
        return config.USER_PROMPT_TEMPLATE.format(context=context, query=query)

    def _generate_answer(self, prompt: str) -> str:
        """Генерация ответа через ProxyAPI (OpenAI-совместимый API)."""
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=config.ANSWER_TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Основной метод для обработки запроса пользователя.

        Поток:
        1. Проверка кеша
        2. Если в кеше нет - поиск в векторном хранилище
        3. Формирование промпта с контекстом
        4. Генерация ответа через LLM API
        5. Сохранение в кеш
        """
        print(f"\n{'=' * 60}")
        print(f"Запрос: {user_query}")
        print(f"{'=' * 60}")

        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(user_query)

            if cached_result:
                print("[+] Ответ найден в кеше")
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": cached_result.get("context"),
                    "cached_at": cached_result.get("created_at"),
                }
            print("[-] Ответ не найден в кеше")

        print("[*] Поиск релевантных документов через ProxyAPI...")
        context_docs = self.vector_store.search(user_query, top_k=config.TOP_K)
        print(f"[+] Найдено {len(context_docs)} релевантных документов")

        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs)

        print(f"[*] Генерация ответа через ProxyAPI ({self.model})...")
        answer = self._generate_answer(prompt)
        print("[+] Ответ получен от API")

        if use_cache:
            print("[*] Сохранение в кеш...")
            context_for_cache = [doc["text"] for doc in context_docs]
            self.cache.set(user_query, answer, context_for_cache)
            print("[+] Сохранено в кеш")

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "ProxyAPI",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики системы."""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "ProxyAPI",
            "proxy_api_url": config.PROXY_API_URL,
        }


if __name__ == "__main__":
    try:
        pipeline = RAGPipeline()

        test_queries = [
            "Что такое машинное обучение?",
            "Что такое RAG?",
            "Как работают трансформеры?",
        ]

        for query in test_queries:
            result = pipeline.query(query)
            print(f"\n{'=' * 60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print(f"{'=' * 60}\n")

        print("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        print(f"Из кеша: {result['from_cache']}")

        stats = pipeline.get_stats()
        print(f"\nСтатистика системы:")
        print(f"Векторное хранилище: {stats['vector_store']}")
        print(f"Кеш: {stats['cache']}")
        print(f"Режим: {stats['mode']}")
        print(f"Proxy API: {stats['proxy_api_url']}")

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
