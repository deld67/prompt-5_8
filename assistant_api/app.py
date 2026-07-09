"""
Консольное приложение для взаимодействия с RAG ассистентом (ProxyAPI).
"""

import sys
from pathlib import Path

# UTF-8 вывод в Windows-консоли (избегаем UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from rag_pipeline import RAGPipeline


def print_banner():
    """Вывод приветственного баннера."""
    banner = """
============================================================
         RAG Ассистент (ProxyAPI)
  Retrieval-Augmented Generation через ProxyAPI
============================================================
    """
    print(banner)
    print(f"Proxy API: {config.PROXY_API_URL}")
    print(f"Модель: {config.CHAT_MODEL}")
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' для просмотра статистики")
    print("Введите 'clear' для очистки кеша\n")


def print_response(result: dict):
    """Форматированный вывод ответа."""
    print(f"\n{'-' * 60}")
    print(f"Вопрос: {result['query']}")
    print(f"{'-' * 60}")

    if result["from_cache"]:
        print("Источник: КЕШ")
        if "cached_at" in result:
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"Источник: ProxyAPI ({result.get('model', 'LLM')})")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")

    print(f"\nОтвет:\n{result['answer']}")

    if not result["from_cache"] and result.get("context_docs"):
        print(f"\nИспользованный контекст:")
        for i, doc in enumerate(result["context_docs"][:2], 1):
            preview = (
                doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
            )
            print(f"   {i}. {preview}")

    print(f"{'-' * 60}\n")


def print_stats(pipeline: RAGPipeline):
    """Вывод статистики системы."""
    stats = pipeline.get_stats()

    print(f"\n{'=' * 60}")
    print("СТАТИСТИКА СИСТЕМЫ")
    print(f"{'=' * 60}")

    print("\nВекторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")

    print("\nКеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats["cache"]["oldest_entry"]:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats["cache"]["newest_entry"]:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")

    print(f"\nМодель: {stats['model']}")
    print(f"Режим: {stats['mode']}")
    print(f"Proxy API: {stats.get('proxy_api_url', config.PROXY_API_URL)}")
    print(f"{'=' * 60}\n")


def main():
    """Главная функция приложения."""
    print_banner()

    try:
        print("Инициализация системы...\n")
        pipeline = RAGPipeline()
        print("\nСистема готова к работе!\n")

    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        print("\nПроверьте файл .env в корне проекта:")
        print("  PROXY_API_KEY=your_proxy_api_key_here")
        sys.exit(1)

    while True:
        try:
            user_input = input("Ваш вопрос: ").strip()

            if user_input.lower() in config.EXIT_COMMANDS:
                print("\nДо свидания!")
                break

            if user_input.lower() == "stats":
                print_stats(pipeline)
                continue

            if user_input.lower() == "clear":
                confirm = input("Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ["yes", "y", "да"]:
                    pipeline.cache.clear()
                    print("Кеш очищен")
                continue

            if not user_input:
                print("Пожалуйста, введите вопрос\n")
                continue

            result = pipeline.query(user_input)
            print_response(result)

        except KeyboardInterrupt:
            print("\n\nПрервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")


if __name__ == "__main__":
    main()
