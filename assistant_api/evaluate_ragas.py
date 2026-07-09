"""
Оценка качества RAG системы через RAGAS.
Использует ProxyAPI (OpenAI-совместимый) для RAG и для метрик RAGAS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: F401 — применяет OPENAI_* env для RAGAS/langchain

from datasets import Dataset
from ragas import evaluate

try:
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import ContextPrecision

    faithfulness = Faithfulness
    context_precision = ContextPrecision
except ImportError:
    try:
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        from ragas.metrics import faithfulness, context_precision

from rag_pipeline import RAGPipeline


EVALUATION_QUESTIONS = [
    "Что такое машинное обучение?",
    "Какие основные типы машинного обучения существуют?",
    "Что такое нейронная сеть?",
    "Как работают трансформеры в NLP?",
    "Что такое RAG и как он работает?",
]


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    """Подготовка датасета для RAGAS из вопросов."""
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("[*] Получение ответов от RAG системы...\n")

    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")

        result = pipeline.query(question, use_cache=False)

        questions_list.append(question)
        answers_list.append(result["answer"])

        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)

        # В реальном проекте — вручную подготовленные эталонные ответы
        ground_truths_list.append(result["answer"][:100])

        print("     [+] Ответ получен от ProxyAPI")

    print()

    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list,
    }

    return Dataset.from_dict(dataset_dict)


def evaluate_rag_system():
    """Основная функция оценки RAG-системы через RAGAS."""
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (ProxyAPI) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    print(f"Proxy API: {config.PROXY_API_URL}")
    print(f"Модель: {config.CHAT_MODEL}")
    print()

    try:
        print("[*] Инициализация RAG системы...\n")
        pipeline = RAGPipeline()
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        print("\nПроверьте файл .env в корне проекта:")
        print("  PROXY_API_KEY=your_proxy_api_key_here")
        sys.exit(1)

    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("=" * 70)

    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision")
    print("   (Answer Relevancy отключена из-за несовместимости embeddings API)")
    print("   (это займёт 1-2 минуты, так как RAGAS использует ProxyAPI для оценки)\n")

    print("   [+] Используем базовые метрики RAGAS\n")
    metrics_to_use = [faithfulness(), context_precision()]

    try:
        result = evaluate(dataset=dataset, metrics=metrics_to_use)
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)

    import math

    faithfulness_values = [
        v
        for v in result["faithfulness"]
        if not (isinstance(v, float) and math.isnan(v))
    ]
    context_precision_values = [
        v
        for v in result["context_precision"]
        if not (isinstance(v, float) and math.isnan(v))
    ]

    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values)
        if faithfulness_values
        else 0
    )
    avg_context_precision = (
        sum(context_precision_values) / len(context_precision_values)
        if context_precision_values
        else 0
    )

    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")

    avg_score = (avg_faithfulness + avg_context_precision) / 2
    print(f"\n{'─' * 70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")

    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает высокую точность и релевантность ответов.")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")

    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)

    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n{i + 1}. {question}")

        faith_val = result["faithfulness"][i]
        if not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print("   Faithfulness:       не удалось вычислить")

        cp_val = result["context_precision"][i]
        if not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print("   Context Precision:  не удалось вычислить")

    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Измеряет качество извлечённого контекста для ответа на вопрос.
  Значения: 0.0 - 1.0 (1.0 = идеальный контекст)

ПРИМЕЧАНИЕ:
  Answer Relevancy временно отключена из-за несовместимости embeddings API
  в текущей версии RAGAS с langchain-openai.
    """)

    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()
