"""
Модуль работы с векторным хранилищем ChromaDB.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import os
import re

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Конфиг проекта в корне репозитория
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
    ):
        """
        Инициализация векторного хранилища.

        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.persist_directory = str(persist_directory or config.CHROMA_DB_PATH)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(**config.CHROMA_SETTINGS),
        )

        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(
                f"Коллекция '{self.collection_name}' загружена. "
                f"Документов: {self.collection.count()}"
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata=config.COLLECTION_METADATA,
            )
            print(f"Создана новая коллекция '{self.collection_name}'")

        self.openai_client = OpenAI(**config.OPENAI_CLIENT_KWARGS)
        self.embedding_model = config.EMBEDDING_MODEL

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None,
    ) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.

        Стратегия:
        1. Приоритет абзацам (разделение по \\n\\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        """
        chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
        overlap = overlap if overlap is not None else config.CHUNK_OVERLAP

        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

            elif current_chunk:
                chunks.append(current_chunk)
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = (
                    overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                )

            else:
                if len(paragraph) > chunk_size:
                    sentence_chunks = self._split_long_paragraph(
                        paragraph, chunk_size, overlap
                    )
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        return chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Получение текста для overlap из конца предыдущего чанка."""
        if len(text) <= overlap_size:
            return text

        overlap_candidate = text[-overlap_size:]
        sentence_starts = [". ", "! ", "? ", "\n"]
        best_start = 0

        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)

        if best_start > 0:
            return overlap_candidate[best_start:].strip()

        return overlap_candidate.strip()

    def _split_long_paragraph(
        self, paragraph: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """Разбиение длинного абзаца на чанки по предложениям."""
        sentences = re.split(r"([.!?]+\s+)", paragraph)

        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])

        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])

        chunks = []
        current_chunk = ""

        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = (
                        overlap_text + " " + sentence if overlap_text else sentence
                    )
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def load_documents(self, file_path: str = None):
        """Загрузка документов из файла в векторное хранилище."""
        file_path = str(file_path or config.DATA_FILE)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = self._chunk_text(text)
        print(f"Текст разбит на {len(chunks)} чанков")

        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return

        documents = []
        ids = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            embedding = self._create_embedding(chunk)
            documents.append(chunk)
            ids.append(f"doc_{i}")
            embeddings.append(embedding)

            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(chunks)} чанков")

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
        )

        print(
            f"Загружено {len(chunks)} документов в коллекцию '{self.collection_name}'"
        )

    def _create_embedding(self, text: str) -> List[float]:
        """Создание векторного представления текста через ProxyAPI (OpenAI)."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model,
            encoding_format="float",
        )
        return response.data[0].embedding

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Поиск релевантных документов по запросу."""
        top_k = top_k if top_k is not None else config.TOP_K
        query_embedding = self._create_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = []
        if results["documents"] and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                documents.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "distance": (
                            results["distances"][0][i]
                            if "distances" in results
                            else None
                        ),
                    }
                )

        return documents

    def get_collection_stats(self) -> Dict[str, Any]:
        """Получение статистики коллекции."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory,
        }


if __name__ == "__main__":
    vector_store = VectorStore()

    if config.DATA_FILE.exists():
        vector_store.load_documents()

    results = vector_store.search("Что такое машинное обучение?")
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")

    print(f"\nСтатистика: {vector_store.get_collection_stats()}")
