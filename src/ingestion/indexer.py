import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, HnswConfigDiff, PointStruct

from src.core.config import QDRANT_HOST, QDRANT_PORT
from src.ingestion.embedder import Embedder
from src.ingestion.chunking import strategy_a_text_only, strategy_b_docling_hybrid, strategy_c_table_aware

def index_document(json_path: str, strategy: str = "B"):
    print(f"⚙️ Загрузка артефакта: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    
    elements = artifact.get("elements", [])
    table_groups = artifact.get("table_groups", [])

    print(f"⚙️ Применение стратегии чанкинга: {strategy}")
    if strategy == "A":
        chunks = strategy_a_text_only(elements)
    elif strategy == "B":
        chunks = strategy_b_docling_hybrid(elements)
    elif strategy == "C":
        chunks = strategy_c_table_aware(elements, table_groups)
    else:
        raise ValueError("Неизвестная стратегия. Используйте 'A', 'B' или 'C'.")

    print(f"✅ Сформировано чанков: {len(chunks)}")

    # Инициализация Qdrant и Embedder
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedder = Embedder()

    collection_name = f"finrag_{strategy.lower()}"
    
    # Создание коллекции с HNSW для быстрого поиска
    try:
        client.get_collection(collection_name)
        print(f"️ Коллекция {collection_name} уже существует. Данные будут обновлены (upsert).")
    except:
        print(f"📦 Создание коллекции: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10000)
        )
    # Векторизация и загрузка
    print("🧠 Векторизация текстов (это может занять минуту)...")
    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.encode(texts)

    points = [
        PointStruct(
            id=chunk["id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                **chunk["metadata"]
            }
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    print(f"🚀 Загрузка точек в Qdrant (батчами по 256)...")
    batch_size = 256
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=True)
    print(f"✅ Успешно загружено {len(points)} точек в коллекцию '{collection_name}'")

if __name__ == "__main__":
    # Для начала прогоним стратегию B как базовую
    index_document("data/processed/DOC_01_V_artifact.json", strategy="C")