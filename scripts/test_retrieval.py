from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)

print("⏳ Загрузка модели эмбеддингов...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

QUESTIONS = [
    "What is the 2024 target allocation?",
    "What are the valuation techniques for Level 3 Fair Value Measurements?",
    "What is the ranking of the ICONs?"
]

COLLECTIONS = ["finrag_a", "finrag_b", "finrag_c"]

print("=" * 80)
print("🔍 НАЧАЛО СРАВНИТЕЛЬНОГО ТЕСТА RETRIEVAL")
print("=" * 80)

for q_idx, question in enumerate(QUESTIONS, 1):
    print(f"\n[ВОПРОС {q_idx}]: {question}")
    print("-" * 80)
    vector = model.encode(question).tolist()

    for collection in COLLECTIONS:
        try:
            hits = client.search(
                collection_name=collection,
                query_vector=vector,
                limit=1
            )
            if hits:
                hit = hits[0]
                score = hit.score
                text = hit.payload.get("text", "NO TEXT")
                chunk_type = hit.payload.get("chunk_type", "N/A")
                strategy = hit.payload.get("strategy", "N/A")

                print(f"✅ Стратегия {strategy} (Коллекция: {collection})")
                print(f"   Score: {score:.4f} | Type: {chunk_type}")
                print(f"   Text: {text[:200]}...")
            else:
                print(f"❌ Коллекция {collection}: Ничего не найдено.")
        except Exception as e:
            print(f"⚠️ Ошибка в коллекции {collection}: {e}")
    print("=" * 80)