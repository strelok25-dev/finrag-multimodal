from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
print("✅ Qdrant подключен!")
print("Коллекции:", client.get_collections())