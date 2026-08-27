from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"⏳ Загрузка модели эмбеддингов: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Модель загружена. Размерность вектора: {self.dim}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        # convert_to_numpy=True ускоряет работу, затем конвертируем в список для Qdrant
        return self.model.encode(texts, convert_to_numpy=True).tolist()