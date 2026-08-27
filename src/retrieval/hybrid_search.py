import re
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi


class HybridSearcher:
    """
    Гибридный поиск: Dense (семантика) + BM25 (точные ключевые слова).
    Фьюжн через Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, client, model, collection: str, max_points: int = 10000):
        self.client = client
        self.model = model
        self.collection = collection

        print(f"📥 Выгрузка точек из {collection} для BM25-индекса...")
        points, _ = client.scroll(
            collection_name=collection,
            limit=max_points,
            with_payload=True,
            with_vectors=False,
        )
        self.ids = [p.id for p in points]
        self.texts = [p.payload.get("text", "") for p in points]
        self.payloads = [p.payload for p in points]
        self.idx = {pid: i for i, pid in enumerate(self.ids)}

        self.bm25 = BM25Okapi([self._tokenize(t) for t in self.texts])
        print(f"✅ BM25-индекс построен по {len(self.ids)} точкам.")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, question: str, limit: int = 5, deep: int = 20, k_rrf: int = 60) -> List[Dict[str, Any]]:
        # 1. Dense-поиск (семантика)
        vector = self.model.encode(question).tolist()
        dense_hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=deep,
        )

        # 2. BM25-поиск (точные слова: "outstandings", "home", "equity")
        bm25_scores = self.bm25.get_scores(self._tokenize(question))
        bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:deep]

        # 3. RRF-фьюжн: суммируем 1/(k + rank) по обеим выдачам
        rrf: Dict[str, float] = {}
        for rank, hit in enumerate(dense_hits):
            rrf[hit.id] = rrf.get(hit.id, 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, i in enumerate(bm25_order):
            pid = self.ids[i]
            rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (k_rrf + rank + 1)

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:limit]

        return [
            {
                "id": pid,
                "rrf_score": score,
                "text": self.texts[self.idx[pid]],
                "payload": self.payloads[self.idx[pid]],
            }
            for pid, score in fused
        ]