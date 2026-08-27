import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import QDRANT_HOST, QDRANT_PORT
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.generation.llm_client import OllamaClient
from src.generation.validator import NumericalValidator
from src.retrieval.hybrid_search import HybridSearcher

def run_evaluation(dataset_path: str, collection: str = "finrag_b"):
    print(f"📂 Загрузка датасета: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = OllamaClient()
    validator = NumericalValidator()
    searcher = HybridSearcher(client, model, collection)

    total_questions = len(dataset)
    correct_answers = 0

    print("\n" + "="*80)
    print(f"🚀 ЗАПУСК ОЦЕНКИ НА {total_questions} ВОПРОСАХ (Коллекция: {collection})")
    print("="*80 + "\n")

    for i, item in enumerate(dataset, 1):
        q_id = item["id"]
        question = item["question"]
        expected_status = item["expected_status"]
        expected_number = item.get("expected_number")

        print(f"[{i}/{total_questions}] ({q_id}) ❓ {question}")
        
        # 1. Retrieval
        hits = searcher.search(question, limit=5)
        context_texts = [h["text"] for h in hits]
        context = "\n---\n".join(context_texts)

        # 2. Generation
        llm_response = llm.generate_financial_answer(question, context)
        
        # 3. Validation
        verdict = validator.validate(llm_response, context_texts)
        final_status = verdict["final_status"]

        # 4. Evaluation Logic
        is_correct = False
        if final_status == expected_status:
            if expected_number is not None:
                # Проверяем, есть ли ожидаемое число в подтвержденных
                if any(expected_number in str(v) for v in verdict["verified"]):
                    is_correct = True
            else:
                is_correct = True

        if is_correct:
            correct_answers += 1
            print(f"   ✅ УСПЕХ | Ожидание: {expected_status} | Факт: {final_status}")
        else:
            print(f"   ❌ ПРОВАЛ | Ожидание: {expected_status} | Факт: {final_status}")
            print(f"      Ответ LLM: {llm_response.get('answer', 'N/A')}")
            
        print("-" * 80)

    accuracy = (correct_answers / total_questions) * 100
    print("\n" + "="*80)
    print(f"🏁 ИТОГОВАЯ ОЦЕНКА (ACCURACY): {accuracy:.1f}% ({correct_answers}/{total_questions})")
    print("="*80)

if __name__ == "__main__":
    run_evaluation("data/eval_dataset.json", collection="finrag_b")