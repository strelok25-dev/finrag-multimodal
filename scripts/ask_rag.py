import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import QDRANT_HOST, QDRANT_PORT
from dotenv import load_dotenv
load_dotenv()

try:  # langfuse v3
    from langfuse import observe, get_client
    def flush():
        get_client().flush()
except ImportError:  # langfuse v2
    from langfuse.decorators import observe, langfuse_context
    def flush():
        langfuse_context.flush()

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.generation.llm_client import OllamaClient
from src.generation.validator import NumericalValidator
from src.retrieval.hybrid_search import HybridSearcher

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaClient()
validator = NumericalValidator()
searcher = HybridSearcher(client, model, "finrag_b")


@observe(name="retrieval")
def retrieve(question: str):
    return searcher.search(question, limit=5)


@observe(name="generation")
def generate(question: str, context: str):
    return llm.generate_financial_answer(question, context)


@observe(name="validation")
def validate(llm_response: dict, context_texts: list):
    return validator.validate(llm_response, context_texts)


@observe(name="rag_pipeline")
def answer_question(question: str):
    hits = retrieve(question)
    context_texts = [h["text"] for h in hits]
    context = "\n---\n".join(context_texts)

    llm_response = generate(question, context)
    verdict = validate(llm_response, context_texts)

    return {
        "question": question,
        "final_status": verdict["final_status"],
        "answer": llm_response.get("answer"),
        "verified": verdict["verified"],
        "hallucinated": verdict["hallucinated"],
    }


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "What were the outstandings of home equity loans at December 31, 2023?"
    result = answer_question(question)
    flush()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("✅ Трейс отправлен в LangFuse.")


if __name__ == "__main__":
    main()