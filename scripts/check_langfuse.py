import time
from dotenv import load_dotenv

load_dotenv()  # читаем .env до инициализации langfuse

try:  # langfuse v3
    from langfuse import observe, get_client
    def flush():
        get_client().flush()
except ImportError:  # langfuse v2
    from langfuse.decorators import observe
    from langfuse import Langfuse
    def flush():
        Langfuse().flush()


@observe
def fake_retrieval():
    time.sleep(0.3)
    return {"chunks_found": 5}


@observe
def fake_generation():
    time.sleep(0.5)
    return {"answer": "connectivity test"}


@observe
def main():
    retrieval = fake_retrieval()
    generation = fake_generation()
    return {"status": "ok", **retrieval, **generation}


if __name__ == "__main__":
    main()
    flush()
    print("✅ Трейс отправлен. Открой LangFuse UI → Traces и найди его.")