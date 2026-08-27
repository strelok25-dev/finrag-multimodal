import json
import requests
from src.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.url = f"{base_url}/api/generate"

        try:
            requests.get(f"{base_url}/api/tags", timeout=5)
            print(f"✅ Ollama сервер доступен. Используем модель: {self.model}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("❌ Ollama сервер не запущен! Запусти 'ollama serve' или приложение Ollama.")

    def generate_financial_answer(self, question: str, context: str) -> dict:
        system_prompt = """Ты — строгий финансовый аналитик. Твоя задача — отвечать на вопросы ТОЛЬКО на основе предоставленного контекста.
Если ответа нет в контексте, ты ДОЛЖЕН вернуть статус "not_found".
Запрещено выдумывать цифры. Запрещено использовать знания вне контекста.

Ты ДОЛЖЕН ответить ИСКЛЮЧИТЕЛЬНО в формате валидного JSON без каких-либо пояснений, markdown-блоков (```json) и лишнего текста.
Схема JSON:
{
  "status": "verified" | "not_found",
  "answer": "Краткий текстовый ответ",
  "evidence": ["Цитата из контекста 1", "Цитата из контекста 2"],
  "numbers_claimed": ["Список всех чисел и финансовых показателей, которые ты назвал в ответе"]
}"""

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": f"Context:\n{context}\n\nQuestion: {question}",
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 512
            }
        }

        print("⏳ Отправка запроса в Ollama (на CPU генерация может занять 10-60 секунд)...")
        response = requests.post(self.url, json=payload, timeout=300)

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")

        raw_response = response.json().get("response", "")

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            print(f"❌ ОШИБКА ПАРСИНГА JSON! Модель вернула мусор:\n{raw_response}")
            raise