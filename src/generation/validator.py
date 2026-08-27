import re
from typing import List, Optional


class NumericalValidator:
    """
    Независимый валидатор числовых утверждений LLM.
    Сверяет каждое число из ответа с числами, извлеченными из контекста.
    Не использует LLM. Только детерминированный код.
    """

    @staticmethod
    def normalize_number(raw: str) -> Optional[float]:
        """'(1,250.5)' -> -1250.5, '$ 25,527' -> 25527.0"""
        raw = str(raw).strip()
        if not raw:
            return None

        negative = False
        if raw.startswith("(") and raw.endswith(")"):
            negative = True
            raw = raw[1:-1]

        # Вырезаем всё, кроме цифр, точки и минуса
        cleaned = re.sub(r"[^0-9.\-]", "", raw)
        if not cleaned or cleaned in {".", "-", "-."}:
            return None

        try:
            value = float(cleaned)
        except ValueError:
            return None

        return -value if negative else value

    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Извлекает все числа из текста контекста, включая (125) как отрицательные."""
        numbers = []
        # Сначала скобочные отрицательные: (125), (1,250.5)
        for match in re.finditer(r"\((\d[\d,]*(?:\.\d+)?)\)", text):
            v = NumericalValidator.normalize_number(match.group(0))
            if v is not None:
                numbers.append(v)
        # Затем обычные: $ 25,527 / -72958812 / 33.88
        for match in re.finditer(r"-?\$?\s?\d[\d,]*(?:\.\d+)?", text):
            v = NumericalValidator.normalize_number(match.group(0))
            if v is not None:
                numbers.append(v)
        return numbers

    def validate(self, llm_response: dict, context_texts: List[str]) -> dict:
        """
        Сверяет числа из ответа LLM с числами из контекста.
        Возвращает финальный вердикт.
        """
        llm_status = llm_response.get("status", "not_found")

        # LLM сама отказалась отвечать — уважаем отказ
        if llm_status == "not_found":
            return {
                "final_status": "not_found",
                "llm_status": llm_status,
                "claimed": [],
                "verified": [],
                "hallucinated": [],
                "details": "LLM корректно отказалась отвечать (данных нет в контексте).",
            }

        claimed = llm_response.get("numbers_claimed", [])

        # Собираем все числа, которые РЕАЛЬНО есть в контексте
        context_numbers = set()
        for text in context_texts:
            context_numbers.update(self.extract_numbers(text))

        verified = []
        hallucinated = []
        for claim in claimed:
            v = self.normalize_number(claim)
            if v is None:
                continue
            if any(abs(v - c) < 1e-6 for c in context_numbers):
                verified.append(claim)
            else:
                hallucinated.append(claim)

        if hallucinated:
            final_status = "validation_failed"  # LLM выдумала цифру — ловим за руку
            details = f"Обнаружены числа, отсутствующие в источнике: {hallucinated}"
        elif not claimed:
            final_status = llm_status  # Текстовый ответ без цифр — доверяем статусу LLM
            details = "Числа не заявлялись, текстовый ответ."
        else:
            final_status = "verified"  # Все числа подтверждены источником
            details = "Все заявленные числа найдены в контексте."

        return {
            "final_status": final_status,
            "llm_status": llm_status,
            "claimed": claimed,
            "verified": verified,
            "hallucinated": hallucinated,
            "details": details,
        }