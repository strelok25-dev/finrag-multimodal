import sys
from pathlib import Path

# Страховка: чтобы скрипт находил пакет src из любой директории
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pprint
from src.generation.llm_client import OllamaClient

# Реальный кусок таблицы из твоего 10-K (Home Equity)
MOCK_CONTEXT = """
(Dollars in millions) | Total.December 31, 2023 | Home Equity Loans.December 31, 2023
Outstandings | $ 25,527 | $ 26,563
Accruing past due 30 days or more | 95 | 96
Nonperforming loans | 450 | 510
Refreshed FICO below 620 | 3 | 2
"""


def main():
    client = OllamaClient(model="qwen2.5:7b")

    question = "What is the Total Outstandings amount for December 31, 2023?"

    try:
        result = client.generate_financial_answer(question, MOCK_CONTEXT)
        print("\n" + "=" * 50)
        print("🎉 УСПЕХ! Модель вернула валидный JSON:")
        print("=" * 50)
        pprint.pprint(result, sort_dicts=False)
    except Exception as e:
        print(f"\n💥 ПРОВАЛ: {e}")


if __name__ == "__main__":
    main()