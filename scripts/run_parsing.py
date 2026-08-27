import json
import sys
from pathlib import Path

# Добавляем корень проекта в пути, чтобы видеть модуль src
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.parser import FinancialDocumentParser

def main():
    # Указываем путь к твоему файлу
    pdf_path = Path("data/raw/01_v.pdf")  
    doc_id = "DOC_01_V"
    
    if not pdf_path.exists():
        print(f"❌ Ошибка: Файл {pdf_path} не найден.")
        print(f"Проверь, что файл 01_v.pdf лежит в папке data/raw/")
        return

    print(f"⚙️ Парсинг {pdf_path.name}...")
    parser = FinancialDocumentParser()
    
    try:
        artifact = parser.parse_pdf(pdf_path, doc_id)
    except Exception as e:
        print(f"❌ Ошибка при парсинге: {e}")
        return
    
    # Сохранение артефакта
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{doc_id}_artifact.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(artifact.model_dump(), f, indent=2, ensure_ascii=False)
        
    print(f"✅ Успешно!")
    print(f"   Всего элементов: {artifact.total_elements}")
    print(f"   Найдено таблиц: {artifact.tables_count}")
    print(f"   Сформировано групп таблиц: {len(artifact.table_groups)}")
    print(f"   Артефакт сохранен: {output_file}")

if __name__ == "__main__":
    main()