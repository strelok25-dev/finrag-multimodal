import hashlib
from typing import List, Dict, Any

def strategy_a_text_only(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Baseline A: Только текст и заголовки. Таблицы игнорируются."""
    chunks = []
    for el in elements:
        if el["element_type"] in ["text", "section_header", "list_item"]:
            chunk_id = hashlib.md5(f"{el['chunk_id']}_A".encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": el["content"],
                "metadata": {**el["metadata"], "strategy": "A_text_only", "element_type": el["element_type"]}
            })
    return chunks

def strategy_b_docling_hybrid(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Baseline B: Готовые чанки Docling как есть (текст + таблицы)."""
    chunks = []
    for el in elements:
        chunk_id = hashlib.md5(f"{el['chunk_id']}_B".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": el["content"],
            "metadata": {**el["metadata"], "strategy": "B_docling_hybrid", "element_type": el["element_type"]}
        })
    return chunks

def strategy_c_table_aware(elements: List[Dict[str, Any]], table_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Baseline C: Table-Aware Parent-Child. Родитель = заголовки, Дети = строки."""
    chunks = []

    # 1. Текстовые элементы
    for el in elements:
        if el["element_type"] != "table":
            chunk_id = hashlib.md5(f"{el['chunk_id']}_C_text".encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": el["content"],
                "metadata": {**el["metadata"], "strategy": "C_table_aware", "chunk_type": "text"}
            })

    # 2. Табличные элементы (Parent-Child)
    for group in table_groups:
        headers_str = " | ".join(str(h) for h in group["headers"])
        pages_str = ", ".join(map(str, group["pages"]))

        # Parent chunk (контекст всей таблицы)
        parent_id = hashlib.md5(f"{group['table_group_id']}_parent".encode()).hexdigest()
        chunks.append({
            "id": parent_id,
            "text": f"TABLE CONTEXT: Headers=[{headers_str}] | Pages=[{pages_str}]",
            "metadata": {
                "strategy": "C_table_aware",
                "chunk_type": "table_parent",
                "table_group_id": group["table_group_id"],
                "pages": group["pages"],
                "headers": group["headers"]
            }
        })

        # Child chunks (отдельные строки)
        for frag in group["fragments"]:
            for row_idx, row in enumerate(frag["rows"]):
                row_text = " | ".join(str(val) for val in row)
                child_id = hashlib.md5(f"{group['table_group_id']}_row_p{frag['page_no']}_r{row_idx}".encode()).hexdigest()
                chunks.append({
                    "id": child_id,
                    "text": f"ROW DATA: {row_text}",
                    "metadata": {
                        "strategy": "C_table_aware",
                        "chunk_type": "table_child",
                        "parent_id": group["table_group_id"],
                        "page_no": frag["page_no"],
                        "table_group_id": group["table_group_id"]
                    }
                })
    return chunks