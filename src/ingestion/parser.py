import re
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Set

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TableItem, TextItem
from docling.document_converter import DocumentConverter

from src.core.models import (
    ParsingArtifact, ParserInfo, SourceInfo, Provenance,
    TableData, TableFragment, TableGroup, DocumentChunk
)


class FinancialDocumentParser:
    def __init__(self) -> None:
        self.converter = DocumentConverter(allowed_formats=[InputFormat.PDF])

    @staticmethod
    def _get_bbox(item: Any) -> List[float] | None:
        if getattr(item, "prov", None) and item.prov[0].bbox:
            bbox = item.prov[0].bbox
            left = float(bbox.l)
            right = float(bbox.r)
            top = float(max(bbox.t, bbox.b))
            bottom = float(min(bbox.t, bbox.b))
            return [left, top, right, bottom]
        return None

    @staticmethod
    def _get_page_no(item: Any) -> int:
        if getattr(item, "prov", None):
            return int(item.prov[0].page_no)
        return 1

    @staticmethod
    def _normalize_financial_value(val: str) -> str:
        """
        Нормализует только числовые значения:
        - (1,250.50) → -1250.50
        - $307.75 → 307.75
        - 1,439 → 1439
        """
        val = val.strip()
        is_numeric_like = bool(re.match(r'^[\$\(]?\d[\d,.]*\)?$', val))
        
        if not is_numeric_like:
            return val
        
        is_negative = val.startswith('(') and val.endswith(')')
        clean_val = re.sub(r'[^\d.]', '', val)
        
        if not clean_val:
            return val
        
        return f"-{clean_val}" if is_negative else clean_val

    def _reconstruct_sec_tables(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Domain-specific парсер для SEC Form 4 (Table I и Table II).
        """
        table_i_elements = []
        table_ii_elements = []

        for chunk in chunks:
            section = (chunk.metadata.get("section") or "").lower()
            if "table i -" in section:
                table_i_elements.append(chunk)
            elif "table ii -" in section:
                table_ii_elements.append(chunk)

        new_chunks: List[DocumentChunk] = []
        processed_chunk_ids: Set[str] = set()

        TABLE_I_COLUMNS = [
            ("title_of_security", 0, 175),
            ("transaction_date", 175, 265),
            ("transaction_code", 265, 315),
            ("amount", 315, 345),
            ("acquired_disposed", 345, 365),
            ("price", 365, 400),
            ("shares_owned_after", 400, 530),
            ("ownership_form", 530, 600),
        ]

        DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
        MONEY_RE = re.compile(r"^\$?[\d,]+\.?\d*$")
        INTEGER_RE = re.compile(r"^-?[\d,]+$")
        CODE_RE = re.compile(r"^[ACPDSFMGKVJ]$")
        OWNERSHIP_RE = re.compile(r"^[DI]$")

        NUMERIC_COLUMNS = {"amount", "price", "shares_owned_after"}

        def is_valid_transaction_row(row_dict: Dict[str, str]) -> bool:
            if not row_dict.get("title_of_security"):
                return False

            has_finance = False
            if row_dict.get("transaction_date") and DATE_RE.match(row_dict["transaction_date"]):
                has_finance = True
            if row_dict.get("amount") and INTEGER_RE.match(row_dict["amount"].replace(",", "")):
                has_finance = True
            if row_dict.get("price") and MONEY_RE.match(row_dict["price"]):
                has_finance = True

            if row_dict.get("transaction_code") and not CODE_RE.match(row_dict["transaction_code"]):
                return False
            if row_dict.get("ownership_form") and not OWNERSHIP_RE.match(row_dict["ownership_form"]):
                return False

            return has_finance

        def group_by_y(elements: List[DocumentChunk], tolerance: float = 4.0):
            rows = []
            for el in sorted(
                elements,
                key=lambda x: max(x.provenance.bbox[1], x.provenance.bbox[3])
            ):
                y = max(el.provenance.bbox[1], el.provenance.bbox[3])
                target = None
                for row in rows:
                    if abs(row["y"] - y) <= tolerance:
                        target = row
                        break
                if target is None:
                    target = {"y": y, "elements": []}
                    rows.append(target)
                target["elements"].append(el)
            return rows

        def process_table_elements(
            elements: List[DocumentChunk],
            table_name: str,
            processed_ids: Set[str],
            columns: List[tuple]
        ):
            data_candidates = []
            for el in elements:
                if el.element_type != "text":
                    continue
                text = el.content.strip()
                if len(text) > 60 or "Instr." in text or (text.startswith("(") and text.endswith(")")):
                    continue
                data_candidates.append(el)

            y_groups = group_by_y(data_candidates, tolerance=4.0)

            valid_rows = []
            for group_data in y_groups:
                group = group_data["elements"]
                row_dict = {col[0]: None for col in columns}
                group.sort(key=lambda x: min(x.provenance.bbox[0], x.provenance.bbox[2]))

                for el in group:
                    bbox = el.provenance.bbox
                    x_center = (min(bbox[0], bbox[2]) + max(bbox[0], bbox[2])) / 2

                    for col_name, min_x, max_x in columns:
                        if min_x <= x_center < max_x:
                            if row_dict[col_name] is None:
                                raw_value = el.content.strip()
                                if col_name in NUMERIC_COLUMNS:
                                    row_dict[col_name] = self._normalize_financial_value(raw_value)
                                else:
                                    row_dict[col_name] = raw_value
                            break

                if is_valid_transaction_row(row_dict):
                    valid_rows.append(row_dict)

            if valid_rows:
                headers = [col[0] for col in columns]
                text_rows = []
                for row in valid_rows:
                    text_rows.append([str(row.get(h, "") or "") for h in headers])

                table_text = " | ".join(headers) + "\n" + "\n".join(" | ".join(row) for row in text_rows)

                all_bboxes = [el.provenance.bbox for el in data_candidates if el.provenance.bbox]
                if all_bboxes:
                    min_l = min(b[0] for b in all_bboxes)
                    min_t = min(b[1] for b in all_bboxes)
                    max_r = max(b[2] for b in all_bboxes)
                    max_b = max(b[3] for b in all_bboxes)
                    combined_bbox = [min_l, min_t, max_r, max_b]
                else:
                    combined_bbox = [0, 0, 0, 0]

                combined_ref = "_".join([el.provenance.source_ref for el in data_candidates[:3]])
                chunk_id = hashlib.sha256(
                    f"{elements[0].doc_id}:{table_name}:{combined_ref}".encode()
                ).hexdigest()[:16]

                new_chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=elements[0].doc_id,
                    element_type="synthetic_table",
                    content=f"[{table_name.upper()}]\n{table_text}",
                    provenance=Provenance(
                        page_no=elements[0].provenance.page_no,
                        bbox=combined_bbox,
                        source_ref=combined_ref
                    ),
                    table_data=TableData(headers=headers, rows=text_rows),
                    metadata={**elements[0].metadata, "reconstructed": True, "table_type": table_name}
                )
                new_chunks.append(new_chunk)

                for el in data_candidates:
                    processed_ids.add(el.chunk_id)

        # Вызовы обработчиков для Table I и Table II
        process_table_elements(table_i_elements, "table_i", processed_chunk_ids, TABLE_I_COLUMNS)
        process_table_elements(table_ii_elements, "table_ii", processed_chunk_ids, TABLE_I_COLUMNS)

        # Добавляем все остальные элементы, которые не вошли в таблицы
        for chunk in chunks:
            if chunk.chunk_id not in processed_chunk_ids:
                new_chunks.append(chunk)

        return new_chunks

    def _group_tables(self, chunks: List[DocumentChunk]) -> List[TableGroup]:
        table_chunks = [c for c in chunks if c.element_type in ["table", "synthetic_table"]]
        groups: List[TableGroup] = []
        current_group: List[DocumentChunk] = []

        for chunk in table_chunks:
            if not current_group:
                current_group.append(chunk)
            else:
                last_chunk = current_group[-1]
                is_adjacent = chunk.provenance.page_no in {
                    last_chunk.provenance.page_no,
                    last_chunk.provenance.page_no + 1
                }
                same_section = chunk.metadata.get("section") == last_chunk.metadata.get("section")

                headers_match = False
                if chunk.table_data and last_chunk.table_data:
                    h1 = chunk.table_data.headers[:2]
                    h2 = last_chunk.table_data.headers[:2]
                    headers_match = (h1 == h2)

                if is_adjacent and (same_section or headers_match):
                    current_group.append(chunk)
                else:
                    groups.append(self._build_group(current_group))
                    current_group = [chunk]

        if current_group:
            groups.append(self._build_group(current_group))

        return groups

    def _build_group(self, chunks: List[DocumentChunk]) -> TableGroup:
        group_id = f"tg_{chunks[0].provenance.source_ref.split('/')[-1].split(':')[0]}"
        fragments = [
            TableFragment(
                fragment_id=c.chunk_id,
                page_no=c.provenance.page_no,
                bbox=c.provenance.bbox,
                rows=c.table_data.rows if c.table_data else []
            )
            for c in chunks
        ]
        return TableGroup(
            table_group_id=group_id,
            pages=[c.provenance.page_no for c in chunks],
            headers=chunks[0].table_data.headers if chunks[0].table_data else [],
            fragments=fragments
        )

    def parse_pdf(self, pdf_path: str | Path, doc_id: str) -> ParsingArtifact:
        pdf_path = Path(pdf_path)
        with open(pdf_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        result = self.converter.convert(str(pdf_path))
        doc = result.document

        chunks: List[DocumentChunk] = []
        current_section = "Unknown"

        for item, level in doc.iterate_items():
            
            # --------------------------------------
            
            page_no = self._get_page_no(item)
            bbox = self._get_bbox(item)
            ref = str(getattr(item, "self_ref", len(chunks)))

            if isinstance(item, TextItem):
                content = item.text.strip()
                if not content:
                    continue

                label = getattr(item, "label", None)
                label_name = getattr(label, "name", "text").lower()

                if label_name in {"title", "section_header"}:
                    current_section = content

                chunk_id = hashlib.sha256(f"{doc_id}:{page_no}:{ref}".encode()).hexdigest()[:16]
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    element_type=label_name or "text",
                    content=content,
                    provenance=Provenance(page_no=page_no, bbox=bbox, source_ref=ref),
                    metadata={"section": current_section, "hierarchy_level": level}
                ))

            elif isinstance(item, TableItem):
                df = item.export_to_dataframe(doc=doc).fillna("")
                df.columns = [str(col) for col in df.columns]
                rows = [
                    [self._normalize_financial_value(str(val)) for val in row]
                    for row in df.astype(str).values.tolist()
                ]
                table_text = " | ".join(df.columns) + "\n" + "\n".join(" | ".join(row) for row in rows)
                chunk_id = hashlib.sha256(f"{doc_id}:{page_no}:{ref}:table".encode()).hexdigest()[:16]

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    element_type="table",
                    content=f"[TABLE]\n{table_text}",
                    provenance=Provenance(page_no=page_no, bbox=bbox, source_ref=ref),
                    table_data=TableData(headers=list(df.columns), rows=rows),
                    metadata={"section": current_section, "hierarchy_level": level}
                ))

        # === Domain-specific реконструкция SEC таблиц ===
        chunks = self._reconstruct_sec_tables(chunks)

        table_groups = self._group_tables(chunks)

        for group in table_groups:
            for frag in group.fragments:
                for chunk in chunks:
                    if chunk.chunk_id == frag.fragment_id:
                        chunk.metadata["table_group_id"] = group.table_group_id

        return ParsingArtifact(
            artifact_version="1.0",
            parser=ParserInfo(version="2.x"),
            source=SourceInfo(filename=pdf_path.name, sha256=file_hash, doc_id=doc_id),
            elements=chunks,
            table_groups=table_groups,
            total_elements=len(chunks),
            tables_count=len([c for c in chunks if c.element_type in ["table", "synthetic_table"]])
        )