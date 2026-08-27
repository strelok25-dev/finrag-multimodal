from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

class ParserInfo(BaseModel):
    name: str = "docling"
    version: str = "2.x"

class SourceInfo(BaseModel):
    filename: str
    sha256: str
    doc_id: str

class Provenance(BaseModel):
    page_no: int = Field(..., ge=1, description="Номер страницы (>= 1)")
    bbox: Optional[List[float]] = Field(None, description="[left, top, right, bottom]")
    source_ref: str = Field(..., description="Внутренний ID элемента Docling")
    coordinate_system: str = "pdf"

    @field_validator('bbox')
    @classmethod
    def validate_bbox(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if len(v) != 4:
                raise ValueError("bbox must contain exactly 4 coordinates")
            # Жесткая нормализация: гарантируем left < right и top < bottom
            left, top, right, bottom = v
            if left > right: left, right = right, left
            if top > bottom: top, bottom = bottom, top
            return [float(left), float(top), float(right), float(bottom)]
        return v

class TableData(BaseModel):
    headers: List[str]
    rows: List[List[str]]

class TableFragment(BaseModel):
    fragment_id: str
    page_no: int
    bbox: Optional[List[float]]
    rows: List[List[str]]

class TableGroup(BaseModel):
    table_group_id: str
    pages: List[int]
    headers: List[str]
    fragments: List[TableFragment]

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    element_type: str  # "text", "table", "title", etc.
    content: str
    provenance: Provenance
    table_data: Optional[TableData] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ParsingArtifact(BaseModel):
    artifact_version: str = "1.0"
    parser: ParserInfo
    source: SourceInfo
    elements: List[DocumentChunk]
    table_groups: List[TableGroup]
    total_elements: int
    tables_count: int