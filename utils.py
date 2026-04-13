"""
utils.py — 문서 처리 유틸리티
현재: 텍스트 추출 (PDF / TXT)
추후: 청킹(Chunking), 벡터 DB 색인 등 확장 예정
"""
from __future__ import annotations

import io


def extract_text_from_file(uploaded_file) -> str:
    """Streamlit UploadedFile 객체에서 텍스트를 추출해 반환."""
    filename: str = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        raw_bytes = uploaded_file.read()
        # UTF-8 → Latin-1 순서로 디코딩 시도
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace")

    elif filename.endswith(".pdf"):
        return _extract_pdf(uploaded_file)

    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {uploaded_file.name}")


def _extract_pdf(uploaded_file) -> str:
    """PDF 파일에서 텍스트를 추출 (pypdf 사용)."""
    try:
        import pypdf  # type: ignore
    except ImportError:
        raise ImportError(
            "PDF 처리를 위해 pypdf 패키지가 필요합니다.\n"
            "설치: pip install pypdf"
        )

    reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
    pages_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    텍스트를 chunk_size 단위로 분할 (RAG 인덱싱용).
    향후 llm_logic.py의 벡터 DB 구축 단계에서 사용됩니다.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
