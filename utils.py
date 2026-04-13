"""
utils.py — 문서 처리 유틸리티
텍스트 추출 (PDF / TXT) · 청킹은 vectorstore.py의 RecursiveCharacterTextSplitter 사용
"""
from __future__ import annotations

import io


def extract_text_from_file(uploaded_file) -> str:
    """Streamlit UploadedFile 객체에서 텍스트를 추출해 반환."""
    filename: str = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        raw_bytes = uploaded_file.read()
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
    """PDF 파일에서 텍스트를 추출 (pdfplumber 우선, 실패 시 pypdf 폴백)."""
    raw = uploaded_file.read()

    # 1차 시도: pdfplumber (한글 폰트 처리 우수)
    try:
        import pdfplumber  # type: ignore
        pages_text: list[str] = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
        result = "\n\n".join(pages_text)
        if result.strip():
            return result
    except Exception:
        pass

    # 2차 시도: pypdf 폴백
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(io.BytesIO(raw))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except Exception as e:
        raise RuntimeError(f"PDF 텍스트 추출 실패: {e}")
