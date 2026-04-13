"""
vectorstore.py — 문서 청킹 및 FAISS 벡터 DB 구축/검색
LangChain RecursiveCharacterTextSplitter + HuggingFace Embeddings 사용
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 임베딩 모델은 최초 1회만 로드 (모듈 레벨 캐시)
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_vector_store(text: str, source_name: str = "document") -> FAISS:
    """
    텍스트를 청킹하고 FAISS 벡터 DB를 생성해 반환합니다.

    Parameters
    ----------
    text        : 전체 문서 텍스트
    source_name : 메타데이터에 저장할 출처 이름

    Returns
    -------
    FAISS 벡터 스토어 객체
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_id": i},
        )
        for i, chunk in enumerate(chunks)
    ]
    vector_store = FAISS.from_documents(documents, get_embeddings())
    return vector_store


def search_documents(vector_store: FAISS, query: str, k: int = 4) -> list[str]:
    """
    쿼리와 가장 유사한 청크 k개를 반환합니다.

    Returns
    -------
    list[str] : 관련 텍스트 청크 목록
    """
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def get_chunk_count(vector_store: FAISS) -> int:
    """저장된 청크 수를 반환합니다."""
    return vector_store.index.ntotal
