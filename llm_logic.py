"""
llm_logic.py — LLM 호출 및 RAG 로직
현재: 벡터 검색은 실제 동작, LLM 응답은 스텁 (3단계 LLM 연동 예정)
"""
from __future__ import annotations


def get_rag_answer(
    query: str,
    doc_text: str,
    model_id: str,
    chat_history: list[dict] | None = None,
    vector_store=None,
) -> str:
    """
    벡터 DB에서 관련 청크를 검색하고 LLM 답변을 생성합니다.

    Parameters
    ----------
    query        : 사용자 질문
    doc_text     : 업로드된 전체 문서 텍스트 (vector_store 없을 때 폴백용)
    model_id     : 선택된 LLM 모델 ID
    chat_history : 이전 대화 기록 [{"role": ..., "content": ...}]
    vector_store : FAISS 벡터 스토어 (None이면 doc_text 앞부분 사용)
    """
    if vector_store is not None:
        from vectorstore import search_documents
        relevant_chunks = search_documents(vector_store, query, k=4)
        context = "\n\n---\n\n".join(relevant_chunks)
    else:
        context = (doc_text or "")[:2000]

    # --- 스텁 응답 (3단계 LLM 연동 전) ---
    context_preview = context[:600].replace("\n", " ")
    return (
        f"> {context_preview}{'...' if len(context) > 600 else ''}\n\n"
        f"※ 실제 LLM 연동은 3단계에서 구현됩니다."
    )


def generate_quiz(
    doc_text: str,
    model_id: str,
    num_questions: int = 3,
    vector_store=None,
) -> list[dict]:
    """
    문서 내용을 바탕으로 객관식 퀴즈를 생성합니다.

    Returns
    -------
    list[dict] : [{"question": str, "options": [str, ...], "answer": str}]
    """
    # --- 스텁 퀴즈 (3단계 LLM 연동 전) ---
    stub_questions = [
        {
            "question": "이 문서의 주요 주제는 무엇입니까? (스텁 문제)",
            "options": ["A. 인공지능", "B. 우주 탐사", "C. 요리법", "D. 스포츠"],
            "answer": "A. 인공지능",
        },
        {
            "question": "문서에서 가장 많이 언급된 개념은? (스텁 문제)",
            "options": ["A. 데이터", "B. 예술", "C. 음악", "D. 건축"],
            "answer": "A. 데이터",
        },
        {
            "question": "이 문서가 속하는 분야는? (스텁 문제)",
            "options": ["A. 기술/IT", "B. 의학", "C. 법률", "D. 역사"],
            "answer": "A. 기술/IT",
        },
    ]
    return stub_questions[:num_questions]
