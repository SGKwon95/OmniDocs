"""
llm_logic.py — LLM 호출 및 RAG 로직 (스텁 / 플레이스홀더)
실제 API 연동은 2단계에서 구현합니다.
"""
from __future__ import annotations


# ── 임시 스텁 함수 ────────────────────────────────────────────────────────────
# UI 개발 단계에서 실제 LLM 없이 앱을 실행·확인할 수 있도록 더미 응답을 반환합니다.
# 2단계에서 Claude / OpenAI / Gemini / Groq 클라이언트로 교체합니다.

def get_rag_answer(
    query: str,
    doc_text: str,
    model_id: str,
    chat_history: list[dict] | None = None,
) -> str:
    """
    문서 텍스트와 대화 기록을 바탕으로 LLM 답변을 생성합니다.

    Parameters
    ----------
    query       : 사용자 질문
    doc_text    : 업로드된 전체 문서 텍스트
    model_id    : 선택된 LLM 모델 ID
    chat_history: 이전 대화 기록 [{"role": ..., "content": ...}]

    Returns
    -------
    str : 모델의 답변 텍스트
    """
    # --- 스텁 응답 (2단계 구현 전) ---
    preview = doc_text[:200].replace("\n", " ")
    return (
        f"[스텁 응답 - 모델: {model_id}]\n\n"
        f"질문: **{query}**\n\n"
        f"문서 앞부분 미리보기: {preview}...\n\n"
        f"※ 실제 LLM 연동은 2단계에서 구현됩니다."
    )


def generate_quiz(
    doc_text: str,
    model_id: str,
    num_questions: int = 3,
) -> list[dict]:
    """
    문서 내용을 바탕으로 객관식 퀴즈를 생성합니다.

    Returns
    -------
    list[dict] : [
        {
            "question": str,
            "options":  [str, str, str, str],
            "answer":   str   # options 중 하나
        },
        ...
    ]
    """
    # --- 스텁 퀴즈 (2단계 구현 전) ---
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
