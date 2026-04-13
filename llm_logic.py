"""
llm_logic.py — LLM 호출 및 RAG 로직
지원: Claude (Anthropic) · GPT (OpenAI) · Gemini (Google) · Groq
"""
from __future__ import annotations
import json
import re


# ── 공급자 감지 ───────────────────────────────────────────────────────────────
def detect_provider(model_id: str) -> str:
    if model_id.startswith("claude"):
        return "anthropic"
    elif model_id.startswith("gpt"):
        return "openai"
    elif model_id.startswith("gemini"):
        return "google"
    else:  # llama*, mixtral* → groq
        return "groq"


# ── 통합 LLM 호출 ─────────────────────────────────────────────────────────────
def _call_llm(
    system: str,
    messages: list[dict],
    model_id: str,
    api_key: str,
    max_tokens: int = 1024,
) -> str:
    provider = detect_provider(model_id)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        full_msgs = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=model_id,
            messages=full_msgs,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system,
        )
        history = [
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [m["content"]],
            }
            for m in messages[:-1]
        ]
        chat = model.start_chat(history=history)
        resp = chat.send_message(messages[-1]["content"])
        return resp.text

    elif provider == "groq":
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        full_msgs = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=model_id,
            messages=full_msgs,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    raise ValueError(f"알 수 없는 모델: {model_id}")


# ── RAG 답변 ──────────────────────────────────────────────────────────────────
def get_rag_answer(
    query: str,
    doc_text: str,
    model_id: str,
    api_key: str,
    chat_history: list[dict] | None = None,
    vector_store=None,
) -> str:
    if vector_store is not None:
        from vectorstore import search_documents
        chunks = search_documents(vector_store, query, k=4)
        context = "\n\n---\n\n".join(chunks)
    else:
        context = (doc_text or "")[:3000]

    system = (
        "You are a helpful document assistant. "
        "Answer the user's question using only the document context below. "
        "If the answer is not in the context, say so clearly. "
        "Respond in the same language as the user's question.\n\n"
        f"[Document Context]\n{context}"
    )

    messages = list(chat_history or [])
    messages.append({"role": "user", "content": query})

    return _call_llm(system, messages, model_id, api_key)


# ── 퀴즈 생성 ─────────────────────────────────────────────────────────────────
def generate_quiz(
    doc_text: str,
    model_id: str,
    api_key: str,
    num_questions: int = 3,
    vector_store=None,
) -> list[dict]:
    context = (doc_text or "")[:4000]

    system = "You are a quiz generator. Output only valid JSON with no markdown fences and no explanation."

    prompt = (
        f"Create {num_questions} multiple-choice questions from the document below.\n\n"
        f"[Document]\n{context}\n\n"
        "Return a JSON array in exactly this format:\n"
        '[{"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A. ..."}]'
    )

    raw = _call_llm(
        system,
        [{"role": "user", "content": prompt}],
        model_id,
        api_key,
        max_tokens=1500,
    )
    return _parse_quiz(raw, num_questions)


def _parse_quiz(raw: str, n: int) -> list[dict]:
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        raw = m.group()
    try:
        questions = json.loads(raw)
        if isinstance(questions, list) and questions:
            return questions[:n]
    except json.JSONDecodeError:
        pass
    return [
        {
            "question": f"Q{i + 1}: 퀴즈 파싱에 실패했습니다. 모델 응답을 확인하세요.",
            "options": ["A. -", "B. -", "C. -", "D. -"],
            "answer": "A. -",
        }
        for i in range(n)
    ]
