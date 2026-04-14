import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from utils import extract_text_from_file
from llm_logic import get_rag_answer, stream_rag_answer, generate_quiz, detect_provider
from vectorstore import build_vector_store, get_chunk_count

# ── 페이지 기본 설정 ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniDocs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 전역 CSS ────────────────────────────────────────────────────────────────────
_css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── session_state 초기화 ────────────────────────────────────────────────────────
defaults = {
    "messages": [],           # [{"role": "user"|"assistant", "content": str}]
    "feedback": {},           # {msg_index: "like" | "dislike"}
    "quiz_questions": [],     # [{"question":str, "options":[str], "answer":str}]
    "quiz_answers": {},       # {q_index: selected_option}
    "quiz_submitted": False,
    "doc_text": None,         # 추출된 문서 텍스트
    "doc_name": None,
    "vector_store": None,     # (향후 RAG 사용)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── 1. 모델 선택 ────────────────────────────────────────────────────────────
    st.subheader("🤖 모델 선택")

    MODELS = {
        "Claude 3.5 Sonnet (유료)":  ("claude-sonnet-4-5",          os.getenv("ANTHROPIC_API_KEY")),
        "Claude 3 Haiku (유료)":     ("claude-haiku-4-5-20251001",   os.getenv("ANTHROPIC_API_KEY")),
        "GPT-4o (유료)":             ("gpt-4o",                      os.getenv("OPENAI_API_KEY")),
        "GPT-4o Mini (유료)":        ("gpt-4o-mini",                 os.getenv("OPENAI_API_KEY")),
        "Gemini 1.5 Pro (무료)":     ("gemini-1.5-pro",              os.getenv("GOOGLE_API_KEY")),
        "Gemini 1.5 Flash (무료)":   ("gemini-1.5-flash",            os.getenv("GOOGLE_API_KEY")),
        "Groq Llama 3 70B (무료)":   ("llama3-70b-8192",             os.getenv("GROQ_API_KEY")),
        "Groq Mixtral (무료)":       ("mixtral-8x7b-32768",          os.getenv("GROQ_API_KEY")),
    }

    model_display = st.selectbox("모델", list(MODELS.keys()), key="model_display")
    selected_model_id, api_key = MODELS[model_display]
    st.caption(f"Model ID: `{selected_model_id}`")

    st.markdown('<hr class="section">', unsafe_allow_html=True)

    # ── 3. 문서 업로드 ──────────────────────────────────────────────────────────
    st.subheader("📁 문서 업로드")
    uploaded_file = st.file_uploader(
        "PDF 또는 TXT 파일을 업로드하세요",
        type=["pdf", "txt"],
        key="file_uploader",
        help="최대 200 MB · PDF / TXT 형식 지원",
    )

    if uploaded_file:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("텍스트 추출 중..."):
                text = extract_text_from_file(uploaded_file)
                st.session_state.doc_text = text
                st.session_state.doc_name = uploaded_file.name
                # 새 문서 → 대화·퀴즈 초기화
                st.session_state.messages = []
                st.session_state.feedback = {}
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.vector_store = None
            with st.spinner("벡터 DB 구축 중... (최초 실행 시 모델 다운로드로 수분 소요)"):
                st.session_state.vector_store = build_vector_store(
                    text, source_name=uploaded_file.name
                )
            chunk_count = get_chunk_count(st.session_state.vector_store)
            st.success(f"✅ **{uploaded_file.name}** 로드 완료")
            chars = len(st.session_state.doc_text or "")
            st.caption(f"추출 문자 수: {chars:,}자 · 청크 수: {chunk_count}개")
    elif st.session_state.doc_name:
        st.info(f"현재 문서: **{st.session_state.doc_name}**")

    st.markdown('<hr class="section">', unsafe_allow_html=True)

    # ── 3. 초기화 버튼 ─────────────────────────────────────────────────────────
    if st.button("🗑️ 초기화 (문서·대화·퀴즈)", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.title("📄 OmniDocs")

# ── 채팅 입력을 탭보다 먼저 처리 → 사용자 메시지를 즉시 session_state에 저장 ──
user_input = st.chat_input(
    "👈 사이드바에서 PDF 또는 TXT 문서를 먼저 업로드해주세요." if not st.session_state.doc_text else
    "문서에 대해 무엇이든 물어보세요...",
    disabled=not st.session_state.doc_text,
)

if user_input and st.session_state.doc_text:
    if not api_key:
        st.error("API Key를 입력해주세요.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": user_input})

tab_chat, tab_quiz = st.tabs(["💬 Chat", "📝 Quiz"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if st.session_state.doc_text:
        # ── 스크롤 가능한 대화 영역 ──────────────────────────────────────────────
        with st.container(height=480, border=False):
            for idx, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-user">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-assistant">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # 피드백 버튼 (assistant 메시지에만)
                    current_fb = st.session_state.feedback.get(idx)
                    col_like, col_dislike, col_spacer = st.columns([1, 1, 10])
                    with col_like:
                        like_type = "primary" if current_fb == "like" else "secondary"
                        if st.button("👍", key=f"like_{idx}", type=like_type):
                            st.session_state.feedback[idx] = (
                                None if current_fb == "like" else "like"
                            )
                            st.rerun()
                    with col_dislike:
                        dislike_type = "primary" if current_fb == "dislike" else "secondary"
                        if st.button("👎", key=f"dislike_{idx}", type=dislike_type):
                            st.session_state.feedback[idx] = (
                                None if current_fb == "dislike" else "dislike"
                            )
                            st.rerun()

            # ── 마지막 메시지가 user면 스트리밍 응답 시작 ──────────────────────
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                answer = st.write_stream(
                    stream_rag_answer(
                        query=st.session_state.messages[-1]["content"],
                        doc_text=st.session_state.doc_text,
                        model_id=selected_model_id,
                        api_key=api_key,
                        chat_history=st.session_state.messages[:-1],
                        vector_store=st.session_state.vector_store,
                    )
                )
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUIZ
# ══════════════════════════════════════════════════════════════════════════════
with tab_quiz:
    if st.session_state.doc_text:
        st.subheader("📝 문서 기반 퀴즈")
        st.caption("업로드된 문서 내용을 바탕으로 객관식 3문제를 자동 생성합니다.")

        col_gen, _ = st.columns([2, 5])
        with col_gen:
            gen_btn = st.button(
                "🎲 퀴즈 생성",
                use_container_width=True,
                disabled=st.session_state.quiz_submitted,
            )

        if gen_btn:
            if not api_key:
                st.error("API Key를 입력해주세요.")
                st.stop()
            with st.spinner("퀴즈 생성 중..."):
                questions = generate_quiz(
                    doc_text=st.session_state.doc_text,
                    model_id=selected_model_id,
                    api_key=api_key,
                )
            st.session_state.quiz_questions = questions
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()

        # ── 스크롤 가능한 퀴즈 영역 ──────────────────────────────────────────────
        with st.container(height=500, border=False):
            if st.session_state.quiz_questions:
                with st.form(key="quiz_form"):
                    for q_idx, q in enumerate(st.session_state.quiz_questions):
                        st.markdown(
                            f'<div class="quiz-card">'
                            f'<b>Q{q_idx + 1}.</b> {q["question"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        choice = st.radio(
                            f"Q{q_idx + 1} 선택",
                            options=q["options"],
                            key=f"quiz_radio_{q_idx}",
                            label_visibility="collapsed",
                        )

                    submit_quiz = st.form_submit_button(
                        "✅ 제출",
                        disabled=st.session_state.quiz_submitted,
                        use_container_width=False,
                    )

                if submit_quiz:
                    for q_idx in range(len(st.session_state.quiz_questions)):
                        radio_key = f"quiz_radio_{q_idx}"
                        st.session_state.quiz_answers[q_idx] = st.session_state.get(
                            radio_key
                        )
                    st.session_state.quiz_submitted = True
                    st.rerun()

                # 채점 결과 표시
                if st.session_state.quiz_submitted:
                    st.markdown("---")
                    st.subheader("🏆 채점 결과")
                    score = 0
                    for q_idx, q in enumerate(st.session_state.quiz_questions):
                        user_ans = st.session_state.quiz_answers.get(q_idx, "")
                        correct = q["answer"]
                        is_correct = user_ans == correct
                        if is_correct:
                            score += 1
                        icon = "✅" if is_correct else "❌"
                        st.markdown(
                            f"**Q{q_idx + 1}.** {q['question']}  \n"
                            f"{icon} 선택: **{user_ans}**  \n"
                            f"정답: **{correct}**"
                        )
                        st.markdown("---")

                    total = len(st.session_state.quiz_questions)
                    pct = int(score / total * 100) if total else 0
                    st.metric("점수", f"{score} / {total}", f"{pct}%")

                    if st.button("🔄 다시 풀기"):
                        st.session_state.quiz_questions = []
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.rerun()


