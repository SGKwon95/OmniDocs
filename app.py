import streamlit as st
from utils import extract_text_from_file
from llm_logic import get_rag_answer, generate_quiz

# ── 페이지 기본 설정 ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniDocs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 전역 CSS ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* 사이드바 헤더 */
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    /* 채팅 말풍선 */
    .chat-user {
        background: #e8f4fd;
        border-radius: 12px 12px 2px 12px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        text-align: right;
        color: #1a1a2e;
    }
    .chat-assistant {
        background: #f0f2f6;
        border-radius: 12px 12px 12px 2px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        color: #1a1a2e;
    }
    /* 피드백 버튼 영역 */
    .feedback-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.3rem;
        margin-left: 0.5rem;
    }
    /* 퀴즈 카드 */
    .quiz-card {
        background: #ffffff;
        border: 1px solid #dde3ec;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    /* 섹션 구분선 */
    hr.section { border: none; border-top: 1px solid #e0e0e0; margin: 1.2rem 0; }
    /* Deploy 버튼 및 더보기(...) 메뉴 숨기기 */
    [data-testid="stAppDeployButton"] { display: none !important; }
    #MainMenu { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    st.markdown('<p class="sidebar-title">📄 OmniDocs</p>', unsafe_allow_html=True)
    st.caption("Multi-LLM 문서 분석 · 퀴즈 생성 플랫폼")
    st.markdown('<hr class="section">', unsafe_allow_html=True)

    # ── 1. 모델 선택 ────────────────────────────────────────────────────────────
    st.subheader("🤖 모델 선택")

    PROVIDER_MODELS = {
        "💰 유료 모델": {
            "Claude 3.5 Sonnet": "claude-sonnet-4-5",
            "Claude 3 Haiku": "claude-haiku-4-5-20251001",
            "GPT-4o": "gpt-4o",
            "GPT-4o Mini": "gpt-4o-mini",
        },
        "🆓 무료 모델": {
            "Gemini 1.5 Pro": "gemini-1.5-pro",
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Groq Llama 3 (70B)": "llama3-70b-8192",
            "Groq Mixtral": "mixtral-8x7b-32768",
        },
    }

    provider_group = st.selectbox(
        "모델 그룹",
        list(PROVIDER_MODELS.keys()),
        key="provider_group",
    )
    model_display = st.selectbox(
        "모델",
        list(PROVIDER_MODELS[provider_group].keys()),
        key="model_display",
    )
    selected_model_id = PROVIDER_MODELS[provider_group][model_display]
    st.caption(f"Model ID: `{selected_model_id}`")

    st.markdown('<hr class="section">', unsafe_allow_html=True)

    # ── 2. 문서 업로드 ──────────────────────────────────────────────────────────
    st.subheader("📁 문서 업로드")
    uploaded_file = st.file_uploader(
        "PDF 또는 TXT 파일을 업로드하세요",
        type=["pdf", "txt"],
        key="file_uploader",
        help="최대 200 MB · PDF / TXT 형식 지원",
    )

    if uploaded_file:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("문서 처리 중..."):
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
            st.success(f"✅ **{uploaded_file.name}** 로드 완료")
            chars = len(st.session_state.doc_text or "")
            st.caption(f"추출 문자 수: {chars:,}자")
    else:
        if st.session_state.doc_name:
            st.info(f"현재 문서: **{st.session_state.doc_name}**")
        else:
            st.info("문서를 업로드하면 분석이 시작됩니다.")

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
st.caption(f"선택 모델: **{model_display}**  |  문서: **{st.session_state.doc_name or '없음'}**")

tab_chat, tab_quiz = st.tabs(["💬 Chat", "📝 Quiz"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.doc_text:
        st.info("👈 사이드바에서 PDF 또는 TXT 문서를 먼저 업로드해주세요.")
    else:
        # 대화 기록 렌더링
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
                fb_key = f"fb_{idx}"
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

        st.markdown("---")

        # 입력창
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "질문을 입력하세요",
                placeholder="문서에 대해 무엇이든 물어보세요...",
                label_visibility="collapsed",
            )
            col_send, col_clear = st.columns([1, 5])
            with col_send:
                submitted = st.form_submit_button("전송 ▶", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.messages.append(
                {"role": "user", "content": user_input.strip()}
            )
            with st.spinner("답변 생성 중..."):
                answer = get_rag_answer(
                    query=user_input.strip(),
                    doc_text=st.session_state.doc_text,
                    model_id=selected_model_id,
                    chat_history=st.session_state.messages[:-1],
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUIZ
# ══════════════════════════════════════════════════════════════════════════════
with tab_quiz:
    if not st.session_state.doc_text:
        st.info("👈 사이드바에서 PDF 또는 TXT 문서를 먼저 업로드해주세요.")
    else:
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
            with st.spinner("퀴즈 생성 중..."):
                questions = generate_quiz(
                    doc_text=st.session_state.doc_text,
                    model_id=selected_model_id,
                )
            st.session_state.quiz_questions = questions
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()

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
