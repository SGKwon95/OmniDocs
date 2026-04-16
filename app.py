import html
import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from PIL import Image

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
# LOGO
# ══════════════════════════════════════════════════════════════════════════════
st.logo(
    "logo.png",
    size='large',
)


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
        "Gemini 2.0 Flash (무료)":    ("gemini-2.0-flash",            os.getenv("GOOGLE_API_KEY")),
        "Gemini 1.5 Flash (무료)":   ("gemini-1.5-flash-002",        os.getenv("GOOGLE_API_KEY")),
        "Groq Llama 3 70B (무료)":   ("llama3-70b-8192",             os.getenv("GROQ_API_KEY")),
        "Groq Mixtral (무료)":       ("mixtral-8x7b-32768",          os.getenv("GROQ_API_KEY")),
    }

    model_display = st.selectbox("모델", list(MODELS.keys()), key="model_display")
    selected_model_id, api_key = MODELS[model_display]
    st.caption(f"Model ID: `{selected_model_id}`")

    st.markdown('<hr class="section">', unsafe_allow_html=True)

    # ── 3. 문서 업로드 ──────────────────────────────────────────────────────────
    st.subheader("📁 문서 업로드")

    if st.session_state.doc_name:
        # 파일이 로드된 상태: 업로더 숨기고 파일명 + X 버튼 표시
        safe_name = html.escape(st.session_state.doc_name)
        st.markdown(
            f'<div class="doc-name-bar">📄 <strong>{safe_name}</strong></div>',
            unsafe_allow_html=True,
        )
        if st.button("✕", key="clear_file"):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()
        chars = len(st.session_state.doc_text or "")
        chunk_count = get_chunk_count(st.session_state.vector_store) if st.session_state.vector_store else 0
    else:
        # 파일 없음: 업로더 표시
        uploaded_file = st.file_uploader(
            "PDF 또는 TXT 파일을 업로드하세요",
            type=["pdf", "txt"],
            key="file_uploader",
            help="최대 200 MB · PDF / TXT 형식 지원",
        )
        if uploaded_file:
            with st.spinner("텍스트 추출 중..."):
                text = extract_text_from_file(uploaded_file)
                st.session_state.doc_text = text
                st.session_state.doc_name = uploaded_file.name
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
            st.rerun()

    st.markdown('<hr class="section">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

col_chat, col_quiz = st.columns(2, gap="medium")

# st.columns에 .card 클래스 주입 + 채팅 입력란을 왼쪽 카드의 자식으로 이동
components.html(
    """
    <script>
        function applyLayout() {
            const doc = window.parent.document;
            const block = doc.querySelector('[data-testid="stHorizontalBlock"]');
            if (!block) { setTimeout(applyLayout, 50); return; }

            // 두 컬럼에 .card 클래스 추가
            const cols = block.querySelectorAll(':scope > [data-testid="stColumn"]');
            cols.forEach(col => col.classList.add('card'));

            const stBottom = doc.querySelector('[data-testid="stBottom"]');
            const leftCard = cols[0];
            if (!stBottom || !leftCard) { setTimeout(applyLayout, 50); return; }

        }
        applyLayout();
    </script>
    """,
    height=0,
)


# ══════════════════════════════════════════════════════════════════════════════
# 왼쪽 패널 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with col_chat:
    st.markdown('<p class="panel-title">💬 Chat</p>', unsafe_allow_html=True)

    if st.session_state.doc_text:
        with st.container(height=460, border=False, key="chat_container"):
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

            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                answer_placeholder = st.empty()
                full_answer = ""
                for chunk in stream_rag_answer(
                    query=st.session_state.messages[-1]["content"],
                    doc_text=st.session_state.doc_text,
                    model_id=selected_model_id,
                    api_key=api_key,
                    chat_history=st.session_state.messages[:-1],
                    vector_store=st.session_state.vector_store,
                ):
                    full_answer += chunk
                    answer_placeholder.markdown(
                        f'<div class="chat-assistant">🤖 {full_answer}</div>',
                        unsafe_allow_html=True,
                    )
                st.session_state.messages.append({"role": "assistant", "content": full_answer})
                st.rerun()

            components.html(
                """
                <script>
                    const doc = window.parent.document;
                    const candidates = doc.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
                    let chatContainer = null;
                    for (const el of candidates) {
                        if (el.scrollHeight > el.clientHeight) {
                            chatContainer = el;
                        }
                    }
                    if (chatContainer) {
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                        const observer = new MutationObserver(() => {
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        });
                        observer.observe(chatContainer, { childList: true, subtree: true });
                    }
                </script>
                """,
                height=0,
            )

    user_input = st.chat_input(
        "문서를 먼저 업로드해주세요." if not st.session_state.doc_text else
        "문서에 대해 무엇이든 물어보세요...",
        disabled=not st.session_state.doc_text,
        key="chat_input",
    )

    if user_input and st.session_state.doc_text:
        if not api_key:
            st.error("API Key를 입력해주세요.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 오른쪽 패널 — QUIZ
# ══════════════════════════════════════════════════════════════════════════════
with col_quiz:
    st.markdown('<p class="panel-title">📝 Quiz</p>', unsafe_allow_html=True)

    if st.session_state.doc_text:
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
                        st.radio(
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
                        st.session_state.quiz_answers[q_idx] = st.session_state.get(
                            f"quiz_radio_{q_idx}"
                        )
                    st.session_state.quiz_submitted = True
                    st.rerun()

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


