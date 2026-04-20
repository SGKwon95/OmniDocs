# OmniDocs

Multi-LLM 기반 문서 분석 및 퀴즈 생성 웹앱 (Streamlit)

---

## 개요

PDF · TXT 문서를 업로드하면 RAG(Retrieval-Augmented Generation) 방식으로 문서 내용을 기반으로 대화하고, LLM이 자동으로 객관식 퀴즈를 생성해주는 웹앱입니다.

---

## 주요 기능

| 기능         | 설명                                              |
| ------------ | ------------------------------------------------- |
| 문서 업로드  | PDF / TXT 파일 업로드 및 텍스트 추출              |
| 벡터 DB 구축 | FAISS + HuggingFace 임베딩으로 청킹 및 인덱싱     |
| RAG 채팅     | 문서 컨텍스트 기반 스트리밍 답변                  |
| 피드백       | 답변별 👍 / 👎 피드백                             |
| 퀴즈 생성    | LLM이 문서에서 3문제 객관식 퀴즈 자동 생성 + 채점 |
| 멀티 LLM     | Claude, GPT 등 여러 모델 선택 지원                |

---

## 개발 환경

| 항목           | 버전 / 내용                                                          |
| -------------- | -------------------------------------------------------------------- |
| Python         | 3.10+                                                                |
| UI 프레임워크  | Streamlit >= 1.35.0                                                  |
| PDF 파싱       | pdfplumber >= 0.10.0 (폴백: pypdf >= 4.0.0)                          |
| 임베딩         | sentence-transformers `all-MiniLM-L6-v2` (CPU)                       |
| 벡터 DB        | FAISS (faiss-cpu >= 1.7.4)                                           |
| RAG 파이프라인 | LangChain >= 0.2.0                                                   |
| LLM SDK        | anthropic >= 0.40.0 · openai >= 1.0.0 · google-generativeai >= 0.8.0 |
| 환경 변수      | python-dotenv                                                        |

---

OmniDocs는 문서를 업로드할 때마다 벡터 DB를 새로 구축하는 구조라서

- 영속성이 필요 없음 → 클라우드 DB 불필요
- 서버 없이 로컬 실행만으로 충분
- LangChain과 궁합이 좋고 설치가 단순

→ FAISS가 가장 적합한 선택

---

chunk_size=1000, chunk_overlap=200으로 정한 이유

chunk_size=1000 (문자 수 기준)

[ chunk 1 : 1000자 ]
[ chunk 2 : 1000자 ]

- 너무 작으면 (예: 200자): 하나의 문장만 담겨 맥락이 부족 → LLM이 제대로 답변 불가
- 너무 크면 (예: 5000자): 검색 정밀도 하락, 토큰 낭비, 응답 속도 저하
- 1000자 ≈ 한국어 약 500단어 = 문단 2~3개 분량으로 맥락을 유지하면서도 검색 정밀도를 확보하는 경험적 기본값

chunk_overlap=200 (청크 간 겹치는 문자 수)

[ ← chunk 1 (1000자) → ]
[ ← chunk 2 (1000자) → ]
↑ overlap (200자) ↑

- 청크 경계에서 문장이 잘리면 중요한 정보가 양쪽 청크에 분산될 수 있음
- overlap이 없으면 경계 근처의 내용을 질문했을 때 검색에서 누락될 위험
- 200자(chunk_size의 20%)는 경계 정보 손실을 방지하는 표준 비율

### 결론

chunk_size=1000 / overlap=200은 LangChain 공식 문서와 커뮤니티에서 범용적으로 권장하는 "일단 잘 동작하는" 기본값입니다. 문서 특성에 따라
조정해 보면서 사용할 예정

---

## 파일 구조

```
OmniDocs/
├── app.py            # 메인 UI (Streamlit)
├── llm_logic.py      # LLM 호출 및 RAG 로직 (스트리밍 포함)
├── vectorstore.py    # 문서 청킹 및 FAISS 벡터 DB 구축/검색
├── utils.py          # 문서 텍스트 추출 (PDF / TXT)
├── style.css         # 커스텀 CSS
├── logo.png          # 앱 로고
├── requirements.txt  # 패키지 의존성
└── .env              # API 키 (gitignore)
```

---

## 지원 LLM 모델

### 유료 (API Key 필요)

| 표시명            | 모델 ID                     | 공급자    |
| ----------------- | --------------------------- | --------- |
| Claude 3.5 Sonnet | `claude-sonnet-4-5`         | Anthropic |
| Claude 3 Haiku    | `claude-haiku-4-5-20251001` | Anthropic |
| GPT-4o            | `gpt-4o`                    | OpenAI    |
| GPT-4o Mini       | `gpt-4o-mini`               | OpenAI    |

> Gemini (Google) · Groq (Llama 3, Mixtral) 모델은 코드에 정의되어 있으나 현재 주석 처리 상태

---

## 아키텍처

```
문서 업로드
    │
    ▼
extract_text_from_file()   ← utils.py
    │  pdfplumber → pypdf 폴백
    ▼
build_vector_store()       ← vectorstore.py
    │  RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
    │  → FAISS + all-MiniLM-L6-v2 임베딩
    ▼
사용자 질문 입력
    │
    ▼
search_documents()         ← vectorstore.py (top-k=4 청크 검색)
    │
    ▼
stream_rag_answer()        ← llm_logic.py
    │  시스템 프롬프트 + 검색된 컨텍스트 + 대화 히스토리
    │  → Claude / OpenAI / Gemini / Groq 스트리밍 API
    ▼
Streamlit 실시간 렌더링
```

---

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...        # 선택
GROQ_API_KEY=gsk_...      # 선택
```

### 3. 앱 실행

```bash
streamlit run app.py
```

---

## 개발 진행 현황

### 완료

- [x] Streamlit UI 구조 (사이드바, 채팅 패널, 퀴즈 패널)
- [x] PDF · TXT 텍스트 추출 (pdfplumber + pypdf 폴백)
- [x] FAISS 벡터 DB 구축 및 유사도 검색
- [x] Claude / OpenAI 연동
- [x] 스트리밍 답변 (실시간 렌더링)
- [x] 대화 히스토리 관리
- [x] 객관식 퀴즈 생성 및 채점
- [x] 커스텀 CSS 스타일링 및 로고

### 예정

- [ ] Next.js로 이관
- [ ] 답변 피드백 로직 처리 (👍 / 👎)
- [ ] Gemini / Groq 무료 모델 재활성화
- [ ] 청크 수 / 문서 통계 사이드바 표시
- [ ] 로그인
- [ ] 과거 쿼리 및 질문 내역 조회
- [ ] 청크 수 / 청크 오버랩 수 조절 기능
- [ ] 모델별 RAG 평가 통계 기능
- [ ] AI에이전트(Tempo, Grafana, Loki 등) 연동
- [ ] RAG 검색 알고리즘(BM25, Dense Retrieval, Hybrid Search 등) 선택 기능
- [ ] 벡터 DB(Chroma, Pinecone) 변경 기능
