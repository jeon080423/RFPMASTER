
import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import os
from kiwipiepy import Kiwi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import datetime
import time
import auth
import email_utils

# -----------------------------------------------------------------------------
# 1. Config & Branding
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="수주비책 - RFP 분석 솔루션",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800; text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem; text-align: center; color: #888;
        margin-bottom: 1.5rem;
    }
    .footer {
        text-align: center; color: #aaa; font-size: 0.85rem;
        margin-top: 3rem; padding: 1rem 0;
        border-top: 1px solid #eee;
    }
    /* Sidebar login styling */
    .sidebar-login-header {
        font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Authentication (Sidebar)
# -----------------------------------------------------------------------------
auth.init_db()

if "user" not in st.session_state:
    st.session_state.user = None

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921222.png", width=50)
    
    if st.session_state.user:
        # --- Logged-in state ---
        st.success(f"👤 **{st.session_state.user['name']}**님 접속 중")
        
        if not st.session_state.user.get('approved', False):
            st.warning("⏳ 계정 승인 대기 중")
        
        if st.button("로그아웃", key="logout_sidebar", use_container_width=True):
            st.session_state.user = None
            st.rerun()
        
        # Admin Logic
        if st.session_state.user.get('role') == 'admin':
            st.markdown("---")
            st.subheader("👑 관리자 메뉴")
            pending_users = auth.get_pending_users()
            if not pending_users.empty:
                st.warning(f"승인 대기: {len(pending_users)}명")
                for _, row in pending_users.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{row['name']}** ({row['email']})")
                    with col2:
                        if st.button("승인", key=f"btn_{row['email']}"):
                            auth.approve_user(row['email'])
                            success, form = email_utils.send_approval_email(row['email'])
                            if success:
                                st.success(f"승인 완료!")
                            else:
                                st.warning(f"승인 완료, 메일 실패")
                            st.rerun()
            else:
                st.info("승인 대기 중인 회원 없음")
    else:
        # --- Not logged-in state ---
        st.markdown('<div class="sidebar-login-header">🔐 로그인</div>', unsafe_allow_html=True)
        
        login_tab, signup_tab = st.tabs(["로그인", "회원가입"])
        
        with login_tab:
            email = st.text_input("이메일", key="login_email")
            password = st.text_input("비밀번호", type="password", key="login_pw")
            if st.button("로그인", type="primary", use_container_width=True):
                user = auth.login_user(email, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("이메일 또는 비밀번호가 잘못되었습니다.")
        
        with signup_tab:
            new_email = st.text_input("이메일", key="signup_email")
            new_name = st.text_input("이름", key="signup_name")
            new_password = st.text_input("비밀번호", type="password", key="signup_pw")
            new_password_confirm = st.text_input("비밀번호 확인", type="password", key="signup_pw_chk")
            
            if st.button("가입하기", use_container_width=True):
                if new_password != new_password_confirm:
                    st.error("비밀번호가 일치하지 않습니다.")
                elif not new_email or not new_password or not new_name:
                    st.error("모든 정보를 입력해주세요.")
                else:
                    if auth.create_user(new_email, new_password, new_name):
                        email_utils.send_admin_notification(new_email, new_name)
                        st.success("가입 요청 완료! 관리자 승인 후 이용 가능합니다.")
                    else:
                        st.error("이미 가입된 이메일입니다.")
    
    # Settings
    st.markdown("---")
    api_key = st.secrets.get("gemini", {}).get("api_key", "")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    st.markdown("---")
    st.markdown("**Developed by ㅈㅅㅎ**")

# -----------------------------------------------------------------------------
# 3. Utility Functions
# -----------------------------------------------------------------------------
def mask_pii(text):
    rrn_pattern = r'\d{6}[-\s]\d{7}'
    masked_text = re.sub(rrn_pattern, '******-*******', text)
    return masked_text

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return f"Error reading PDF: {e}"
    return mask_pii(text)

def analyze_keywords(text):
    kiwi = Kiwi()
    tokens = kiwi.tokenize(text[:200000])
    nouns = [token.form for token in tokens if token.tag.startswith('NN') and len(token.form) > 1]
    stopwords = ['대한', '관련', '위해', '경우', '사항', '이상', '이하', '기타', '포함', '수행', '작성', '제출', '해당']
    nouns = [n for n in nouns if n not in stopwords]
    count = Counter(nouns)
    return count.most_common(20)

def create_word_chart(keywords):
    if not keywords: return None
    words, counts = zip(*keywords)
    fig, ax = plt.subplots(figsize=(10, 6))
    import platform
    import matplotlib.font_manager as fm
    
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
        ]
        font_name = None
        for path in paths:
            if os.path.exists(path):
                font_name = fm.FontProperties(fname=path).get_name()
                break
        if font_name:
            plt.rc('font', family=font_name)
        else:
            plt.rc('font', family='NanumGothic')
    
    plt.rc('axes', unicode_minus=False)
    ax.barh(words, counts, color='#3B82F6')
    ax.invert_yaxis()
    ax.set_xlabel('빈도수')
    ax.set_title('상위 20개 핵심 키워드')
    return fig

def get_relevant_context(text, keywords, box_size=2000, max_len=4000):
    """Extracts relevant text chunks around keywords. Sizes reduced for Groq TPM limit."""
    relevant_chunks = []
    text_lower = text.lower()
    for kw in keywords:
        start_idx = 0
        while True:
            idx = text_lower.find(kw, start_idx)
            if idx == -1: break
            start = max(0, idx - 500)
            end = min(len(text), idx + box_size)
            chunk = text[start:end]
            relevant_chunks.append(chunk)
            start_idx = idx + len(kw)
    if not relevant_chunks:
        return text[:max_len]
    combined = "\n...\n".join(relevant_chunks)
    return combined[:max_len]

# -----------------------------------------------------------------------------
# 4. Main Content (Always visible)
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">수주비책 (Win Strategy)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">공공기관 입찰 성공을 위한 제안요청서(RFP) 심층 분석 솔루션</div>', unsafe_allow_html=True)

# Rate limit retry helper
def invoke_with_retry(chain, params, max_retries=3):
    """Invoke LLM chain with retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(params)
        except Exception as e:
            error_str = str(e)
            if 'rate_limit' in error_str.lower() or '413' in error_str or '429' in error_str:
                wait_time = 15 * (attempt + 1)
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")

st.info("⚠️ 정확한 분석을 위해 모든 문서는 **PDF 형식**으로 변환하여 업로드해 주세요.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. 금년도 공고 자료 (필수)")
    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)
    current_rfp = st.file_uploader("올해 제안요청서 또는 과업지시서", type=["pdf"], key="curr_rfp")

with col2:
    st.subheader("2. 직전 연도 공고 자료 (선택)")
    no_prev_rfp = st.checkbox("직전 제안요청서 없음", key="chk_no_prev_rfp")
    prev_rfp = st.file_uploader("직전 년도 제안요청서 또는 과업지시서", type=["pdf"], disabled=no_prev_rfp, key="prev_rfp")

# --- Conditional: Show analysis button only for logged-in & approved users ---
is_logged_in = st.session_state.user is not None
is_approved = is_logged_in and st.session_state.user.get('approved', False)

if not is_logged_in:
    st.warning("🔒 분석 기능을 이용하려면 좌측 사이드바에서 **로그인**해 주세요.")
    start_analysis = False
elif not is_approved:
    st.warning("⏳ 계정 승인 대기 중입니다. 관리자 승인 후 분석 기능을 이용할 수 있습니다.")
    start_analysis = False
else:
    start_analysis = st.button("제안요청서 분석 시작 🚀", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 5. Analysis Logic (only for approved users)
# -----------------------------------------------------------------------------
if start_analysis:
    if not api_key:
        st.error("분석을 시작하려면 API Key 설정이 필요합니다.")
        st.stop()
    if not current_rfp:
        st.error("올해 제안요청서는 필수 업로드 항목입니다.")
        st.stop()

    with st.spinner("문서를 분석 중입니다..."):
        full_current_text = extract_text_from_pdf(current_rfp)
        
        prev_text = ""
        if not no_prev_rfp and prev_rfp:
            prev_text = extract_text_from_pdf(prev_rfp)
        
        top_keywords = analyze_keywords(full_current_text)

    # --- Diagnostics for user ---
    curr_len = len(full_current_text.strip())
    if curr_len < 200:
        st.error(f"⚠️ **문서 텍스트 추출 부족 (현재 {curr_len}자)**")
        st.info("문서에서 텍스트를 거의 추출하지 못했습니다. 스캔된 이미지 PDF이거나 파일에 문제가 있을 수 있습니다. 텍스트가 포함된 PDF인지 확인해 주세요.")
        with st.expander("추출된 텍스트 확인"):
            st.code(full_current_text[:1000])
        st.stop()
    else:
        st.success(f"✅ 텍스트 추출 완료! (총 {curr_len}자)")
        with st.expander("추출된 텍스트 일부 확인"):
            st.code(full_current_text[:1000] + "...")

    try:
        # Use a more robust model name to avoid 404 errors
        MODEL_NAME = "gemini-1.5-pro"
        llm = ChatGoogleGenerativeAI(temperature=0.0, model=MODEL_NAME, google_api_key=api_key)

        has_prev = bool(prev_text.strip())

        tabs = st.tabs(["📋 제안요청서 분석 결과", "📊 키워드 인사이트"])
        
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}

        # =====================================================================
        # Unified Analysis logic
        # =====================================================================
        with tabs[0]:
            st.header("📋 제안요청서 분석 결과")
            with st.spinner("전문가 모드로 제안요청서를 정밀 분석 중입니다..."):
                try:
                    # Stricter Sys Prompt to prevent placeholders
                    sys_prompt = f"""
# Role Definition
당신은 공공기관 입찰 전략 컨설턴트이자 20년 경력의 수석 리서치 연구원입니다. 
당신의 임무는 절대적으로 제공된 [금년도 문서]의 텍스트를 기반으로 분석을 수행하는 것입니다.

# [CRITICAL RULE] NO HALLUCINATIONS
1. **절대로** 문서에 없는 정보를 지어내지 마세요 (예: 1억원, 2025~2026년, 리서치 연구원 등 generic한 수치는 금지).
2. 제공된 텍스트에서 정보를 찾을 수 없는 항목은 반드시 **"명시되지 않음"** 또는 **"확인 불가"**라고 작성하세요.
3. 당신의 개인적인 지식이나 일반적인 입찰 정보를 섞지 말고, 오직 업로드된 문서 속의 사실(Fact)만 추출하세요.

# Analysis Instructions
아래 5가지 섹션에 맞춰 분석 결과를 출력하세요. 각 항목은 구체적인 근거(조항, 페이지 등)가 있다면 함께 언급하세요.

## 1. 제안요청서 핵심 비교 및 전략 (RFP Analysis)
*금년도와 직전 연도 정보를 비교하되, 직전 자료가 없으면 '정보 없음'으로 표기하고 금년도 자료를 심층 분석하세요. 4열 표 형식을 유지하세요.*

| 구분 | 2025년(금년) 요구사항 | 2024년(직전) 요구사항 | 변경 내용 및 전략적 해설 |
| :-- | :--- | :--- | :--- |
| **사업 예산 및 기간** | | | |
| **조사 대상 및 표본** | | | |
| **조사 방법** | | | |
| **품질 및 검증 관리** | | | |
| **필수 인력 요건** | | | |
| **성과품 및 활용** | | | |

## 2. 배점표 기반 승부처 분석 (Scoring Strategy)
**문서 내 '배점표'를 찾아 가장 배점이 높거나 중요한 요건 3가지를 도출하세요. 배점표가 없으면 '배점표 확인 불가'로 표기하세요.**

## 3. 과업 내용 기반 필수 수행 체크리스트 (Must-Do List)
**과업지시서에 명시된 '반드시 수행해야 할 과업'들을 추출하세요.**

## 4. 행정 서류 및 제안서 규격 체크리스트 (Administrative Check)
**제출 서류, 제안서 분량, 익명성 처리 규칙 등을 정리하세요.**

## 5. 상세 전략 및 가점 요인 (Bonus Strategy)
**정량평가, 가점 항목, 제안 목차 요구사항을 정리하세요.**

---
# Mandatory Rules
- **Fact-Only:** 제공된 텍스트에 있는 내용만 사용.
- **Strict Citation:** 가능한 경우 조항이나 제목을 인용.
- **No Fillers:** "1억원" 같은 가짜 데이터를 채우지 말 것.
"""
                    # Use a balanced slice of the text (Beginning and End are often most important)
                    def get_balanced_context(text, max_chars=30000):
                        if len(text) <= max_chars:
                            return text
                        half = max_chars // 2
                        return text[:half] + "\n\n... (중략) ...\n\n" + text[-half:]

                    current_context = get_balanced_context(full_current_text, 30000)
                    prev_context = get_balanced_context(prev_text, 10000) if prev_text else "없음"
                    
                    user_content = f"[금년도 문서]\n{current_context}\n\n[직전 연도 문서]\n{prev_context}"
                    
                    prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "{text}")])
                    chain = prompt | llm | StrOutputParser()
                    
                    # Run consolidated analysis
                    response = invoke_with_retry(chain, {"text": user_content})
                    st.markdown(response)
                    st.session_state.analysis_results["제안요청서 분석 결과"] = response
                    
                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")

        # =====================================================================
        # Tab 2: Keyword Insight (Moved to end)
        # =====================================================================
        with tabs[1]:
            st.header("📊 키워드 인사이트")
            chart = create_word_chart(top_keywords)
            if chart: st.pyplot(chart)
            with st.spinner("핵심 키워드 기반 사업 요약 중..."):
                try:
                    prompt = ChatPromptTemplate.from_template(
                        "당신은 공공기관 입찰 전문가입니다. "
                        "아래 제안요청서의 상위 핵심 키워드를 분석하여 다음을 마크다운 표로 정리하세요.\n\n"
                        "| 구분 | 내용 |\n|---|---|\n"
                        "| 사업명(추정) | ... |\n"
                        "| 발주 기관(추정) | ... |\n"
                        "| 핵심 주제 | 1~2문장 요약 |\n"
                        "| 주요 키워드 군집 | 관련 키워드 그룹핑 |\n"
                        "| 사업 유형 | 연구용역/시스템개발/조사사업 등 |\n\n"
                        "표 외에 다른 형식(불릿, 번호 목록 등)은 절대 사용하지 마세요. "
                        "반드시 한국어로만 작성하세요. 키워드: {keywords}"
                    )
                    chain = prompt | llm | StrOutputParser()
                    insight = invoke_with_retry(chain, {"keywords": str(top_keywords)})
                    st.markdown(insight)
                    st.session_state.analysis_results["키워드 인사이트"] = f"Top Keywords: {str(top_keywords)}\n\n{insight}"
                except Exception as e:
                    st.error(f"키워드 분석 중 오류: {e}")

        # Download Button
        st.markdown("---")
        import report_utils
        if st.session_state.analysis_results:
            docx_file = report_utils.generate_word_report(st.session_state.analysis_results)
            st.download_button(
                label="📥 분석 결과 워드 파일 다운로드",
                data=docx_file,
                file_name="win_strategy_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"AI 분석 중 오류가 발생했습니다: {e}")

st.markdown('<div class="footer">Developed by ㅈㅅㅎ | Powered by Streamlit & Google Gemini 1.5 Pro</div>', unsafe_allow_html=True)
