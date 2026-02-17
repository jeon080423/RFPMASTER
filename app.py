
import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import os
from kiwipiepy import Kiwi
from langchain_groq import ChatGroq
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
    api_key = st.secrets.get("groq", {}).get("api_key", "")
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    
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

    st.success("텍스트 추출 완료! AI 분석을 시작합니다.")

    try:
        MODEL_NAME = "llama-3.3-70b-versatile"
        llm = ChatGroq(temperature=0.0, model=MODEL_NAME, api_key=api_key)

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
                    # Combined Sys Prompt based on User Request
                    sys_prompt = f"""
# Role Definition
당신은 공공기관 입찰 전략 컨설턴트이자 20년 경력의 수석 리서치 연구원입니다. 
사용자가 업로드한 [금년도 제안요청서(과업지시서)]와 [직전 연도 자료(선택사항)]의 텍스트를 정밀 분석하여, 경쟁 우위를 점할 수 있는 제안 전략과 필수 점검 사항을 도출해 주세요.

# Analysis Instructions
아래 5가지 섹션에 맞춰 분석 결과를 출력하세요. 
**[경고] 반드시 업로드된 문서의 텍스트에 기반하여 작성하세요. 문서에서 확인되지 않는 정보(예산, 기간, 인원 등)를 임의로 지어내지 마세요. 확인되지 않는 정보는 반드시 "확인 불가" 또는 "명시되지 않음"으로 표기하세요.**
각 항목은 가능한 경우 구체적인 근거(페이지 또는 조항)를 함께 언급하세요.

## 1. 제안요청서 핵심 비교 및 전략 (RFP Analysis)
**지침:** 직전 자료가 없을 경우 3열은 '정보 없음', 4열은 '금년도 과업 중심 분석'으로 채우되, **4열 표 형식을 반드시 유지하세요.**

| 구분 | 2025년 주요 요구사항 (금년) | 2024년 요구사항 (직전/비고) | 변경 내용 및 전략적 해설 |
| :-- | :--- | :--- | :--- |
| **사업 예산 및 기간** | | | |
| **조사 대상 및 표본** | | | |
| **조사 방법** | | | |
| **품질 및 검증 관리** | | | |
| **필수 인력 요건** | | | |
| **성과품 및 활용** | | | |

## 2. 배점표 기반 승부처 분석 (Scoring Strategy)
**제안요청서 내 '평가항목 및 배점표'를 분석하여 가장 점수 비중이 높거나 까다로운 '핵심 승부처' 3가지를 도출하세요.** 
*배점표가 확인되지 않으면 '배점표 확인 불가'로 표기하세요.*

## 3. 과업 내용 기반 필수 수행 체크리스트 (Must-Do List)
**과업지시서 텍스트에서 '반드시 수행해야 할 과업'을 추출하여 체크리스트로 정리하세요.**

## 4. 행정 서류 및 제안서 규격 체크리스트 (Administrative Check)
**입찰 공고와 작성 지침에 명시된 필수 서류와 규격(분량, 익명성 등)을 정리하세요.**

## 5. 상세 전략 및 가점 요인 (Bonus Strategy)
**문서에서 확인되는 정량평가 기준, 가점 항목, 제안 목차 요구사항을 정리하세요.**

---

# Mandatory Rules for AI
1. **Fact-Only:** 제공된 [금년도 문서]와 [직전 연도 문서] 텍스트에 있는 내용만 사용하세요.
2. **No Hallucinations:** 문서에 없는 사업명, 예산, 날짜, 인원수 등을 절대로 상상해서 적지 마세요. 모르는 정보는 빈칸으로 두지 말고 "명시되지 않음"이라고 명확히 적으세요.
3. **Citations:** 가능한 경우 정보의 근거가 되는 조항이나 맥락을 덧붙여 신뢰도를 높이세요.
4. **Fallback:** 직전 연도 자료가 없으면 비교 대신 금년도 자료의 상세 내용을 바탕으로 분석을 수행하세요.
"""
                    # Increase context limit to capture more of the PDF
                    user_content = f"[금년도 문서]\n{full_current_text[:30000]}\n\n[직전 연도 문서]\n{prev_text[:15000] if prev_text else '없음'}"
                    
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

st.markdown('<div class="footer">Developed by ㅈㅅㅎ | Powered by Streamlit & Groq Llama3</div>', unsafe_allow_html=True)
