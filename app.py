
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
    current_rfp = st.file_uploader("올해 제안요청서 업로드", type=["pdf"], key="curr_rfp")
    no_task_desc = st.checkbox("과업지시서 없음 (제안요청서 내 포함)", key="chk_no_task")
    current_task = st.file_uploader("올해 과업지시서 업로드", type=["pdf"], disabled=no_task_desc, key="curr_task")

with col2:
    st.subheader("2. 직전 연도 공고 자료 (선택)")
    no_prev_rfp = st.checkbox("직전 제안요청서 없음", key="chk_no_prev_rfp")
    prev_rfp = st.file_uploader("직전 연도 제안요청서 업로드", type=["pdf"], disabled=no_prev_rfp, key="prev_rfp")
    no_prev_task = st.checkbox("직전 과업지시서 없음", key="chk_no_prev_task")
    prev_task = st.file_uploader("직전 연도 과업지시서 업로드", type=["pdf"], disabled=no_prev_task, key="prev_task")

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
        curr_rfp_text = extract_text_from_pdf(current_rfp)
        curr_task_text = ""
        if not no_task_desc and current_task:
            curr_task_text = extract_text_from_pdf(current_task)
        full_current_text = curr_rfp_text + "\n" + curr_task_text
        
        prev_text = ""
        if not no_prev_rfp and prev_rfp:
            prev_text += extract_text_from_pdf(prev_rfp) + "\n"
        if not no_prev_task and prev_task:
            prev_text += extract_text_from_pdf(prev_task)
        
        top_keywords = analyze_keywords(full_current_text)

    st.success("텍스트 추출 완료! AI 분석을 시작합니다.")

    try:
        MODEL_NAME = "llama-3.3-70b-versatile"
        llm = ChatGroq(temperature=0.0, model=MODEL_NAME, api_key=api_key)

        tabs = st.tabs(["📊 키워드 인사이트", "🔄 직전 문서 비교", "🔬 조사설계", "📐 표본설계", "📋 필수 제안 항목", "📄 준비서류", "✅ 목차 체크리스트", "🎯 상세 전략"])
        
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}

        # =====================================================================
        # Tab 1: Keyword Insight
        # =====================================================================
        with tabs[0]:
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
                    insight = insight.replace("<br>", " ").replace("<br/>", " ")
                    st.markdown(insight)
                    st.session_state.analysis_results["키워드 인사이트"] = f"Top Keywords: {str(top_keywords)}\n\n{insight}"
                except Exception as e:
                    st.error(f"키워드 분석 중 오류: {e}")

        time.sleep(5)  # Delay to avoid TPM rate limit

        # =====================================================================
        # Tab 2: Previous Document Comparison
        # =====================================================================
        with tabs[1]:
            st.header("🔄 직전 제안요청서 비교")
            if not prev_text.strip():
                st.warning("직전 연도 자료가 없어 비교 분석을 생략합니다.")
                st.session_state.analysis_results["직전 제안요청서 비교"] = "비교 데이터 없음"
            else:
                with st.spinner("직전 연도와 비교 분석 중..."):
                    try:
                        prompt = ChatPromptTemplate.from_template(
                            "당신은 공공기관 입찰 전문가입니다. "
                            "아래의 [직전 연도]와 [금년도] 제안요청서를 비교 분석하세요.\n\n"
                            "**반드시 아래 마크다운 표 형식만 사용하세요. 표 외에 불릿 목록이나 텍스트 설명은 절대 추가하지 마세요.**\n\n"
                            "| 비교 항목 | 직전 연도 | 금년도 | 변경 내용 요약 |\n"
                            "|---|---|---|---|\n"
                            "| 사업 예산 | | | |\n"
                            "| 사업 기간 | | | |\n"
                            "| 표본 규모 | | | |\n"
                            "| 조사 방법론 | | | |\n"
                            "| 평가 기준/배점 | | | |\n"
                            "| 참가 자격 요건 | | | |\n"
                            "| 납품 성과물 | | | |\n"
                            "| 기타 주요 변경 | | | |\n\n"
                            "문서에 명시되지 않은 항목은 '명시 없음'. "
                            "반드시 한국어로만 작성.\n\n"
                            "[직전 연도]\n{prev_text}\n\n[금년도]\n{curr_text}"
                        )
                        chain = prompt | llm | StrOutputParser()
                        res = invoke_with_retry(chain, {"prev_text": prev_text[:3000], "curr_text": full_current_text[:3000]})
                        res = res.replace("<br>", " ").replace("<br/>", " ")
                        st.markdown(res)
                        st.session_state.analysis_results["직전 제안요청서 비교"] = res
                    except Exception as e:
                        st.error(f"비교 분석 중 오류: {e}")

        # =====================================================================
        # Tab 3~8: Structured RFP Analysis
        # =====================================================================
        def run_analysis(tab_name, instructions, keywords, context_text, target_tab):
            with target_tab:
                st.header(tab_name)
                relevant_text = get_relevant_context(context_text, keywords)
                with st.spinner(f"{tab_name} 분석 중..."):
                    try:
                        time.sleep(10)  # Delay between calls to avoid TPM rate limit
                        sys_prompt = (
                            "당신은 공공기관 입찰 및 제안요청서(RFP) 전문 분석가입니다.\n"
                            "다음 규칙을 반드시 준수하세요:\n"
                            "1. 문서에 있는 '사실(Fact)'만을 추출하여 정리하세요.\n"
                            "2. 문서에 명시되지 않은 도구, 기술, 방법론, 의견은 절대로 추가하지 마세요.\n"
                            "3. 내용이 없으면 '해당 내용 없음'으로 표기하세요.\n"
                            "4. 반드시 마크다운 표(table) 형식만 사용하세요. 표 외에 불릿 목록이나 텍스트 설명은 추가하지 마세요.\n"
                            "5. HTML 태그를 사용하지 말고 마크다운만 사용하세요.\n"
                            "6. 반드시 자연스러운 '한국어'로만 작성하세요.\n"
                            "7. 영어, 중국어, 일본어, 아랍어가 절대 포함되지 않게 하세요.\n\n"
                            f"[분석 지시사항]\n{instructions}"
                        )
                        prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "{text}")])
                        chain = prompt | llm | StrOutputParser()
                        response = invoke_with_retry(chain, {"text": relevant_text})
                        response = response.replace("<br>", " ").replace("<br/>", " ")
                        st.markdown(response)
                        st.session_state.analysis_results[tab_name] = response
                    except Exception as e:
                        st.error(f"{tab_name} 분석 중 오류가 발생했습니다: {e}")

        # --- Tab 3: 조사설계 ---
        run_analysis(
            "🔬 조사설계",
            "제안요청서에서 조사설계 관련 내용을 추출하여 아래 표로 정리하세요:\n\n"
            "| 구분 | 내용 |\n|---|---|\n"
            "| 사업 목적 | |\n"
            "| 사업 개요 | |\n"
            "| 사업 기간 | |\n"
            "| 사업 예산(총 예산/부가세 포함 여부) | |\n"
            "| 주요 과업 내용 | |\n"
            "| 조사 대상/범위 | |\n"
            "| 조사 방법(정량/정성) | |\n"
            "| 예비조사 여부 | |\n"
            "| 납품 성과물 목록 | |\n"
            "| 기타 특이사항 | |\n\n"
            "각 항목은 문서에 명시된 내용만 기재하고, 없으면 '해당 내용 없음'으로 표기하세요.",
            ["조사 개요", "과업 내용", "과업 목적", "과업 범위", "수행 내용", "사업 목적", "사업 개요", "사업 기간", "사업 예산", "조사 대상", "조사 방법"],
            full_current_text, tabs[2]
        )

        # --- Tab 4: 표본설계 ---
        run_analysis(
            "📐 표본설계",
            "제안요청서에서 표본설계 관련 내용을 추출하여 아래 표로 정리하세요:\n\n"
            "| 구분 | 내용 |\n|---|---|\n"
            "| 모집단 정의 | |\n"
            "| 목표 표본 크기 | |\n"
            "| 표본 추출 방법(확률/비확률) | |\n"
            "| 층화 기준(지역/성별/연령 등) | |\n"
            "| 할당 방법 | |\n"
            "| 표본 오차 한계 | |\n"
            "| 신뢰 수준 | |\n"
            "| 가중치 적용 방법 | |\n"
            "| 조사 도구(설문지/면접 등) | |\n"
            "| 조사 기간 | |\n"
            "| 기타 특이사항 | |\n\n"
            "각 항목은 문서에 명시된 수치와 방법만 기재하세요.",
            ["표본", "모집단", "오차", "신뢰 수준", "추출 방법", "할당", "층화", "가중치", "표본 크기", "표본 설계"],
            full_current_text, tabs[3]
        )

        # --- Tab 5: 필수 제안 항목 ---
        run_analysis(
            "📋 필수 제안 항목",
            "제안요청서에 명시된 필수 수행 활동과 성과품을 추출하여 아래 표들로 정리하세요:\n\n"
            "**[필수 수행 활동]**\n\n"
            "| 번호 | 수행 활동 | 세부 내용 | 비고 |\n|---|---|---|---|\n"
            "| 1 | | | |\n\n"
            "**[납품 성과물]**\n\n"
            "| 번호 | 성과물 명칭 | 규격/형식 | 수량 | 제출 시기 |\n|---|---|---|---|---|\n"
            "| 1 | | | | |\n\n"
            "**[보고 체계]**\n\n"
            "| 보고 유형 | 시기 | 내용 | 비고 |\n|---|---|---|---|\n"
            "| 착수보고 | | | |\n"
            "| 중간보고 | | | |\n"
            "| 최종보고 | | | |\n\n"
            "문서에 명시된 내용만 기재하세요.",
            ["제안 요구사항", "수행 지침", "착수 보고", "중간 보고", "최종 보고", "성과품", "납품", "성과물", "제출물", "보고서"],
            full_current_text, tabs[4]
        )

        # --- Tab 6: 준비서류 ---
        run_analysis(
            "📄 준비서류",
            "제안요청서에서 입찰 참가 자격과 제출 서류를 추출하여 아래 표들로 정리하세요:\n\n"
            "**[입찰 참가 자격 요건]**\n\n"
            "| 번호 | 자격 요건 | 세부 조건 |\n|---|---|---|\n"
            "| 1 | | |\n\n"
            "**[제출 서류 목록]**\n\n"
            "| 번호 | 서류명 | 부수 | 제출 형식 | 비고 |\n|---|---|---|---|---|\n"
            "| 1 | | | | |\n\n"
            "**[입찰 일정]**\n\n"
            "| 일정 항목 | 일시 | 장소/방법 |\n|---|---|---|\n"
            "| 입찰 공고일 | | |\n"
            "| 제안서 제출 마감 | | |\n"
            "| 기술 평가 | | |\n"
            "| 낙찰자 발표 | | |\n\n"
            "문서에 명시된 내용만 기재하세요.",
            ["참가 자격", "제출 서류", "입찰 보증금", "실적 증명", "사업자 등록", "입찰 일정", "제안서 제출", "참가 등록", "적격 심사"],
            full_current_text, tabs[5]
        )

        # --- Tab 7: 목차 체크리스트 ---
        run_analysis(
            "✅ 목차 체크리스트",
            "제안요청서에 제시된 제안서 목차 구성이나 평가 항목을 추출하여 아래 표로 정리하세요:\n\n"
            "**[제안서 평가 항목 및 배점]**\n\n"
            "| 평가 영역 | 평가 항목 | 배점 | 주요 평가 내용 |\n|---|---|---|---|\n"
            "| 기술 능력 | | | |\n"
            "| 수행 실적 | | | |\n"
            "| 사업 이해도 | | | |\n"
            "| 수행 방법론 | | | |\n"
            "| 프로젝트 관리 | | | |\n\n"
            "**[권장 목차 구성]**\n\n"
            "| 번호 | 목차 항목 | 대응 평가 항목 | 권장 분량 |\n|---|---|---|---|\n"
            "| 1 | | | |\n\n"
            "문서에 명시된 평가 기준과 배점만 기재하세요.",
            ["제안서 목차", "평가 항목", "배점 한도", "작성 지침", "평가 기준", "기술 평가", "배점", "평가 위원", "심사 기준"],
            full_current_text, tabs[6]
        )

        # --- Tab 8: 상세 전략 ---
        run_analysis(
            "🎯 상세 전략",
            "제안요청서에서 제안 전략 수립에 필요한 핵심 정보를 추출하여 아래 표들로 정리하세요:\n\n"
            "**[정량 평가 기준]**\n\n"
            "| 평가 항목 | 기준 | 배점 | 비고 |\n|---|---|---|---|\n"
            "| 1 | | | |\n\n"
            "**[수행 인력 요건]**\n\n"
            "| 구분 | 자격 요건 | 인원 | 투입 기간 | 비고 |\n|---|---|---|---|---|\n"
            "| 사업책임자(PM) | | | | |\n"
            "| 연구원 | | | | |\n\n"
            "**[데이터 품질/보안 요건]**\n\n"
            "| 구분 | 요구사항 | 세부 내용 |\n|---|---|---|\n"
            "| 데이터 품질 | | |\n"
            "| 보안 대책 | | |\n"
            "| 개인정보 보호 | | |\n\n"
            "**[사후 관리/유지보수]**\n\n"
            "| 구분 | 내용 | 기간 |\n|---|---|---|\n"
            "| 하자 보수 | | |\n"
            "| 유지보수 | | |\n"
            "| 기술 지원 | | |\n\n"
            "문서에 명시된 팩트만 기재하세요.",
            ["정량 평가", "수행 인력", "참여 인력", "데이터 품질", "보안 대책", "사후 지원", "유지 보수", "하자 보수", "개인정보", "인력 자격"],
            full_current_text, tabs[7]
        )

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
