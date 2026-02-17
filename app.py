
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

        has_prev = bool(prev_text.strip())

        tabs = st.tabs(["📊 키워드 인사이트", "🔬 조사/분석 설계", "📐 표본/품질 관리", "� 사업 관리", "📄 준비서류", "✅ 목차 체크리스트", "🎯 상세 전략"])
        
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}

        # =====================================================================
        # Common: run_analysis (no comparison)
        # =====================================================================
        def run_analysis(tab_name, instructions, keywords, context_text, target_tab):
            with target_tab:
                st.header(tab_name)
                relevant_text = get_relevant_context(context_text, keywords)
                with st.spinner(f"{tab_name} 분석 중..."):
                    try:
                        time.sleep(3)
                        # st.toast("API 호출 준비 중... (잠시 대기)", icon="⏳")
                        sys_prompt = (
                            "당신은 공공기관 입찰 및 제안요청서(RFP) 전문 분석가입니다.\n"
                            "다음 규칙을 반드시 준수하세요:\n"
                            "1. 문서에 있는 '사실(Fact)'만을 추출하여 정리하세요.\n"
                            "2. 문서에 명시되지 않은 도구, 기술, 방법론, 의견은 절대로 추가하지 마세요.\n"
                            "3. 내용이 없으면 '해당 내용 없음'으로 표기하세요.\n"
                            "4. 반드시 마크다운 표(table) 형식만 사용하세요. 표 외에 불릿 목록이나 텍스트 설명은 추가하지 마세요.\n"
                            "5. HTML 태그를 사용하지 말고 마크다운만 사용하세요.\n"
                            "6. 반드시 자연스러운 '한국어'로만 작성하세요.\n"
                            "7. 영어, 중국어, 일본어, 아랍어가 절대 포함되지 않게 하세요.\n"
                            "8. 발주기관마다 용어가 다를 수 있으므로, 유사한 의미의 용어는 같은 항목으로 분류하세요. "
                            "예: '과업범위'='사업범위'='수행범위', '모집단'='조사대상', '사업비'='예산'='용역비' 등\n\n"
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

        # =====================================================================
        # Common: run_comparison_analysis
        # Refactored: Always show Current Analysis (List Format) + Comparison (Table if exists)
        # =====================================================================
        def run_comparison_analysis(tab_name, instructions_current, instructions_compare, keywords, current_text, previous_text, target_tab):
            """Analyze current doc (New List Format), then compare with previous if available."""
            with target_tab:
                st.header(tab_name)
                relevant_current = get_relevant_context(current_text, keywords)
                
                # 1. Current Analysis (Senior Researcher Persona, List Format)
                with st.spinner(f"{tab_name} - 금년도 핵심 요건 분석 중..."):
                    try:
                        time.sleep(3)
                        # st.toast("API 호출 준비 중... (잠시 대기)", icon="⏳")
                        # New Persona and Goal from User Request
                        sys_prompt = (
                            "당신은 공공기관 및 정부 부처의 조사·연구 용역 제안서를 작성하는 수석 연구원(Senior Researcher)입니다.\n"
                            "업로드된 제안요청서의 텍스트를 분석하여, 제안서 본문에 기술해야 할 핵심 과업 요건을 체계적으로 정리해 주세요.\n\n"
                            "# Analysis Goal\n"
                            "제안서의 '수행 계획' 파트를 작성하기 위해, 발주처가 요구하는 구체적인 조사 방법, 절차, 분석 수준, 관리 요건을 빠짐없이 추출하세요.\n\n"
                            "# Output Rule\n"
                            "1. 반드시 마크다운 불릿 리스트(Bulleted List) 형식을 사용하세요. (표 사용 지양)\n"
                            "2. 각 항목의 핵심 내용은 **굵게(Bold)** 표시하세요.\n"
                            "3. 문서에 명시된 내용이 있으면 구체적인 수치나 조건을 적고, 해당 내용이 발견된 페이지나 문맥이 있다면 [출처]를 간략히 표기하세요.\n"
                            "4. 내용이 없으면 '명시되지 않음'으로 적으세요.\n"
                            "5. 반드시 한국어로 작성하세요.\n\n"
                            f"# Analysis Scope (Categories to Extract)\n{instructions_current}"
                        )
                        prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "{text}")])
                        chain = prompt | llm | StrOutputParser()
                        response = invoke_with_retry(chain, {"text": relevant_current})
                        response = response.replace("<br>", " ").replace("<br/>", " ")
                        
                        st.subheader("� 금년도 핵심 과업 요건")
                        st.markdown(response)
                        st.session_state.analysis_results[tab_name] = response
                    except Exception as e:
                        st.error(f"{tab_name} 분석 중 오류: {e}")

                # 2. Previous Comparison (Table Format, Only if prev exists)
                if previous_text.strip():
                    relevant_prev = get_relevant_context(previous_text, keywords)
                    with st.spinner(f"{tab_name} - 직전 연도 대비 비교 분석 중..."):
                        try:
                            time.sleep(5)
                            compare_prompt = ChatPromptTemplate.from_template(
                                "당신은 공공기관 입찰 전문 분석가입니다.\n"
                                "아래 [직전 연도]와 [금년도] 문서를 비교하여 변경사항을 분석하세요.\n\n"
                                "**핵심 규칙:**\n"
                                "- 반드시 마크다운 표(Table) 형식만 사용하세요.\n"
                                "- 발주기관마다 용어가 다를 수 있으니 유연하게 비교하세요.\n"
                                "- '비고' 열에는 변경이 수주 전략에 미치는 시사점을 요약하세요.\n\n"
                                f"{instructions_compare}\n\n"
                                "[직전 연도 문서]\n{prev_text}\n\n[금년도 문서]\n{curr_text}"
                            )
                            chain = compare_prompt | llm | StrOutputParser()
                            compare_res = invoke_with_retry(chain, {"prev_text": relevant_prev, "curr_text": relevant_current})
                            compare_res = compare_res.replace("<br>", " ").replace("<br/>", " ")
                            
                            st.subheader("� 직전 연도 대비 변경 사항")
                            st.markdown(compare_res)
                            st.session_state.analysis_results[f"{tab_name} (비교)"] = compare_res
                        except Exception as e:
                            st.error(f"{tab_name} 비교 분석 중 오류: {e}")

        # =====================================================================
        # Common: run_task_comparison_analysis (Specific for Tab 4)
        # =====================================================================
        def run_task_comparison_analysis(tab_name, instructions_current, keywords, current_text, previous_text, target_tab):
            """Specific comparison for Task Description (Scope, Content, Cautions)."""
            with target_tab:
                st.header(tab_name)
                relevant_current = get_relevant_context(current_text, keywords)
                
                # 1. Current Task Analysis
                with st.spinner(f"{tab_name} - 금년도 과업 분석 중..."):
                    try:
                        time.sleep(3)
                        # st.toast("API 호출 준비 중... (잠시 대기)", icon="⏳")
                        sys_prompt = (
                            "당신은 공공기관 과업지시서 전문 분석가입니다.\n"
                            "제안서 작성에 필수적인 과업 수행 요건을 아래 기준에 맞춰 추출하세요.\n\n"
                            "# Output Rule\n"
                            "- 마크다운 리스트 형식을 권장합니다.\n"
                            "- 문서에 명시된 '필수' 사항을 빠짐없이 기록하세요.\n\n"
                            f"{instructions_current}"
                        )
                        prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "{text}")])
                        chain = prompt | llm | StrOutputParser()
                        response = invoke_with_retry(chain, {"text": relevant_current})
                        st.subheader("� 금년도 필수 과업 요건")
                        st.markdown(response)
                        st.session_state.analysis_results[tab_name] = response
                    except Exception as e:
                        st.error(f"{tab_name} 분석 중 오류: {e}")

                # 2. Comparison Analysis
                if previous_text.strip():
                    relevant_prev = get_relevant_context(previous_text, keywords)
                    with st.spinner(f"{tab_name} - 과업지시서 비교 분석 중..."):
                        try:
                            time.sleep(5)
                            compare_prompt = ChatPromptTemplate.from_template(
                                "당신은 과업지시서 비교 분석 전문가입니다.\n"
                                "아래 [직전 과업지시서]와 [금년도 과업지시서]를 비교하여,\n"
                                "**조사 범위, 과업 내용, 주의사항(특이사항)** 측면에서 상세히 비교하세요.\n\n"
                                "**핵심 규칙:**\n"
                                "- 반드시 마크다운 표 형식만 사용하세요.\n"
                                "- '비고' 열에는 변경사항 대응 전략을 기술하세요.\n\n"
                                "**[비교 항목]**\n"
                                "| 구분 | 직전 과업지시서 | 금년도 과업지시서 | 비고(제안 전략) |\n"
                                "|---|---|---|---|\n"
                                "| 조사/과업 범위 | | | |\n"
                                "| 주요 과업 내용 | | | |\n"
                                "| 수행 시 주의사항 | | | |\n"
                                "| 기타 변경사항 | | | |\n\n"
                                "[직전 과업지시서]\n{prev_text}\n\n[금년도 과업지시서]\n{curr_text}"
                            )
                            chain = compare_prompt | llm | StrOutputParser()
                            compare_res = invoke_with_retry(chain, {"prev_text": relevant_prev, "curr_text": relevant_current})
                            
                            st.subheader("� 과업지시서 변경 사항 비교")
                            st.markdown(compare_res)
                            st.session_state.analysis_results[f"{tab_name} (과업 비교)"] = compare_res
                        except Exception as e:
                            st.error(f"{tab_name} 과업 비교 분석 중 오류: {e}")

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
        
        time.sleep(2)

        # =====================================================================
        # Tab 2: 조사/분석 설계 (Categories 1 & 3 & 5)
        # =====================================================================
        run_comparison_analysis(
            "🔬 조사/분석 설계",
            # Current analysis instructions (New List Categories)
            "다음 카테고리 내용을 추출하세요:\n"
            "## 1. 조사 설계 (Research Design)\n"
            "* 조사 방법 (구체적 방식, 온/오프라인 등)\n"
            "* 조사 대상 및 범위 (모집단 정의, 유효 표본 수)\n"
            "* 조사 지역\n\n"
            "## 3. 조사 내용 및 도구 (Instruments)\n"
            "* 설문지 구성 요건 (시계열 유지, 신규 개발 등)\n"
            "* 사전 조사(Pilot Test) 요건\n"
            "* 전문가 자문 및 검토 요건\n\n"
            "## 5. 데이터 분석 및 활용 (Analysis)\n"
            "* 필수 분석 기법 (빈도/교차 분석, 가중치 산출, 고급 통계 등)\n"
            "* 결과 활용 방안 (인포그래픽, 정책 제언 등)",
            # Comparison instructions
            "아래 표 형식으로 '조사 설계' 및 '분석 요건' 관련 변경사항을 비교하세요:\n\n"
            "| 구분 | 직전 연도 | 금년도 | 비고 |\n|---|---|---|---|\n"
            "| 조사 방법 | | | |\n"
            "| 조사 대상/표본 | | | |\n"
            "| 조사 내용/설문 | | | |\n"
            "| 분석 방법 | | | |\n"
            "| 결과 활용 | | | |\n",
            ["조사 개요", "과업 내용", "과업 목적", "과업 범위", "수행 내용", "사업 목적", "사업 개요", "사업 기간", "사업 예산", "조사 대상", "조사 방법", "과업 수행", "세부 과업", "연구 목적", "용역 내용", "분석 방법", "통계"],
            full_current_text, prev_text, tabs[1]
        )

        # =====================================================================
        # Tab 3: 표본/품질 관리 (Categories 2 & 4)
        # =====================================================================
        run_comparison_analysis(
            "📐 표본/품질 관리",
            # Current analysis instructions
            "다음 카테고리 내용을 추출하세요:\n"
            "## 2. 표본 설계 (Sampling Design)\n"
            "* 모집단 및 표본추출틀(Frame)\n"
            "* 표본 추출 방법 (층화/할당/무작위 등 상세)\n"
            "* 층화 기준 및 할당 표\n"
            "* 표본 오차 및 신뢰수준\n\n"
            "## 4. 실사 운영 및 품질 관리 (Fieldwork & QC)\n"
            "* 조사원 운용 (자격 요건, 교육 필수 사항)\n"
            "* 데이터 검증(Verification) 비율 및 방법 (Back-check 등)\n"
            "* 데이터 클리닝 (Logic Check) 및 이상치 처리\n"
            "* 응답률 제고 방안 (답례품, 콜백 기준 등)",
            # Comparison instructions
            "아래 표 형식으로 '표본 설계' 및 '실사 품질관리' 관련 변경사항을 비교하세요:\n\n"
            "| 구분 | 직전 연도 | 금년도 | 비고 |\n|---|---|---|---|\n"
            "| 표본 추출 방법 | | | |\n"
            "| 목표 표본 수 | | | |\n"
            "| 오차/신뢰 수준 | | | |\n"
            "| 조사원 요건 | | | |\n"
            "| 품질관리/검증 | | | |\n",
            ["표본", "모집단", "오차", "신뢰 수준", "추출 방법", "할당", "층화", "가중치", "표본 크기", "표본 설계", "조사 대상", "응답자", "설문", "면접", "조사 인원", "분석 방법", "품질", "검증", "에디팅"],
            full_current_text, prev_text, tabs[2]
        )

        # =====================================================================
        # Tab 4: 사업 관리 (Category 6)
        # =====================================================================
        run_task_comparison_analysis(
            "� 사업 관리",
            "다음 카테고리 내용을 추출하세요:\n"
            "## 6. 사업 관리 및 보고 (Project Management)\n"
            "* 보고 주기 및 체계 (착수/중간/최종/수시)\n"
            "* 일정 관리 (계약일 기준 착수보고, 중간보고 시기 등)\n"
            "* 인력 운영 및 보안 관리 계획\n\n"
            "## 필수 과업 및 성과물\n"
            "* 필수 수행 활동 상세 목록\n"
            "* 최종 납품 성과물 목록 및 형태",
            ["과업 지시", "과업 내용", "수행 사항", "주의 사항", "유의 사항", "특이 사항", "과업 범위", "제안 요구", "세부 과업", "보고", "일정", "관리"],
            full_current_text, prev_text, tabs[3]
        )

        # =====================================================================
        # Tab 5: 준비서류
        # =====================================================================
        run_analysis(
            "📄 준비서류",
            "제안요청서에서 입찰 참가 자격과 제출 서류를 면밀히 추출하여 아래 표들로 정리하세요.\n\n"
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
            ["참가 자격", "제출 서류", "입찰 보증금", "실적 증명", "사업자 등록", "입찰 일정", "제안서 제출", "참가 등록", "적격 심사", "입찰 공고"],
            full_current_text, tabs[4]
        )

        # =====================================================================
        # Tab 6: 목차 체크리스트
        # =====================================================================
        run_analysis(
            "✅ 목차 체크리스트",
            "제안요청서에 제시된 제안서 목차 구성이나 평가 항목을 면밀히 추출하여 아래 표로 정리하세요.\n\n"
            "**[제안서 평가 항목 및 배점]**\n\n"
            "| 평가 영역 | 평가 항목 | 배점 | 주요 평가 내용 |\n|---|---|---|---|\n"
            "| | | | |\n\n"
            "**[권장 목차 구성]**\n\n"
            "| 번호 | 목차 항목 | 대응 평가 항목 | 권장 분량 |\n|---|---|---|---|\n"
            "| 1 | | | |\n\n"
            "문서에 명시된 평가 기준과 배점만 기재하세요.",
            ["제안서 목차", "평가 항목", "배점 한도", "작성 지침", "평가 기준", "기술 평가", "배점", "평가 위원", "심사 기준", "평가표"],
            full_current_text, tabs[5]
        )

        # =====================================================================
        # Tab 7: 상세 전략
        # =====================================================================
        run_analysis(
            "🎯 상세 전략",
            "제안요청서에서 제안 전략 수립에 필요한 핵심 정보를 면밀히 추출하여 아래 표들로 정리하세요.\n\n"
            "**[정량 평가 기준]**\n\n"
            "| 평가 항목 | 기준 | 배점 | 비고 |\n|---|---|---|---|\n"
            "| | | | |\n\n"
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
            ["정량 평가", "수행 인력", "참여 인력", "데이터 품질", "보안 대책", "사후 지원", "유지 보수", "하자 보수", "개인정보", "인력 자격", "책임자", "기술 인력"],
            full_current_text, tabs[6]
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
