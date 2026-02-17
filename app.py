
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
import google.generativeai as genai
import auth
import email_utils

# -----------------------------------------------------------------------------
# 1. Config & Branding
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="수주비책 - RFP 분석 솔루션",
    page_icon="favicon.png",
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
        
        # --- Change Password flow ---
        with st.expander("🛠️ 비밀번호 변경"):
            new_pw_settings = st.text_input("새 비밀번호", type="password", key="settings_new_pw")
            confirm_pw_settings = st.text_input("새 비밀번호 확인", type="password", key="settings_new_pw_chk")
            if st.button("비밀번호 변경 적용", use_container_width=True):
                if new_pw_settings and new_pw_settings == confirm_pw_settings:
                    if auth.update_password(st.session_state.user['email'], new_pw_settings):
                        st.success("비밀번호가 성공적으로 변경되었습니다.")
                    else:
                        st.error("비밀번호 변경에 실패했습니다.")
                else:
                    st.warning("비밀번호가 일치하지 않거나 비어있습니다.")
        
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
            
            st.markdown("---")
            st.subheader("📊 사용자 활동 현황")
            all_users = auth.get_all_users()
            if not all_users.empty:
                # Clean up display (email, name, role, last_login, analysis_count)
                display_df = all_users[['name', 'email', 'role', 'last_login', 'analysis_count']].copy()
                display_df.columns = ['이름/닉네임', '이메일', '구분', '최종로그인', '사용횟수']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("등록된 사용자 없음")
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
            
            # Forgot Password Logic
            with st.expander("🔑 비밀번호를 잊으셨나요?"):
                reset_email = st.text_input("가입한 이메일을 입력하세요", key="reset_email_input")
                if st.button("임시 비밀번호 발급", use_container_width=True):
                    if reset_email:
                        new_pw = auth.reset_password(reset_email)
                        if new_pw:
                            success, msg = email_utils.send_password_reset_email(reset_email, new_pw)
                            if success:
                                st.success("임시 비밀번호가 메일로 발송되었습니다.")
                            else:
                                st.error(f"메일 발송 실패: {msg}")
                        else:
                            st.error("해당 이메일로 가입된 정보를 찾을 수 없습니다.")
                    else:
                        st.warning("이메일을 입력해주세요.")
        
        with signup_tab:
            new_email = st.text_input("이메일", key="signup_email")
            new_name = st.text_input("이름 또는 닉네임", key="signup_name")
            new_password = st.text_input("비밀번호", type="password", key="signup_pw")
            st.caption("🔒 비밀번호는 암호화되어 실시간 보안 저장됩니다.")
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
    st.markdown("""
    <div style='font-size: 0.8rem; color: #666; background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 10px;'>
        💰 <b>후원:</b> 카뱅 3333-23-866708 ㅈㅅㅎ<br>
        유료 API 결제 및 서버 유지비에 소중히 사용됩니다.
    </div>
    """, unsafe_allow_html=True)

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

def get_best_available_model(api_key):
    """Dynamically find the best available model for the given API key."""
    try:
        genai.configure(api_key=api_key)
        # Fetch available models
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority list
        priority = [
            "models/gemini-1.5-pro",
            "models/gemini-1.5-pro-latest",
            "models/gemini-2.0-flash-exp", # 2.0 experimental if available
            "models/gemini-1.5-flash",
            "models/gemini-pro" # legacy
        ]
        
        for p in priority:
            if p in available_models:
                return p.split("/")[-1]
        
        if available_models:
            return available_models[0].split("/")[-1]
    except Exception as e:
        print(f"Model listing error: {e}")
    
    return "gemini-1.5-flash" # Safe fallback

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
    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)
    prev_rfp = st.file_uploader("직전 년도 제안요청서 또는 과업지시서", type=["pdf"], key="prev_rfp_uploader")

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
    # --- Helper to detect year from text ---
    def detect_year(text, default_label):
        # Look for 4 digits followed by '년' in the first 2000 chars (usually cover/title)
        match = re.search(r'20\d{2}년', text[:2000])
        if match:
            return match.group(0)
        return default_label

    def clean_ai_output(text):
        """Forcefully removes <br> tags and replacements with \n."""
        if not text: return ""
        cleaned = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        return cleaned

    if start_analysis:
        if not api_key:
            st.error("분석을 시작하려면 API Key 설정이 필요합니다.")
            st.stop()
        if not current_rfp:
            st.error("올해 제안요청서는 필수 업로드 항목입니다.")
            st.stop()

        # Clear old results for new analysis
        st.session_state.analysis_results = {}

        auth.increment_analysis_count(st.session_state.user['email'])

        with st.spinner("문서를 분석 중입니다..."):
            full_current_text = extract_text_from_pdf(current_rfp)
            
            prev_text = ""
            if prev_rfp:
                prev_text = extract_text_from_pdf(prev_rfp)
            
            top_keywords = analyze_keywords(full_current_text)

        # Detect Years
        curr_year = detect_year(full_current_text, "금년")
        prev_year = detect_year(prev_text, "직전") if prev_text else "없음"

        # --- Diagnostics for user ---
        curr_len = len(full_current_text.strip())
        prev_len = len(prev_text.strip()) if prev_text else 0

        if curr_len < 200:
            st.error(f"⚠️ **금년도 문서 텍스트 추출 부족 (현재 {curr_len}자)**")
            st.info("문서에서 텍스트를 거의 추출하지 못했습니다. 텍스트가 포함된 PDF인지 확인해 주세요.")
            st.stop()
        else:
            prev_msg = f" & 직전 연도 {prev_len}자" if prev_len > 0 else ""
            st.success(f"✅ 텍스트 추출 완료! (금년도 {curr_len}자{prev_msg})")

        try:
            MODEL_NAME = get_best_available_model(api_key)
            st.info(f"✨ 분석 모델: `{MODEL_NAME}` (자동 최적화)")
            llm = ChatGoogleGenerativeAI(temperature=0.0, model=MODEL_NAME, google_api_key=api_key)

            has_prev = bool(prev_text.strip())
            
            # Section 1 ALWAYS appears now. AI handles empty prev info.
            section_1_prompt = f"""
## 1. 제안요청서 핵심 비교 및 전략 (RFP Analysis)
*금년도({curr_year})와 직전 연도({prev_year}) 정보를 비교하세요. 직전 연도 정보가 '없음'인 경우 해당 칸은 '정보 없음'으로 기입하고 금년도 내용을 중점적으로 분석하세요.*

| 구분 | {curr_year} 요구사항 | {prev_year} 요구사항 | 변경 내용 및 전략적 해설 |
| :-- | :--- | :--- | :--- |
| **사업 예산 및 기간** | | | |
| **모집단** | | | |
| **표본틀** | | | |
| **표본할당방법** | | | |
| **조사지역** | | | |
| **표본수** | | | |
| **조사방법(온라인/면접 등)** | | | |
| **품질 및 검증 관리** | | | |
| **필수 인력 요건** | | | |
| **성과품 및 활용** | | | |
"""

            sys_prompt = f"""
# Role Definition
당신은 공공기관 입찰 전략 컨설턴트이자 20년 경력의 수석 리서치 연구원입니다. 
당신의 임무는 절대적으로 제공된 [금년도 문서]의 텍스트를 기반으로 분석을 수행하는 것입니다.

# [CRITICAL RULE] NO HALLUCINATIONS & TABLE STABILITY
1. **절대로** 문서에 없는 정보를 지어내지 마세요.
2. 정보가 없는 항목은 반드시 **"명시되지 않음"** 또는 **"확인 불가"**라고 작성하세요.
3. **[표(Table) 작성 규칙]**: 모든 표(Section 1, 4, 5) 내부의 각 셀은 반드시 **한 줄**로 작성하세요. 셀 내부에서 불릿(`-`)이나 줄바꿈을 절대 사용하지 마세요. 줄바꿈이 필요한 경우 쉼표(`,`) 또는 세미콜론(`;`)을 사용하여 한 줄로 나열하세요. 표의 구조(`|`)가 깨지지 않도록 극도로 주의하세요.

# [FORMATTING RULE] CONCISE TONE & LINE BREAKS
- 모든 문장은 **명사형 어미**(~함, ~임, ~필요, ~준비 등)를 사용하여 간결하게 설명하세요.
- 줄바꿈이 필요한 경우 반드시 실제 줄바꿈(`\\n`)을 사용하세요. **`<br>` 태그는 절대 사용하지 마세요.**

# [CITATION RULE]
- **섹션 1 (표)**: 표 내부에는 **출처(페이지, 제목 등)를 절대 표기하지 마세요.**
- **섹션 2, 3, 4, 5**: 각 근거 뒤에 반드시 괄호를 사용하여 페이지만 표기하세요 (예: (10p)).

# Analysis Instructions
아래 섹션에 맞춰 분석 결과를 출력하세요.
{section_1_prompt}

## 2. 배점표 기반 승부처 분석 (Scoring Strategy)
**배점이 높거나 중요한 요건 3가지를 명사형으로 기술하고 출처 페이지를 표기하세요.**

## 3. 과업 내용 기반 필수 수행 체크리스트 (Must-Do List)
**과업지시서상 필수 수행 과업을 추출하세요. [중요] 반드시 제안요청서의 '목차' 순서에 맞추어 재배치하여 제시하세요.**

## 4. 행정 서류 및 제안서 규격 체크리스트 (Administrative Check)
**제출 서류 및 규격을 정리하고 출처 페이지를 표기하세요.**

## 5. 상세 전략 및 가점 요인 (Bonus Strategy)
**가점 항목 및 전략적 제언을 아래 표 형식으로 정리하세요.**

| 구분 | 상세 내용 | 전략적 제언 |
| :--- | :--- | :--- |
| **가점 항목** | | |
| **차별화 요소** | | |
| **핵심 제언** | | |
"""
            # Use a balanced slice of the text
            def get_balanced_context(text, max_chars=30000):
                if len(text) <= max_chars: return text
                half = max_chars // 2
                return text[:half] + "\n\n... (중략) ...\n\n" + text[-half:]

            user_content = f"[금년도 문서]\n{get_balanced_context(full_current_text, 30000)}\n\n[직전 연도 문서]\n{get_balanced_context(prev_text, 10000) if prev_text else '없음'}"
            
            prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "{text}")])
            chain = prompt | llm | StrOutputParser()
            
            # Run consolidated analysis
            with st.spinner("전문가 모드로 제안요청서를 정밀 분석 중입니다..."):
                response = invoke_with_retry(chain, {"text": user_content})
                # Clean Output aggressively
                cleaned_response = clean_ai_output(response)
                st.session_state.analysis_results["top_keywords"] = top_keywords
                st.session_state.analysis_results["main_analysis"] = cleaned_response
                
                # Pre-generate and cache Docx report
                import report_utils
                report_data = {
                    "제안요청서 분석 결과": cleaned_response,
                    "키워드 인사이트": "" # Will be updated if summary exists
                }
                st.session_state.analysis_results["docx_file"] = report_utils.generate_word_report(report_data)

        except Exception as e:
            st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
            # Diagnostic for 404
            if "NOT_FOUND" in str(e) or "not found" in str(e).lower():
                with st.expander("🛠️ API 모델 접근 진단"):
                    try:
                        genai.configure(api_key=api_key)
                        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        st.write("현재 API 키로 사용 가능한 모델 목록:")
                        st.code("\n".join(models))
                    except: pass

    # --- Persistent Display Area ---
    if "analysis_results" in st.session_state and st.session_state.analysis_results:
        tabs = st.tabs(["📋 제안요청서 분석 결과", "📊 키워드 인사이트"])
        
        with tabs[0]:
            st.header("📋 제안요청서 분석 결과")
            analysis_text = st.session_state.analysis_results.get("main_analysis", "")
            st.markdown(analysis_text)
            
            st.markdown("---")
            st.warning("⚠️ **[주의] 현재 분석 결과는 임시 상태입니다. 상단 '워드 파일 다운로드' 버튼을 눌러 결과물을 저장하세요. 새로운 자료를 업로드하여 분석을 시작하면 기존 내용은 사라집니다.**")

        with tabs[1]:
            st.header("📊 키워드 인사이트")
            keywords = st.session_state.analysis_results.get("top_keywords", [])
            chart = create_word_chart(keywords)
            if chart: st.pyplot(chart)
            
            # Key Summary via LLM only if not already done
            with st.spinner("핵심 키워드 기반 사업 요약 중..."):
                try:
                    if "keyword_summary" not in st.session_state.analysis_results:
                        MODEL_NAME = get_best_available_model(api_key)
                        llm_k = ChatGoogleGenerativeAI(temperature=0.0, model=MODEL_NAME, google_api_key=api_key)
                        prompt_k = ChatPromptTemplate.from_template(
                            "당신은 공공기관 입찰 전문가입니다. 상위 키워드를 분석하여 표로 정리하세요. 키워드: {keywords}"
                        )
                        chain_k = prompt_k | llm_k | StrOutputParser()
                        st.session_state.analysis_results["keyword_summary"] = invoke_with_retry(chain_k, {"keywords": str(keywords)})
                    
                    st.markdown(st.session_state.analysis_results["keyword_summary"])
                    
                    # Update Docx with keyword summary if not already included
                    if st.session_state.analysis_results.get("docx_file"):
                        import report_utils
                        report_data = {
                            "제안요청서 분석 결과": st.session_state.analysis_results.get("main_analysis", ""),
                            "키워드 인사이트": st.session_state.analysis_results.get("keyword_summary", "")
                        }
                        st.session_state.analysis_results["docx_file"] = report_utils.generate_word_report(report_data)
                except:
                    pass

        # Display cached download button
        if st.session_state.analysis_results.get("docx_file"):
            st.markdown("---")
            st.download_button(
                label="📥 분석 결과 워드 파일 다운로드",
                data=st.session_state.analysis_results["docx_file"],
                file_name="win_strategy_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True,
                key="final_dw_btn_stable_cached"
            )
st.markdown('<div class="footer">Developed by ㅈㅅㅎ | Powered by Streamlit & Google Gemini</div>', unsafe_allow_html=True)
