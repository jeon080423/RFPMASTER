import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import datetime
import time
import google.generativeai as genai
import traceback
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
try:
    auth.init_db()
except Exception as e:
    st.error(f"⚠️ 시스템 초기화 오류: {e}")

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

            st.markdown("---")
            with st.expander("🛠️ 관리자 설정", expanded=True):
                # Cleaned up model options: removing 2.5 (beta/exp) and old versions if necessary
                # Keeping stable and high-perf models
                model_options = [
                    "자동 최적화 (권장)", 
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-2.0-flash", 
                    "gemini-1.5-pro", 
                    "gemini-1.5-flash",
                    "gemini-1.5-flash-latest",
                    "gemini-2.0-pro-exp-02-05"
                ]
                
                # Model selection for the Admin themselves
                st.selectbox(
                    "관리자 전용 모델 지정", 
                    options=model_options, 
                    key="admin_selected_model",
                    help="현재 세션에서 관리자 본인이 사용할 모델을 고릅니다."
                )

                # Global model setting for regular users
                current_global_default = auth.get_global_setting("user_default_model", "gemini-2.5-flash")
                # Find index of current setting in options
                try: 
                    default_idx = model_options.index(current_global_default)
                except: 
                    default_idx = 2 # Default to 2.5 flash if not found

                new_global_default = st.selectbox(
                    "🌟 일반 사용자 기본 모델 설정",
                    options=model_options,
                    index=default_idx,
                    help="모든 일반 사용자가 기본적으로 사용하게 될 모델을 지정합니다. (실시간 반영)"
                )
                
                if new_global_default != current_global_default:
                    if auth.set_global_setting("user_default_model", new_global_default):
                        st.success(f"기본 모델이 {new_global_default}로 변경되었습니다!")
                        st.cache_data.clear() # Clear cache to reflect change
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
    
    # Settings (Multi-key support)
    st.markdown("---")
    sec_gemini = st.secrets.get("gemini", {})
    api_keys = []
    
    # Dynamically collect all keys starting with 'api_key'
    try:
        # Streamlit secrets might not be a literal dict, but should be iterable
        sorted_keys = sorted(sec_gemini.keys())
        for k in sorted_keys:
            if k.startswith("api_key") and sec_gemini[k]:
                api_keys.append(sec_gemini[k])
    except:
        # Fallback for unexpected secrets structure
        if sec_gemini.get("api_key"): 
            api_keys.append(sec_gemini.get("api_key"))
    
    # Fallback to env if empty
    if not api_keys and os.environ.get("GOOGLE_API_KEY"):
        api_keys.append(os.environ.get("GOOGLE_API_KEY"))
    
    # Current primary key for simple usage
    api_key = api_keys[0] if api_keys else ""
    
    st.markdown("---")
    st.markdown("**Developed by ㅈㅅㅎ**")
    st.markdown("""
    <div style='
        font-size: 0.95rem; 
        color: white; 
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); 
        padding: 20px; 
        border-radius: 12px; 
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        line-height: 1.6;
    '>
        <div style='font-weight: 800; font-size: 1.1rem; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px;'>
            Developer Contact
        </div>
        <div style='margin-bottom: 8px;'>
            <b>이메일</b><br>jeon080423@gmail.com
        </div>
        <div style='margin-bottom: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
            <b>후원계좌</b><br>
            <span style='font-size: 0.9rem;'>카카오뱅크 3333-23-866708 ㅈㅅㅎ</span>
        </div>
        <div style='font-size: 0.8rem; background: rgba(0,0,0,0.15); padding: 10px; border-radius: 8px; font-weight: 500;'>
            지속 가능한 서비스를 위해 여러분의 응원이 필요합니다. 모인 후원금은 서버 비용 및 API 업그레이드를 위한 자금으로 사용됩니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Utility Functions
# -----------------------------------------------------------------------------
def mask_pii(text):
    rrn_pattern = r'\d{6}[-\s]\d{7}'
    masked_text = re.sub(rrn_pattern, '******-*******', text)
    return masked_text

def remove_repeating_lines(pages_text_list):
    """Detects and removes frequent headers/footers appearing at top/bottom of pages."""
    if len(pages_text_list) < 3: return pages_text_list # Too few pages to identify patterns
    
    header_counts = {}
    footer_counts = {}
    
    for page in pages_text_list:
        lines = [l.strip() for l in page.split('\n') if l.strip()]
        if not lines: continue
        
        h = lines[0]
        f = lines[-1]
        
        header_counts[h] = header_counts.get(h, 0) + 1
        footer_counts[f] = footer_counts.get(f, 0) + 1
    
    # Threshold: line appears in more than 50% of pages
    threshold = len(pages_text_list) * 0.5
    headers_to_remove = {k for k, v in header_counts.items() if v > threshold}
    footers_to_remove = {k for k, v in footer_counts.items() if v > threshold}
    
    cleaned_pages = []
    for page in pages_text_list:
        lines = page.split('\n')
        if not lines:
            cleaned_pages.append("")
            continue
            
        new_lines = []
        for i, line in enumerate(lines):
            ls = line.strip()
            # Remove if it's the first/last non-empty line and in removal set
            if i == 0 and ls in headers_to_remove: continue
            if i == len(lines)-1 and ls in footers_to_remove: continue
            new_lines.append(line)
        cleaned_pages.append('\n'.join(new_lines))
        
    return cleaned_pages

def extract_text_from_pdf(uploaded_file):
    if not uploaded_file: return ""
    
    # --- Extraction Caching ---
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if "extraction_cache" not in st.session_state:
        st.session_state.extraction_cache = {}
        
    if file_id in st.session_state.extraction_cache:
        return st.session_state.extraction_cache[file_id]
    
    text = ""
    boilerplate_keywords = ["청렴계약", "서식 제", "별지 제", "조세포탈", "청렴 서약", "행정 처분", "입찰 참가 신청서"]
    preserve_keywords = ["제출 서류", "서류 목록", "평가항목", "배점표"]
    
    filtered_pages = 0
    pages_text_list = []
    
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    # Logic: If boilerplate keyword exists, check if any preserve keyword exists
                    # If it's pure boilerplate (forms/stamps), skip it to save tokens
                    has_bp = any(kw in page_text for kw in boilerplate_keywords)
                    has_pr = any(kw in page_text for kw in preserve_keywords)
                    
                    if has_bp and not has_pr:
                        filtered_pages += 1
                        continue # Skip this page
                    
                    # Prepend explicit Page Marker for accurate citations despite H/F removal
                    page_with_marker = f"[Page {page.page_number}]\n{page_text}"
                    pages_text_list.append(page_with_marker)
        
        # --- Advanced Optimization: Header/Footer Removal ---
        cleaned_pages = remove_repeating_lines(pages_text_list)
        text = "\n".join(cleaned_pages)
        
        # --- Token Saving Cleanup ---
        # Remove redundant newlines (3 or more -> 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove redundant spaces (2 or more -> 1)
        text = re.sub(r' +', ' ', text)
        # Structural Optimization: Collapse | | spaces in potential tables
        text = re.sub(r'\|\s+\|', '||', text)
        # Remove leading/trailing whitespace per line
        text = '\n'.join([line.strip() for line in text.split('\n')])
        
        if filtered_pages > 0:
            st.info(f"💡 행정 서식 및 단순 양식 페이지 {filtered_pages}개를 분석에서 제외하여 토큰을 절약했습니다. (전체 {total_pages}p)")
        
        # Mask PII and Cache
        final_text = mask_pii(text)
        st.session_state.extraction_cache[file_id] = final_text
        return final_text
        
    except Exception as e:
        return f"Error reading PDF: {e}"

def get_best_available_model(api_key):
    """Dynamically find the best available model (Pro first) for the given API key."""
    if "model_cache" not in st.session_state:
        st.session_state.model_cache = {}
        
    try:
        # Check cache first to save quota calls
        if api_key in st.session_state.model_cache:
            available_models = st.session_state.model_cache[api_key]
        else:
            genai.configure(api_key=api_key)
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.session_state.model_cache[api_key] = available_models
        
        priority = [
            "models/gemini-2.5-pro",
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-pro",
            "models/gemini-1.5-pro-latest",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-pro"
        ]
        
        for p in priority:
            if p in available_models:
                return p
        
        if available_models:
            return available_models[0]
    except: pass
    return "gemini-1.5-flash"

def get_flash_model(api_key):
    """Dynamically find the fastest/cheapest available model (Flash first)."""
    if "model_cache" not in st.session_state:
        st.session_state.model_cache = {}

    try:
        # Check cache first to save quota calls
        if api_key in st.session_state.model_cache:
            available_models = st.session_state.model_cache[api_key]
        else:
            genai.configure(api_key=api_key)
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.session_state.model_cache[api_key] = available_models
        
        # Priority: Flash 2.5 -> Flash 2.0 -> Flash 1.5
        priority = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro"
        ]
        
        for p in priority:
            if p in available_models:
                return p
    except: pass
    return "gemini-1.5-flash"

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
# -----------------------------------------------------------------------------
# Global Quota Status Notification
# -----------------------------------------------------------------------------
is_exhausted, reset_time = auth.check_quota_status()
if is_exhausted:
    st.warning(f"⚠️ **금일 모든 분석 API 쿼터가 소진되었습니다.**\n\n모든 제미나이(Gemini) API 키의 일일 할당량이 소진되었습니다. 다음 초기화 시간(**{reset_time} KST**) 이후에 다시 분석이 가능합니다.")

st.markdown('<div class="main-header">수주비책 (Win Strategy)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">공공기관 입찰 성공을 위한 제안요청서(RFP) 심층 분석 솔루션</div>', unsafe_allow_html=True)

# Rate limit retry helper with Key Rotation
def invoke_with_retry(prompt_template, params, api_keys, use_flash=False, model_name=None):
    """Invoke LLM chain with Gemini keys (once each) and rotate on failure."""
    if not api_keys:
        raise Exception("API Key가 설정되지 않았습니다.")
    
    # Try each Gemini key exactly once
    for i, key in enumerate(api_keys):
        try:
            # If a specific Gemini model was requested, use it, otherwise detect best
            if model_name and model_name.startswith("gemini-"):
                actual_model = model_name
            else:
                actual_model = get_flash_model(key) if use_flash else get_best_available_model(key)
                
            # Normalize: ensure 'models/' prefix for Gemini models
            if actual_model.startswith("gemini-") and not actual_model.startswith("models/"):
                actual_model = f"models/{actual_model}"
                
            # Default to stable v1 (v1beta often causes 404 for standard models)
            llm = ChatGoogleGenerativeAI(temperature=0.0, model=actual_model, google_api_key=key, version="v1")
            chain = prompt_template | llm | StrOutputParser()
            return chain.invoke(params)
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle 404 Not Found or other naming issues by falling back to guaranteed stable IDs and versions
            if "not found" in error_str or "404" in error_str:
                # Sequence of safe fallbacks: (model_name, api_version)
                safe_fallbacks = [
                    ("models/gemini-1.5-flash-latest", "v1"),
                    ("models/gemini-1.5-pro-latest", "v1"),
                    ("models/gemini-pro", "v1")
                ]
                for fallback_model, fallback_version in safe_fallbacks:
                    if actual_model != fallback_model:
                        try:
                            llm = ChatGoogleGenerativeAI(
                                temperature=0.0, 
                                model=fallback_model, 
                                google_api_key=key,
                                version=fallback_version
                            )
                            return (prompt_template | llm | StrOutputParser()).invoke(params)
                        except Exception as inner_e:
                            inner_msg = str(inner_e).lower()
                            if "not found" not in inner_msg and "404" not in inner_msg:
                                break 
                                
            # Define skipable errors that should trigger rotation to NEXT key
            skipable_errors = [
                'rate_limit', '429', 'resource_exhausted', # Limits
                '404', 'not found',                        # Still getting 404 after fallbacks
                '401', 'unauthorized',                     # Invalid key (Expired/Old)
                '403', 'forbidden', 'permission'           # Permission issue
            ]
            
            if any(err in error_str for err in skipable_errors):
                st.warning(f"🔄 제미나이 {i + 1}번 키 오류 ({error_str[:120]}...). 다음 키로 전환합니다.")
                continue # Try the next key in the list
            else:
                raise e
                
    # If we reach here, everything failed.
    auth.record_quota_exhaustion()
    raise Exception("모든 제미나이 API 키가 작동하지 않거나 호출 한도에 도달했습니다. (404/401/429 등) 각 키가 유효한지, 그리고 모델명이 올바른지 다시 확인해 주세요. 약 1분 후 다시 시도하시거나 오후 5시 초기화 이후 이용을 권장합니다.")

st.info("⚠️ 정확한 분석을 위해 모든 문서는 **PDF 형식**으로 변환하여 업로드해 주세요.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. 금년도 공고 자료 (필수)")
    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)
    current_rfp = st.file_uploader("올해 제안요청서 또는 과업지시서", type=["pdf"], key="curr_rfp")

with col2:
    st.subheader("2. 직전 회차 공고 자료 (선택)")
    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)
    prev_rfp = st.file_uploader("직전 회차 제안요청서 또는 과업지시서", type=["pdf"], key="prev_rfp_uploader")

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
    def sanitize_spaces(text):
        """Replaces non-breaking spaces and other artifacts with standard spaces."""
        if not text: return text
        return text.replace('\xa0', ' ').replace('\u200b', '').replace('\uFEFF', '').strip()

    def detect_project_name(text):
        """Attempts to extract the project name from the first page of the RFP with robust regex."""
        if not text: return "미지정 사업"
        
        # Sanitize input text first
        text = sanitize_spaces(text)
        
        # 1. Look for keywords using regex to handle prefixes (1., 가., 등)
        lines = [l.strip() for l in text[:3000].split('\n') if l.strip()]
        keywords = ["사업명", "과업명", "용역명", "명칭", "프로젝트명", "공고명"]
        
        for i, line in enumerate(lines):
            for kw in keywords:
                # Regex: optional prefix, keyword, optional colon/bracket
                pattern = rf'^(?:[0-9가-힣\d\.]+\s*)?{kw}\s*[:：\s\]\)]'
                if re.search(pattern, line):
                    # Try to get content after colon
                    if ':' in line: 
                        name = line.split(':', 1)[1].strip()
                        if len(name) > 3: return name
                    elif '：' in line:
                        name = line.split('：', 1)[1].strip()
                        if len(name) > 3: return name
                    
                    # If line ends with keyword, title is likely on the next line
                    if i + 1 < len(lines):
                        next_line = lines[i+1].strip()
                        if len(next_line) > 3: return next_line
        
        # 2. Heuristic fallback: Look for a long line in the first 10 non-empty lines
        # Usually titles are prominent.
        if lines:
            for line in lines[:10]:
                # Guess it's a title if it's long and doesn't look like an address or simple date
                if 15 < len(line) < 100 and not any(x in line for x in ["주소", "일시", "일자", "연락처"]):
                    return sanitize_spaces(line)
        
        return "미지정 사업"

    def detect_year(text, default_label):
        """Attempts to detect the year from the text (e.g., '2024년')."""
        if not text: return default_label
        match = re.search(r'20\d{2}년', text[:3000])
        if match:
            return match.group(0)
        return default_label

    def clean_ai_output(text):
        """
        Forcefully removes <br> tags and cleans artifacts.
        """
        if not text: return ""
        
        # Remove non-breaking spaces and artifacts
        text = sanitize_spaces(text)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if '|' in line:
                # Inside table row: replace <br> with ; to keep it on one line
                cleaned_line = re.sub(r'<br\s*/?>', '; ', line, flags=re.IGNORECASE)
                cleaned_lines.append(cleaned_line)
            else:
                # Outside table: replace <br> with \n
                cleaned_line = re.sub(r'<br\s*/?>', '\n', line, flags=re.IGNORECASE)
                cleaned_lines.append(cleaned_line)
        
        # Final step for UI: Convert '; ' back to '<br>' for rendering inside tables
        result = '\n'.join(cleaned_lines)
        if '|' in result:
            final_lines = []
            for line in result.split('\n'):
                if '|' in line:
                    final_lines.append(line.replace('; ', '<br>'))
                else:
                    final_lines.append(line)
            return '\n'.join(final_lines)
        return result

    if start_analysis:
        if not api_keys: # Changed from api_key to api_keys
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

        # 1. Resolve Model
        admin_model = st.session_state.get("admin_selected_model")
        
        # Security: ensure model is still in allowed options
        valid_options = [
            "자동 최적화 (권장)", "gemini-2.5-flash", "gemini-2.5-pro",
            "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
            "gemini-1.5-flash-latest", "gemini-2.0-pro-exp-02-05"
        ]
        if admin_model and admin_model not in valid_options:
            admin_model = "자동 최적화 (권장)"
            
        is_manual = admin_model and admin_model != "자동 최적화 (권장)"
        
        if is_manual:
            MODEL_NAME = admin_model
            model_display = f"{MODEL_NAME} (사용자 지정)"
        else:
            # Load global default set by admin (cached for 10 mins)
            @st.cache_data(ttl=600)
            def fetch_default_model():
                return auth.get_global_setting("user_default_model", "gemini-2.5-flash")
            
            global_model = fetch_default_model()
            MODEL_NAME = global_model
            model_display = f"{MODEL_NAME} (관리자 지정 기본값)"
        
        st.info(f"✨ 분석 모델: `{model_display}`")

        try:
            has_prev = bool(prev_text.strip())
            
            # Use a balanced slice of the text (Optimized for tokens)
            def get_balanced_context(text, max_chars=20000):
                if not text: return ""
                if len(text) <= max_chars: return text
                half = max_chars // 2
                return text[:half] + "\n\n... (중략) ...\n\n" + text[-half:]

            user_content = f"[금년도 문서]\n{get_balanced_context(full_current_text, 20000)}\n\n[직전 회차 문서]\n{get_balanced_context(prev_text, 8000) if prev_text else '없음'}"
            
            # Detect project name and store in session state
            project_name = detect_project_name(user_content)
            st.session_state.analysis_results["project_name"] = project_name
            st.session_state.analysis_results["main_analysis"] = ""

            # Common System Rules
            common_rules = """
당신은 공공기관 입찰 전략 컨설턴트입니다. 
당신의 임무는 절대적으로 제공된 [금년도 문서]의 텍스트를 기반으로 분석을 수행하는 것입니다.

[CRITICAL RULE]
1. 절대로 지어내지 말 것. 문서에 없는 정보는 반드시 "명시되지 않음" 또는 "확인 불가"라고 작성할 것.
2. 모든 문장은 명사형 어미(~함, ~임, ~필요)를 사용하여 간결하게 설명할 것.
3. 모든 표 내부의 각 셀은 반드시 한 줄로 작성할 것. 셀 내부에서 불릿(-)이나 줄바꿈을 절대 사용하지 말 것. 줄바꿈이 필요한 경우 반드시 세미콜론(; )을 사용하여 한 줄로 나열할 것.
4. <br> 태그는 절대 사용하지 말 것.
5. 근거 뒤에 반드시 괄호를 사용하여 페이지만 표기할 것 (예: (10p)).
6. 표 내부의 셀에 출처를 중복해서 표기하지 말 것. 출처 열이 따로 있다면 그곳에만 표기할 것.
"""

            # Definition of Analysis Sections
            sections = [
                {
                    "id": "sec1",
                    "title": "1. 제안요청서 핵심 비교 및 전략",
                    "prompt": f"""{common_rules}
아래 금년도와 직전 정보를 비교분석하여 표로 작성하세요. 직전 정보가 없는 경우 해당 칸은 '정보 없음'으로 기입하세요.
| 구분 | 금년도 | 직전 회차 | 변경 내용 및 전략적 해설 |
| :-- | :--- | :--- | :--- |
| **사업 예산 및 기간** | | | |
| **모집단 및 표본틀** | | | |
| **표본 수 및 할당** | | | |
| **조사 지역 및 방법** | | | |
| **품질 및 검증 관리** | | | |
| **인력 요건 및 성과품** | | | |
"""
                },
                {
                    "id": "sec2",
                    "title": "2. 배점표 기반 승부처 분석",
                    "prompt": f"""{common_rules}
배점이 높거나 중요한 요건 3가지를 추출하여 아래 표 형식으로 정리하세요.
| 주요 요건 | 배점 | 상세 내용 및 전략 | 출처 |
| :--- | :--- | :--- | :--- |
"""
                },
                {
                    "id": "sec3",
                    "title": "3. 과업 내용 기반 필수 수행 체크리스트",
                    "prompt": f"""{common_rules}
과업지시서상 필수 수행 과업을 추출하여 목차 순서에 맞게 정리하세요. 상세 수행 내용은 세미콜론(; )으로 연결하세요.
| 순서 | 필수 과업 내용 | 상세 수행 내용 | 출처 |
| :--- | :--- | :--- | :--- |
"""
                },
                {
                    "id": "sec4",
                    "title": "4. 행정 서류 및 제안서 규격 체크리스트",
                    "prompt": f"""{common_rules}
제출 서류 및 규격을 정리하고 출처 페이지를 표기하세요.
| 분류 | 상세 규격 및 서류 | 제출처 및 방식 | 출처 |
| :--- | :--- | :--- | :--- |
"""
                },
                {
                    "id": "sec5",
                    "title": "5. 상세 전략 및 가점 요인",
                    "prompt": f"""{common_rules}
가점 항목 및 전략적 제언을 아래 표 형식으로 정리하세요.
| 구분 | 상세 내용 | 전략적 제언 |
| :--- | :--- | :--- |
| **가점 항목** | | |
| **차별화 요소** | | |
| **핵심 제언** | | |
"""
                },
                {
                    "id": "sec6",
                    "title": "6. 제안서 목차 및 구성안 (Skeleton)",
                    "prompt": f"""{common_rules}
분석된 과업과 배점을 바탕으로 승률을 높이는 최적의 제안서 구성(Skeleton)을 제안하세요.
| 대목차 | 중/소목차 | 핵심 포함 내용 및 전략 | 비중(%) |
| :--- | :--- | :--- | :--- |
"""
                }
            ]

            # Live Update Container
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            for i, sec in enumerate(sections):
                current_percent = int((i / len(sections)) * 100)
                progress_bar.progress(current_percent)
                status_container.info(f"⏳ **[{project_name}]** 분석 중: {sec['title']} ({i+1}/{len(sections)})")
                
                full_prompt = f"{sec['prompt']}\n\n[CONTEXT]\n{user_content}"
                prompt_obj = ChatPromptTemplate.from_template(full_prompt)
                
                result = invoke_with_retry(prompt_obj, {}, api_keys, model_name=MODEL_NAME)
                
                # Check for PROJECT_NAME in the first section only if needed, but we already have detect_project_name
                if i == 0:
                    ai_name_match = re.search(r'\[PROJECT_NAME:\s*(.*?)\]', result)
                    if ai_name_match:
                        project_name = ai_name_match.group(1).strip()
                        st.session_state.analysis_results["project_name"] = project_name
                        result = result.replace(ai_name_match.group(0), "").strip()

                cleaned_sec = clean_ai_output(result)
                
                # Append to main_analysis
                st.session_state.analysis_results["main_analysis"] += f"## {sec['title']}\n{cleaned_sec}\n\n"
            
            progress_bar.progress(100)
            status_container.success(f"✅ **[{project_name}]** 분석 완료!")
            time.sleep(1)
            status_container.empty()
            progress_bar.empty()


            # 2. Similar Research Discovery (Search & Sort)
            with st.spinner("유사 학술연구 및 보도자료 검색 중..."):
                try:
                    search_prompt = ChatPromptTemplate.from_template("""
당신은 학술연구 전문 사서이자 정부 보고서 분석 전문가입니다. 
다음 [과업명]과 유사한 **국내** 학술 연구, 논문, 그리고 정부/공공기관의 조사 보고서를 7~10개 정도 찾아내어 표로 정리하세요.
**[중요] 반드시 국내 자료만 리스트업하고, 해외 연구는 제외하세요.**
**[중요] 반드시 최근 2년 이내(2024년 1월 ~ 2026년 현재)에 발표/발간된 자료여야 합니다.**

[과업명]
{project_name}

[분석 지침]
1. **학술 연구(논문)**를 최우선적으로 리스트업하세요.
2. 각 항목에 대해 아래 정보를 반드시 포함하세요:
   - 연도: 연도 4자리
   - 논문/보고서명: 연구의 정식 제목
   - 저자명: 대표 저자명
   - 저자 소속기관: 대학교 또는 연구기관명
   - 보고서 발간 기간: (예: 2023.01 ~ 2023.12 또는 단일 시점)
3. **정렬 규칙**:
   - 1순위: 학술연구(논문) 여부 (논문을 상단에)
   - 2순위: 저자명 가나다/ABC 순
   - 3순위: 소속기관 가나다/ABC 순
4. 표 형식으로만 출력하세요 (| 연도 | 논문/보고서명 | 저자명 | 저자 소속기관 | 보고서 발간 기간 |).
5. 실제 존재하는 연구 데이터만 기반으로 작성하세요.
""")
                    research_result = invoke_with_retry(search_prompt, {"project_name": project_name}, api_keys, use_flash=False, model_name=MODEL_NAME)
                    st.session_state.analysis_results["similar_research"] = clean_ai_output(research_result)
                except Exception as e:
                    st.session_state.analysis_results["similar_research"] = f"유사연구 검색 중 오류 발생: {e}"

            # 3. Pre-generate Docx report
            import report_utils
            report_data = {
                "제안요청서 분석 결과": st.session_state.analysis_results["main_analysis"],
                "유사연구 분석 리스트": st.session_state.analysis_results.get("similar_research", "")
            }
            st.session_state.analysis_results["docx_file"] = report_utils.generate_word_report(report_data, project_name=project_name)

        except Exception as e:
            st.error(f"AI 분석 중 오류가 발생했습니다: {type(e).__name__}: {e}")
            with st.expander("🛠️ 상세 오류 정보 (디버깅용)"):
                st.code(traceback.format_exc())
            
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
        tabs = st.tabs(["📋 제안요청서 분석 결과", "🔍 유사연구"])

        with tabs[0]:
            project_name = sanitize_spaces(st.session_state.analysis_results.get("project_name", "미지정 사업"))
            st.header(f"📋 제안요청서 분석 결과 [{project_name}]")
            analysis_text = st.session_state.analysis_results.get("main_analysis", "")
            st.markdown(analysis_text, unsafe_allow_html=True)

            st.markdown("---")
            st.warning("⚠️ **[주의] 현재 분석 결과는 임시 상태입니다. 하단 '워드 파일 다운로드' 버튼을 눌러 결과물을 저장하세요. 새로운 자료를 업로드하여 분석을 시작하면 기존 내용은 사라집니다.**")

        with tabs[1]:
            st.header("🔍 유사연구")
            st.info("💡 본 리스트는 제안서 작성을 위한 자문위원 섭외를 돕기 위해 관련 연구자와 유관기관 전문가를 통합 검색한 결과입니다.")
            research_text = st.session_state.analysis_results.get("similar_research", "")
            if research_text:
                st.markdown(research_text, unsafe_allow_html=True)
            else:
                st.info("유사연구 분석 결과가 없습니다.")

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
st.markdown('<div class="footer">Developed by ㅈㅅㅎ<br>jeon080423@gmail.com | Powered by Streamlit & Google Gemini</div>', unsafe_allow_html=True)
