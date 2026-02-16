
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
import auth
import email_utils

# -----------------------------------------------------------------------------
# 1. Config & Branding
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="수주비책 (Win Strategy)",
    page_icon="🏆",
    layout="wide",
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F1F5F9;
        color: #64748B;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Authentication Flow
# -----------------------------------------------------------------------------
auth.init_db()

if "user" not in st.session_state:
    st.session_state.user = None

def login_page():
    st.markdown('<div class="main-header">수주비책 (Win Strategy)</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["로그인", "회원가입"])
    
    with tab1:
        st.subheader("로그인")
        email = st.text_input("이메일", key="login_email")
        password = st.text_input("비밀번호", type="password", key="login_pw")
        if st.button("로그인", type="primary"):
            user = auth.login_user(email, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("이메일 또는 비밀번호가 잘못되었습니다.")

    with tab2:
        st.subheader("회원가입")
        new_email = st.text_input("이메일", key="signup_email")
        new_name = st.text_input("이름", key="signup_name")
        new_password = st.text_input("비밀번호", type="password", key="signup_pw")
        new_password_confirm = st.text_input("비밀번호 확인", type="password", key="signup_pw_chk")
        
        if st.button("가입하기"):
            if new_password != new_password_confirm:
                st.error("비밀번호가 일치하지 않습니다.")
            elif not new_email or not new_password or not new_name:
                st.error("모든 정보를 입력해주세요.")
            else:
                if auth.create_user(new_email, new_password, new_name):
                    email_utils.send_admin_notification(new_email, new_name)
                    st.success("가입 요청이 완료되었습니다. 관리자 승인 후 이용 가능합니다.")
                else:
                    st.error("이미 가입된 이메일입니다.")

def admin_dashboard():
    st.sidebar.markdown("---")
    st.sidebar.subheader("👑 관리자 메뉴")
    
    # 1. Pending Approvals
    pending_users = auth.get_pending_users()
    if not pending_users.empty:
        st.sidebar.warning(f"승인 대기: {len(pending_users)}명")
        with st.expander("회원 승인 관리", expanded=True):
            for _, row in pending_users.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{row['name']}** ({row['email']})")
                with col2:
                    if st.button("승인", key=f"btn_{row['email']}"):
                        auth.approve_user(row['email'])
                        
                        # Send Email
                        success, form = email_utils.send_approval_email(row['email'])
                        if success:
                            st.success(f"승인 완료 및 메일 발송! ({row['email']})")
                        else:
                            st.warning(f"승인 완료되었으나 메일 발송 실패: {form}")
                        st.rerun()
    else:
        st.sidebar.info("승인 대기 중인 회원이 없습니다.")


# If logic for non-logged in users
if not st.session_state.user:
    login_page()
    st.stop()

# If logic for logged in but unapproved users
if not st.session_state.user['approved']:
    st.markdown('<div class="main-header">수주비책 (Win Strategy)</div>', unsafe_allow_html=True)
    st.warning(f"환영합니다, {st.session_state.user['name']}님!")
    st.info("현재 계정 승인 대기 중입니다. 관리자 승인 후 이메일 알림이 발송됩니다.")
    if st.button("로그아웃"):
        st.session_state.user = None
        st.rerun()
    st.stop()

# -----------------------------------------------------------------------------
# 3. Authenticated Main Application
# -----------------------------------------------------------------------------

# Sidebar Logic for Authenticated Users
with st.sidebar:
    st.write(f"접속자: **{st.session_state.user['name']}**님")
    if st.button("로그아웃", key="logout_sidebar"):
        st.session_state.user = None
        st.rerun()
    
    # Admin Logic
    if st.session_state.user['role'] == 'admin':
        admin_dashboard()
        
    st.markdown("---")
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921222.png", width=50) 
    st.header("설정 (Settings)")
    
    # API Key Management (Secrets & Env)
    api_key = st.secrets["groq"]["api_key"]
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
 
    
    st.markdown("---")
    st.markdown("**Developed by ㅈㅅㅎ**")


# --- Main App Logic (from previous version) ---

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
    elif system_name == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
    else: # Linux (Streamlit Cloud)
        # Try to find Nanum font explicitly
        path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        if os.path.exists(path):
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rc('font', family=font_name)
        else:
            # Fallback
            plt.rc('font', family='NanumGothic')
    
    plt.rc('axes', unicode_minus=False)

    ax.barh(words, counts, color='#3B82F6')
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title('Top 20 Keywords')
    return fig

# UI Layout
st.markdown('<div class="main-header">수주비책 (Win Strategy)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">공공기관 입찰 성공을 위한 제안요청서(RFP) 심층 분석 솔루션</div>', unsafe_allow_html=True)

st.info("⚠️ 정확한 분석을 위해 모든 문서는 **PDF 형식**으로 변환하여 업로드해 주세요.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. 금년도 공고 자료 (필수)")
    current_rfp = st.file_uploader("2025년 제안요청서 업로드", type=["pdf"], key="curr_rfp")
    no_task_desc = st.checkbox("과업지시서 없음 (제안요청서 내 포함)", key="chk_no_task")
    current_task = st.file_uploader("2025년 과업지시서 업로드", type=["pdf"], disabled=no_task_desc, key="curr_task")

with col2:
    st.subheader("2. 직전 연도 공고 자료 (선택)")
    no_prev_rfp = st.checkbox("직전 제안요청서 없음", key="chk_no_prev_rfp")
    prev_rfp = st.file_uploader("직전 연도 제안요청서 업로드", type=["pdf"], disabled=no_prev_rfp, key="prev_rfp")
    
    no_prev_task = st.checkbox("직전 과업지시서 없음", key="chk_no_prev_task")
    prev_task = st.file_uploader("직전 연도 과업지시서 업로드", type=["pdf"], disabled=no_prev_task, key="prev_task")

start_analysis = st.button("제안요청서 분석 시작 🚀", type="primary", use_container_width=True)

if start_analysis:
    if not api_key:
        st.error("분석을 시작하려면 API Key 설정이 필요합니다.")
        st.stop()
        
    if not current_rfp:
        st.error("2025년 제안요청서는 필수 업로드 항목입니다.")
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
        llm = ChatGroq(temperature=0.0, model_name="openai/gpt-oss-20b", api_key=api_key)
        tabs = st.tabs(["키워드 인사이트", "직전 문서 비교", "조사설계", "표본설계", "필수 제안 항목", "준비서류", "목차 체크리스트", "상세 전략"])
        
        # Store results in session state for report generation
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}

        # 1. Keyword Insight
        with tabs[0]:
            st.header("키워드 인사이트")
            chart = create_word_chart(top_keywords)
            if chart: st.pyplot(chart)
            with st.spinner("비정형 데이터 분석 중..."):
                prompt = ChatPromptTemplate.from_template("다음 키워드를 바탕으로 발주처의 의도를 1~2문장으로 요약: {keywords}")
                chain = prompt | llm | StrOutputParser()
                insight = chain.invoke({"keywords": str(top_keywords)})
                st.info(f"**AI Insight:** {insight}")
                st.session_state.analysis_results["키워드 인사이트"] = f"Top Keywords: {str(top_keywords)}\n\nAI Insight: {insight}"

        # 2. Previous Comparison
        with tabs[1]:
            st.header("직전 제안요청서 비교")
            if not prev_text.strip(): 
                st.warning("직전 연도 자료가 없어 비교 분석을 생략합니다.")
                st.session_state.analysis_results["직전 제안요청서 비교"] = "비교 데이터 없음"
            else:
                with st.spinner("직전 연도와 비교 분석 중..."):
                    prompt = ChatPromptTemplate.from_template("""
                        Compare [Previous] and [Current] documents. 
                        Analyze changes in Budget, Period, Sample Size, Methodology, Evaluation Criteria.
                        Output in Markdown table.
                        [Previous] {prev_text} [Current] {curr_text}
                    """)
                    chain = prompt | llm | StrOutputParser()
                    res = chain.invoke({"prev_text": prev_text[:15000], "curr_text": full_current_text[:15000]})
                    st.markdown(res)
                    st.session_state.analysis_results["직전 제안요청서 비교"] = res

        # 3. Detailed Analysis
        def run_analysis(tab_name, instructions, context_text, target_tab):
            with target_tab:
                st.header(tab_name)
                with st.spinner(f"{tab_name} 분석 중..."):
                    sys_prompt = f"당신은 입찰 전략 컨설턴트입니다. 다음 지시에 따라 분석하세요: {instructions}"
                    prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "{text}")])
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({"text": context_text[:25000]})
                    st.markdown(response)
                    st.session_state.analysis_results[tab_name] = response

        run_analysis("조사설계", "조사 개요, 필수 과업, 예비조사 여부 등", full_current_text, tabs[2])
        run_analysis("표본설계", "모집단, 표본 추출 방식, 오차, 관리 방안", full_current_text, tabs[3])
        run_analysis("필수 제안 항목", "필수 수행 활동, 데이터 품질, 성과품 규격", full_current_text, tabs[4])
        run_analysis("준비서류", "입찰 자격, 제출 서류 리스트", full_current_text, tabs[5])
        run_analysis("목차 체크리스트", "필수 목차, 공고기관 강조 포인트(CSF)", full_current_text, tabs[6])
        run_analysis("상세 전략", "정량평가, 핵심 인력, 데이터 품질, 사후관리, 보안", full_current_text, tabs[7])

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
