
import gspread
import pandas as pd
import streamlit as st
import bcrypt
from google.oauth2.service_account import Credentials

# Google Sheets Configuration
SHEET_NAME = "RFP MASTER"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_connection():
    """Establishes connection to Google Sheets using secrets."""
    try:
        # Load credentials from secrets
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Google Sheets 연결 실패: {e}. secrets.toml을 확인해주세요.")
        return None

def init_db():
    """Initializes the Google Sheet (creates it if not exists, adds header)."""
    client = get_connection()
    if not client: return

    try:
        # Try to open existing sheet
        sheet = client.open(SHEET_NAME).sheet1
    except gspread.SpreadsheetNotFound:
        try:
            # Create new sheet if possible (requires Drive scope and permission)
            sh = client.create(SHEET_NAME)
            sh.share(st.secrets["gcp_service_account"]["client_email"], perm_type='user', role='owner')
            sheet = sh.sheet1
            # Add Header
            sheet.append_row(["email", "password", "name", "approved", "role"])
        except Exception as e:
            st.error(f"시트 생성 실패. 구글 드라이브에 '{SHEET_NAME}' 시트를 직접 생성하고 서비스 계정과 공유해주세요. ({e})")
            return
            
    # Check header
    current_data = sheet.get_all_values()
    if not current_data:
         sheet.append_row(["email", "password", "name", "approved", "role"])

def get_all_users():
    """Fetches all user data as a DataFrame."""
    client = get_connection()
    if not client: return pd.DataFrame()
    
    try:
        sheet = client.open(SHEET_NAME).sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        # st.error(f"데이터 조회 실패: {e}")
        return pd.DataFrame()

def create_user(email, password, name):
    """Creates a new user in Google Sheets."""
    client = get_connection()
    if not client: return False
    
    sheet = client.open(SHEET_NAME).sheet1
    
    # Check for duplicates
    existing_users = get_all_users()
    if not existing_users.empty and email in existing_users['email'].values:
        return False
        
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Auto-approve admin from secrets
    role = 'user'
    approved = 0 # 0 for False, 1 for True
    
    try:
        admin_id = st.secrets["admin"]["id"]
        if email == admin_id:
            role = 'admin'
            approved = 1
    except Exception:
        pass # If secrets are not set, no default admin
        
    # Append row
    sheet.append_row([email, hashed_pw, name, approved, role])
    return True

def login_user(email, password):
    """Verifies user credentials."""
    users = get_all_users()
    if users.empty: return None
    
    user_row = users[users['email'] == email]
    if not user_row.empty:
        user_data = user_row.iloc[0]
        stored_pw = user_data['password']
        if bcrypt.checkpw(password.encode('utf-8'), stored_pw.encode('utf-8')):
            return {
                "email": user_data['email'], 
                "name": user_data['name'], 
                "approved": bool(user_data['approved']), # Ensure boolean
                "role": user_data['role']
            }
    return None

def get_pending_users():
    """Returns a DataFrame of unapproved users."""
    users = get_all_users()
    if users.empty: return pd.DataFrame()
    
    # Filter where approved is 0 or False
    # Google Sheets might return integers or strings depending on input
    # Let's standardize filtering
    pending = users[ (users['approved'] == 0) | (users['approved'] == '0') | (users['approved'] == False) ]
    return pending

def approve_user(email):
    """Approves a user by updating the Google Sheet."""
    client = get_connection()
    if not client: return
    
    sheet = client.open(SHEET_NAME).sheet1
    
    # Find the cell with the email using find()
    try:
        cell = sheet.find(email)
        if cell:
            # Update 'approved' column (Column 4, D)
            # Row matches the cell row
            sheet.update_cell(cell.row, 4, 1) # Set approved to 1
    except Exception as e:
        st.error(f"승인 처리 중 오류: {e}")
