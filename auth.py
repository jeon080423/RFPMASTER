
import gspread
import pandas as pd
import streamlit as st
import bcrypt
from google.oauth2.service_account import Credentials
from datetime import datetime

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
    
    # Standards: email, password, name, approved, role, last_login, analysis_count
    cols = ["email", "password", "name", "approved", "role", "last_login", "analysis_count"]
    
    if not current_data:
         sheet.append_row(cols)
    else:
        # Check if first row contains all columns
        first_row = [str(cell).strip().lower() for cell in current_data[0]]
        for col in cols:
            if col not in first_row:
                # Add missing column to the end
                idx = len(first_row) + 1
                sheet.update_cell(1, idx, col)
                first_row.append(col)

    # --- Initialize CONFIGS sheet ---
    try:
        sh = client.open(SHEET_NAME)
        try:
            config_sheet = sh.worksheet("CONFIGS")
        except gspread.WorksheetNotFound:
            config_sheet = sh.add_worksheet(title="CONFIGS", rows="100", cols="2")
            config_sheet.append_row(["key", "value"])
    except:
        pass

def get_global_setting(key, default=""):
    """Fetches a global setting from the CONFIGS sheet."""
    try:
        client = get_connection()
        sh = client.open(SHEET_NAME)
        config_sheet = sh.worksheet("CONFIGS")
        cell = config_sheet.find(key)
        if cell:
            return config_sheet.cell(cell.row, 2).value
    except:
        pass
    return default

def set_global_setting(key, value):
    """Sets a global setting in the CONFIGS sheet."""
    try:
        client = get_connection()
        sh = client.open(SHEET_NAME)
        config_sheet = sh.worksheet("CONFIGS")
        cell = config_sheet.find(key)
        if cell:
            config_sheet.update_cell(cell.row, 2, str(value))
        else:
            config_sheet.append_row([key, str(value)])
        return True
    except:
        return False


def get_all_users():
    """Fetches all user data as a DataFrame."""
    client = get_connection()
    if not client: return pd.DataFrame()
    
    try:
        sheet = client.open(SHEET_NAME).sheet1
        data = sheet.get_all_values()
        
        if not data: # Empty sheet
            return pd.DataFrame(columns=["email", "password", "name", "approved", "role"])
            
        headers = [h.strip().lower() for h in data[0]]
        rows = data[1:]
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Ensure required columns exist
        required_cols = ["email", "password", "name", "approved", "role", "last_login", "analysis_count"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = "" # Add missing column
                
        return df
    except Exception as e:
        # st.error(f"데이터 조회 실패: {e}")
        return pd.DataFrame(columns=["email", "password", "name", "approved", "role", "last_login", "analysis_count"])

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
        
    # Append row: last_login is empty, analysis_count is 0
    sheet.append_row([email, hashed_pw, name, approved, role, "", 0])
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
            # Update last login time
            try:
                client = get_connection()
                sheet = client.open(SHEET_NAME).sheet1
                cell = sheet.find(email)
                if cell:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Column 6 is last_login
                    sheet.update_cell(cell.row, 6, now)
            except:
                pass
                
            return {
                "email": user_data['email'], 
                "name": user_data['name'], 
                "approved": bool(user_data['approved']),
                "role": user_data['role']
            }
    return None

def get_user_by_email(email):
    """Fetches user data by email for cookie login."""
    users = get_all_users()
    if users.empty: return None
    
    user_row = users[users['email'] == email]
    if not user_row.empty:
        user_data = user_row.iloc[0]
        # Helper to safely convert to bool
        is_approved = str(user_data['approved']).lower() in ('true', '1', 'yes')
        
        return {
            "email": user_data['email'],
            "name": user_data['name'],
            "approved": is_approved,
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

def update_password(email, new_password):
    """Updates a user's password with a new one (hashed)."""
    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        client = get_connection()
        sheet = client.open(SHEET_NAME).sheet1
        cell = sheet.find(email)
        if cell:
            # Column 2 is password
            sheet.update_cell(cell.row, 2, hashed_pw)
            return True
    except Exception as e:
        st.error(f"비밀번호 변경 실패: {e}")
    return False

def increment_analysis_count(email):
    """Increments the analysis_count for a specific user."""
    try:
        client = get_connection()
        sheet = client.open(SHEET_NAME).sheet1
        cell = sheet.find(email)
        if cell:
            users = get_all_users()
            user_data = users[users['email'] == email].iloc[0]
            try:
                current_count = int(user_data.get('analysis_count', 0))
            except:
                current_count = 0
            # Column 7 is analysis_count
            sheet.update_cell(cell.row, 7, current_count + 1)
    except:
        pass

def reset_password(email):
    """Generates a new random password, updates the DB, and returns it."""
    import secrets
    import string
    
    # Generate random password
    alphabet = string.ascii_letters + string.digits
    new_password = ''.join(secrets.choice(alphabet) for i in range(10))
    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    try:
        client = get_connection()
        sheet = client.open(SHEET_NAME).sheet1
        cell = sheet.find(email)
        if cell:
            # Column 2 is password
            sheet.update_cell(cell.row, 2, hashed_pw)
            return new_password
    except Exception as e:
        st.error(f"비밀번호 초기화 실패: {e}")
    
    return None
