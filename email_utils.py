
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

def send_approval_email(to_email):
    """
    Sends an approval notification email to the user.
    Uses credentials from st.secrets.
    """
    try:
        # Load secrets
        smtp_secrets = st.secrets["smtp"]
        sender_email = smtp_secrets["email_sender"]
        password = smtp_secrets["email_password"]
        
        # Email Content
        subject = "[수주비책] 회원가입 승인 안내"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: #f8f9fa; padding: 20px;">
                <h2 style="color: #1E3A8A;">수주비책 (Win Strategy)</h2>
                <p>안녕하세요.</p>
                <p>귀하의 <strong>수주비책</strong> 서비스 이용 권한이 <span style="color: #10B981; font-weight: bold;">승인되었습니다.</span></p>
                <p>지금 바로 로그인하여 입찰 제안 전략 분석을 시작해보세요.</p>
                <hr>
                <p style="font-size: 12px; color: #6b7280;">본 메일은 발신 전용입니다.</p>
            </div>
        </body>
        </html>
        """
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        # Send Email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            
        return True, "이메일 전송 성공"
        
    except Exception as e:
        return False, f"이메일 전송 실패: {e}"

def send_admin_notification(new_user_email, new_user_name):
    """
    Sends a notification email to the admin when a new user signs up.
    """
    try:
        # Load secrets
        smtp_secrets = st.secrets["smtp"]
        sender_email = smtp_secrets["email_sender"]
        password = smtp_secrets["email_password"]
        
        # Admin email (from secrets or fallback to sender)
        try:
            admin_email = st.secrets["admin"].get("email", sender_email)
        except:
             admin_email = sender_email
        
        # Email Content
        subject = f"[수주비책] 신규 회원가입 요청 ({new_user_name})"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: #fff3cd; padding: 20px;">
                <h2 style="color: #856404;">신규 회원가입 알림</h2>
                <p>새로운 사용자가 가입 승인을 요청했습니다.</p>
                <ul>
                    <li><strong>이름:</strong> {new_user_name}</li>
                    <li><strong>이메일:</strong> {new_user_email}</li>
                </ul>
                <p>관리자 대시보드에서 승인 처리를 진행해주세요.</p>
                <hr>
                <p style="font-size: 12px; color: #6b7280;">System Notification</p>
            </div>
        </body>
        </html>
        """
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = admin_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        # Send Email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            
        return True, "관리자 알림 전송 성공"
        
    except Exception as e:
        return False, f"관리자 알림 전송 실패: {e}"

def send_password_reset_email(to_email, new_password):
    """
    Sends a new temporary password to the user.
    """
    try:
        smtp_secrets = st.secrets["smtp"]
        sender_email = smtp_secrets["email_sender"]
        password = smtp_secrets["email_password"]
        
        subject = "[수주비책] 임시 비밀번호 안내"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: #fce7f3; padding: 20px;">
                <h2 style="color: #be185d;">수주비책 비밀번호 초기화</h2>
                <p>안녕하세요.</p>
                <p>요청하신 비밀번호 초기화가 완료되었습니다.</p>
                <div style="background-color: #ffffff; padding: 15px; border-radius: 5px; border: 1px solid #f9a8d4; text-align: center; margin: 20px 0;">
                    <p style="margin: 0; font-size: 14px; color: #6b7280;">임시 비밀번호</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold; color: #db2777; letter-spacing: 2px;">{new_password}</p>
                </div>
                <p>로그인 후 반드시 비밀번호를 변경하시기 바랍니다.</p>
                <hr>
                <p style="font-size: 12px; color: #6b7280;">보안을 위해 임시 비밀번호를 타인에게 노출하지 마세요.</p>
            </div>
        </body>
        </html>
        """
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            
        return True, "비밀번호 안내 메일 전송 성공"
    except Exception as e:
        return False, f"메일 전송 실패: {e}"
