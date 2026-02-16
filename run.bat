
@echo off
echo [Setting up Environment...]
pip install -r requirements.txt
echo.
echo [Running Win Strategy App...]
streamlit run app.py
pause
