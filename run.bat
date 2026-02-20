
@echo off
echo [Setting up Environment...]
py -m pip install -r requirements.txt
echo.
echo [Running Win Strategy App...]
py -m streamlit run app.py
pause
