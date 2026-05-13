@echo off
echo Instalimi i dependencies...
pip install -r requirements.txt --quiet
echo.
echo Duke nisur XAUUSD Predictor...
echo Hap browser-in tek: http://localhost:8501
echo.
streamlit run app.py
pause
