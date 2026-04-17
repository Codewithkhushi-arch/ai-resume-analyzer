@echo off
echo Installing required python packages...
pip install -r requirements.txt
echo.
echo Starting ResumeIQ on your browser...
python -m streamlit run app.py
pause
