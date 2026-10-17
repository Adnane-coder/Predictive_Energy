@echo off
echo Lancement PredictiveEnergy...

start "API FastAPI" cmd /k "cd /d C:\Users\MSI\Predictive_Energy\api && conda activate base && uvicorn main:app --reload --port 8000"

timeout /t 6

start "Dashboard" cmd /k "cd /d C:\Users\MSI\Predictive_Energy\dashboard && conda activate base && streamlit run app.py"

timeout /t 10

start http://localhost:8501

echo Tout est lance !