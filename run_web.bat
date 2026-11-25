@echo off
setlocal
title Re:Bakery Web Server

:: Masuk ke direktori file .bat ini berada
cd /d "%~dp0"

echo ===================================================
echo     Re:Bakery - AI Bakery Classifier Web App
echo ===================================================
echo.

:: Cari Python yang terinstall
set "PY_EXE="
if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PY_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
) else (
    for /f "delims=" %%I in ('where python 2^>nul') do (
        if not defined PY_EXE set "PY_EXE=%%I"
    )
)

if not defined PY_EXE (
    echo [ERROR] Python tidak ditemukan di komputer kamu!
    echo Silakan install Python terlebih dahulu.
    echo.
    pause
    exit /b 1
)

:: Cek keberadaan file model
if not exist "backend\models\rebakery.onnx" (
    echo [PERINGATAN] File backend\models\rebakery.onnx belum ditemukan!
    echo Silakan download rebakery.onnx dan labels.json dari Google Drive,
    echo lalu masukkan ke dalam folder backend\models\
    echo.
    pause
    exit /b 1
)

echo [OK] Menggunakan: %PY_EXE%
echo [1/2] Memeriksa paket pendukung...
"%PY_EXE%" -m pip install -r "backend\requirements.txt" -q

echo.
echo [2/2] Menjalankan Server Re:Bakery di http://localhost:8000 ...
echo Buka browser kamu dan akses: http://localhost:8000
echo Tekan Ctrl+C di jendela ini jika ingin mematikan server.
echo.

cd backend
"%PY_EXE%" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
