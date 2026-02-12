@echo off
echo ========================================================
echo  UPLOADING CODE TO SERVER (Skipping .git and huge files)
echo ========================================================

:: 1. Upload individual files
echo [1/2] Uploading core files (app.py, matcher.py, etc)...
scp -i "your_pem.pem" app.py run_cli.py config.py matcher.py prompts.py requirements.txt .env ubuntu@your-ec2-ip:~/saudi_matcher/

:: 2. Upload directories (data and docs)
echo [2/2] Uploading folders (data)...
:: Note: This might take time if data is large
scp -i "your_pem.pem" -r data ubuntu@your-ec2-ip:~/saudi_matcher/

echo.
echo ========================================================
echo  UPLOAD COMPLETE!
echo ========================================================
pause
