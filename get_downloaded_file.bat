@echo off
echo ========================================================
echo  DOWNLOADING RESULTS FROM SERVER
echo ========================================================

:: 1. Download the output file
:: Make sure to edit 'your_pem.pem' and 'your-ec2-ip' before running!
scp -i "your_pem.pem" ubuntu@your-ec2-ip:~/saudi_matcher/data/output/final_output.xlsx ./downloaded_output.xlsx

echo.
echo ========================================================
if exist "downloaded_output.xlsx" (
    echo  DOWNLOAD COMPLETE! File saved as 'downloaded_output.xlsx'
) else (
    echo  DOWNLOAD FAILED or file not found on server.
)
echo ========================================================
pause
