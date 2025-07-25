# EW-PSSK 實驗運行腳本 (PowerShell)
# 使用方法: .\run_experiment.ps1

Write-Host "EW-PSSK 抗癌胜肽預測實驗" -ForegroundColor Green
Write-Host ("="*50) -ForegroundColor Green

# 檢查 Python 是否安裝
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python 版本: $pythonVersion" -ForegroundColor Blue
} catch {
    Write-Host "錯誤: 找不到 Python，請確保 Python 已安裝並添加到 PATH" -ForegroundColor Red
    exit 1
}

# 檢查數據集文件
$datasetPath = "src\dataset\acp740.txt"
if (-not (Test-Path $datasetPath)) {
    Write-Host "錯誤: 找不到數據集文件 $datasetPath" -ForegroundColor Red
    exit 1
}

Write-Host "數據集文件: $datasetPath" -ForegroundColor Blue

# 安裝依賴包 (可選)
$installDeps = Read-Host "是否安裝/更新依賴包？ (y/N)"
if ($installDeps -eq "y" -or $installDeps -eq "Y") {
    Write-Host "安裝依賴包..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "警告: 依賴包安裝可能有問題，但將繼續執行" -ForegroundColor Yellow
    }
}

# 運行實驗
Write-Host "開始運行 EW-PSSK 實驗..." -ForegroundColor Green
Write-Host ("="*50) -ForegroundColor Green

$startTime = Get-Date

try {
    python main.py --dataset $datasetPath --gamma 1.0 --C 1.0 --cv_folds 10 --kernel_method linear --max_length 50 --random_state 42
    
    $endTime = Get-Date
    $totalTime = $endTime - $startTime
    
    Write-Host ("="*50) -ForegroundColor Green
    Write-Host "實驗成功完成！" -ForegroundColor Green
    Write-Host "總執行時間: $($totalTime.TotalSeconds.ToString('F2')) 秒" -ForegroundColor Blue
    Write-Host "結果保存在 results/ 目錄中" -ForegroundColor Blue
    
} catch {
    Write-Host "實驗執行失敗: $_" -ForegroundColor Red
    exit 1
}

Write-Host "按任意鍵退出..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
