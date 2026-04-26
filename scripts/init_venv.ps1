$ErrorActionPreference = "Stop"

function Ask-YesNo {
    param(
        [string]$Question,
        [bool]$Default = $true
    )

    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    while ($true) {
        $answer = Read-Host "$Question $suffix"
        if ([string]::IsNullOrWhiteSpace($answer)) {
            return $Default
        }
        switch ($answer.Trim().ToLowerInvariant()) {
            "y" { return $true }
            "yes" { return $true }
            "n" { return $false }
            "no" { return $false }
        }
        Write-Host "y / n"
    }
}

function Ask-Choice {
    param(
        [string]$Question,
        [string[]]$Choices,
        [string]$Default
    )

    $choiceText = ($Choices -join "/")
    while ($true) {
        $answer = Read-Host "$Question [$choiceText] (default: $Default)"
        if ([string]::IsNullOrWhiteSpace($answer)) {
            return $Default
        }
        foreach ($choice in $Choices) {
            if ($answer.Trim().ToLowerInvariant() -eq $choice.ToLowerInvariant()) {
                return $choice
            }
        }
        Write-Host "選択肢: $choiceText"
    }
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

if (-not (Test-Path ".venv")) {
    Write-Host ".venv を作成します..."
    python -m venv .venv
}

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "仮想環境の Python が見つかりません: $pythonExe"
}

Write-Host "pip を更新します..."
& $pythonExe -m pip install --upgrade pip

Write-Host "基本依存をインストールします..."
& $pythonExe -m pip install -r requirements.txt

$installTorch = Ask-YesNo "PyTorch をインストールしますか?" $true
if ($installTorch) {
    $torchMode = Ask-Choice "どの PyTorch を入れますか?" @("cpu", "gpu", "skip") "cpu"
    switch ($torchMode) {
        "cpu" {
            Write-Host "PyTorch CPU 版をインストールします..."
            & $pythonExe -m pip install torch
        }
        "gpu" {
            Write-Host "PyTorch GPU(CUDA 12.8) 版をインストールします..."
            & $pythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
        }
        "skip" {
            Write-Host "PyTorch のインストールをスキップしました。"
        }
    }
} else {
    Write-Host "PyTorch のインストールをスキップしました。"
}

$buildCpp = Ask-YesNo "C++ エンジンをビルドしますか?" $true
if ($buildCpp) {
    Write-Host "C++ エンジンをビルドします..."
    & $pythonExe setup.py build_ext --inplace
}

$checkTorch = Ask-YesNo "PyTorch / CUDA 状態を確認しますか?" $true
if ($checkTorch) {
    & $pythonExe -c "import importlib.util; spec = importlib.util.find_spec('torch'); print('torch_installed', spec is not None); import sys;`nif spec is not None:`n import torch; print('torch', torch.__version__); print('cuda_build', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"
}

Write-Host ""
Write-Host "セットアップ完了"
Write-Host "起動方法:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python .\sesamum.py"
