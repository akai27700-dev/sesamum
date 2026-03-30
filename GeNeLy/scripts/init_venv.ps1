$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& ".\.venv\Scripts\python.exe" setup.py build_ext --inplace

Write-Host "PyTorch check:"
& ".\.venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda_build', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"

Write-Host ""
Write-Host "Virtual environment is ready."
Write-Host "Run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python .\Pattern_Firing_Light_NN_MCTS_TT_Standerdmove_PVS_LMR_Killer_History_NNPolicy_C++_ver.py"
