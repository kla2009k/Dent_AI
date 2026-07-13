# DentScan AI — Phase 1 one-shot setup
# รันตอนพร้อมจะ train จริง (ใช้เน็ต + เวลา: torch ~3GB, data ~10GB)
# วิธีรัน:  ใน PowerShell ที่ folder Project_DentScanAI:  .\setup_phase1.ps1

$ErrorActionPreference = "Stop"
Write-Host "=== DentScan AI — Phase 1 Setup ===" -ForegroundColor Cyan

# 1) PyTorch + CUDA 12.8 (RTX 5060 Blackwell sm_120)
Write-Host "`n[1/4] Installing PyTorch (cu128) ..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) training deps
Write-Host "`n[2/4] Installing training deps ..." -ForegroundColor Yellow
pip install -r requirements.txt

# 3) verify GPU
Write-Host "`n[3/4] Verifying GPU ..." -ForegroundColor Yellow
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU OK:', torch.cuda.get_device_name(0))"

# 4) download training data (10.4 GB)
Write-Host "`n[4/4] Downloading DENTEX training data (10.4 GB) ..." -ForegroundColor Yellow
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ibrahimhamamci/DENTEX', filename='DENTEX/training_data.zip', repo_type='dataset', local_dir='./dentex_data'); print('Download done.')"

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. python scripts/prepare_labels.py      # แตก zip + ทำ label CSV"
Write-Host "  2. python scripts/train_multilabel.py    # train (~30-60 min บน RTX 5060)"
Write-Host "  3. python scripts/inference_gradcam.py dentex_data/DENTEX/validation_data/quadrant_enumeration_disease/xrays/val_0.png"
