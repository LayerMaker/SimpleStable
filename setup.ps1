Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

Write-Host "Installing PyTorch with CUDA support and other required packages..."
# Install PyTorch with CUDA support for any NVIDIA GPU
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

Write-Host "Installing other dependencies..."
pip install transformers
pip install numpy==1.23.5
pip install matplotlib
pip install diffusers

Write-Host "Verifying CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

Write-Host "Setup complete. Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
