Write-Host "Creating new virtual environment..."
python -m venv venv_stable

Write-Host "Activating virtual environment..."
.\venv_stable\Scripts\Activate.ps1

Write-Host "Installing required packages..."
pip install transformers
pip install torch
pip install numpy==1.23.5
pip install matplotlib
pip install gradio
pip install diffusers

Write-Host "Setup complete. Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
