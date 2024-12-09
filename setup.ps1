Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

Write-Host "Installing required packages..."
pip install transformers
pip install torch
pip install numpy==1.23.5
pip install matplotlib

Write-Host "Setup complete. Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
