Write-Host "Starting Simple Stable Diffusion..."

# Activate the virtual environment
.\venv_stable\Scripts\Activate.ps1

# Check if CUDA is available
Write-Host "`nChecking CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Launch the application
Write-Host "`nLaunching application..."
python app.py

# If there was an error, pause to show the message
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nAn error occurred. Check the error message above."
    Write-Host "Press any key to continue..."
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
}
