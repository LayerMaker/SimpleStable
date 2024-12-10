# Run as administrator check
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Warning "Script is not running as Administrator. Some operations might fail."
}

# Set environment variables
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
$pythonExe = "C:\Users\crock\AppData\Local\Programs\Python\Python310\python.exe"

Write-Host "Cleaning up any existing environment..."
if (Test-Path "venv_stable") {
    Get-Process python* | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Remove-Item -Recurse -Force "venv_stable" -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host "Creating new virtual environment..."
try {
    & $pythonExe -c @"
import venv
import sys
venv.create('venv_stable', clear=True, with_pip=True, system_site_packages=False)
"@
    Start-Sleep -Seconds 2
} catch {
    Write-Error "Failed to create virtual environment: $_"
    exit 1
}

$pipExe = ".\venv_stable\Scripts\python.exe"
if (-not (Test-Path $pipExe)) {
    Write-Error "Virtual environment creation failed - pip not found"
    exit 1
}

Write-Host "Installing PyTorch with CUDA 12.4 support..."
& $pipExe -m pip install --upgrade pip
& $pipExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing accelerate for GPU optimization..."
& $pipExe -m pip install accelerate

Write-Host "Installing other dependencies..."
& $pipExe -m pip install transformers
& $pipExe -m pip install numpy==1.23.5
& $pipExe -m pip install matplotlib
& $pipExe -m pip install gradio
& $pipExe -m pip install diffusers

Write-Host "Verifying CUDA availability and configuration..."
& $pipExe -c @"
import torch
import sys
print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU Device:', torch.cuda.get_device_name(0))
    print('GPU Count:', torch.cuda.device_count())
    print('Current GPU:', torch.cuda.current_device())
    print('GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1024**2, 'MB')
    print('\nMemory Optimizations:')
    print('torch.compile available:', hasattr(torch, 'compile'))
    print('\nCUDA Memory Settings:')
    print('Memory allocated:', torch.cuda.memory_allocated() / 1024**2, 'MB')
    print('Memory reserved:', torch.cuda.memory_reserved() / 1024**2, 'MB')
    print('Max memory allocated:', torch.cuda.max_memory_allocated() / 1024**2, 'MB')
    print('Memory allocation config:', torch.cuda.get_allocator_backend())
    
    # Test CUDA computation
    print('\nTesting CUDA Computation:')
    test_tensor = torch.randn(1000, 1000).cuda()
    result = torch.matmul(test_tensor, test_tensor.t())
    print('CUDA computation test successful')
    del test_tensor, result
    torch.cuda.empty_cache()
else:
    print('CUDA not available. Checking why:')
    print('CUDA built:', torch.backends.cuda.is_built())
    if hasattr(torch.backends.cuda, 'get_device_capability'):
        print('Device capability:', torch.backends.cuda.get_device_capability())
"@

Write-Host "`nSetup complete. Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
