# Simple Stable Diffusion UI

A simplified version of the Stable Diffusion Web UI using Gradio, focused on providing a streamlined experience for text-to-image generation.

## Features

- Model Selection
- Prompt and Negative Prompt Input
- Basic Generation Controls:
  - Width and Height
  - Sampling Steps
  - CFG Scale
  - Seed Control
- Sampling Methods and Scheduler Options
- LoRA Integration with Preset Weights
- Control Net Options:
  - OpenPose
  - Depth
  - Canny
  - IP Adapter

## Directory Structure

```
SimpleStable/
├── models/
│   ├── stable-diffusion/  # Place your base SD model here
│   ├── controlnet/        # ControlNet models
│   ├── lora/             # LoRA models
│   └── vae/              # VAE models
├── input_images/          # For ControlNet and IP Adapter inputs
├── output_images/         # Generated images are saved here
└── lora_previews/        # LoRA preview thumbnails
```

## Model Setup

1. Place your Stable Diffusion v1.5 model in the `models/stable-diffusion` directory
   - Rename the model file to `v1-5-pruned.safetensors`
2. (Optional) Place any ControlNet models in `models/controlnet`
3. (Optional) Place LoRA models in `models/lora`
4. (Optional) Place custom VAE models in `models/vae`

## Installation

1. Clone this repository
2. Run the provided launch script:
```bash
launch_simplestable.bat
```

This will:
- Create a virtual environment if it doesn't exist
- Install all required dependencies
- Launch the application

Alternatively, you can manually install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using the Launch Script (Recommended)
Simply run `launch_simplestable.bat` and the application will start automatically.

### Manual Launch
Run the application:
```bash
python app.py
```

The UI will be available at `http://localhost:7867` by default.

### Configuration

The port can be configured by setting the `SIMPLESTABLE_PORT` environment variable:
```bash
set SIMPLESTABLE_PORT=7867
python app.py
```

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU recommended for faster generation
- At least 8GB of VRAM for optimal performance

## Troubleshooting

If you encounter any issues:

1. Check the console output for detailed error messages
2. Ensure all models are placed in their correct directories
3. Verify CUDA is available if using a GPU
4. Check the virtual environment is activated
5. Make sure all dependencies are installed with the correct versions

## Note

Make sure to place your Stable Diffusion v1.5 model in the correct directory before running the application. The model should be renamed to `v1-5-pruned.safetensors` and placed in the `models/stable-diffusion` directory.

## Error Logging

The application now includes comprehensive error logging that will help diagnose any issues:
- Model loading status
- CUDA/CPU device information
- Memory usage statistics
- Detailed error tracebacks

Check the console output for this information if you encounter any problems during image generation.
