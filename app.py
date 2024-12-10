import gradio as gr
import numpy as np
from PIL import Image
import os
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler
)
import logging
import sys
from safetensors.torch import load_file
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", message="You have disabled the safety checker")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure PyTorch
torch._dynamo.config.suppress_errors = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCHEDULERS = {
    "DPM++ 2M": lambda: DPMSolverMultistepScheduler(steps_offset=1),
    "UniPC": lambda: UniPCMultistepScheduler(),
    "PNDM": lambda: PNDMScheduler(),
    "DDIM": lambda: DDIMScheduler(),
    "LMS": lambda: LMSDiscreteScheduler(),
    "Euler": lambda: EulerDiscreteScheduler(),
    "Euler A": lambda: EulerAncestralDiscreteScheduler(),
    "Heun": lambda: HeunDiscreteScheduler(),
    "DPM2": lambda: KDPM2DiscreteScheduler(steps_offset=1),
    "DPM2 A": lambda: KDPM2AncestralDiscreteScheduler(),
}

def create_directories():
    directories = [
        'models/stable-diffusion',
        'models/lora',
        'output_images'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def get_available_models():
    models_dir = os.path.join('models', 'stable-diffusion')
    models = []
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.safetensors'):
                models.append(file)
    logger.info(f"Found models: {models if models else 'No models found'}")
    return models if models else ["No models found"]

def get_available_loras():
    lora_dir = os.path.join('models', 'lora')
    loras = ["None"]
    if os.path.exists(lora_dir):
        for file in os.listdir(lora_dir):
            if file.endswith('.safetensors'):
                loras.append(file)
    logger.info(f"Found LoRAs: {loras if len(loras) > 1 else 'No LoRAs found'}")
    return loras

def optimize_pipeline(pipe):
    """Apply various optimizations to the pipeline"""
    if torch.cuda.is_available():
        # Enable memory efficient attention
        pipe.enable_attention_slicing(1)
        
        # Try to enable xformers if available
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.info(f"Could not enable xformers: {e}")
        
        # Enable model offloading
        pipe.enable_model_cpu_offload()
        logger.info("Enabled CPU offloading")
        
        # Enable sequential CPU offloading
        pipe.enable_sequential_cpu_offload()
        logger.info("Enabled sequential CPU offloading")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    return pipe

def load_lora(pipe, lora_path, alpha):
    if not lora_path or lora_path == "None":
        return pipe
    
    logger.info(f"Loading LoRA from {lora_path} with alpha {alpha}")
    state_dict = load_file(lora_path)
    
    visited = []
    
    for key in state_dict:
        # Extract the LoRA suffix (if any)
        if ".alpha" in key or key in visited:
            continue
            
        if "lora_down" in key:
            suffix = key.split("_down")[0]
            # Find target layer
            target_down = key
            target_up = key.replace("lora_down", "lora_up")
            target_shape = state_dict[target_down].shape
            
            # Merge LoRA weights
            if target_up in state_dict:
                down = state_dict[target_down].float()
                up = state_dict[target_up].float()
                
                if len(target_shape) == 4:
                    # For 2D convolutions
                    _, _, kernel_size, _ = up.shape
                    scale = alpha * (kernel_size ** -2)
                else:
                    scale = alpha
                    
                # Compute merged weights
                weights = torch.mm(up.reshape(up.shape[0], -1), 
                                 down.reshape(down.shape[0], -1))
                weights = weights.reshape(up.shape[0], down.shape[1])
                weights = weights * scale
                
                # Find original layer
                layer_name = suffix.split("lora_")[0]
                curr_layer = pipe
                for name in layer_name.split("."):
                    curr_layer = getattr(curr_layer, name)
                    
                if hasattr(curr_layer, "weight"):
                    curr_layer.weight.data += weights.to(curr_layer.weight.device)
                    visited.append(target_down)
                    visited.append(target_up)
                    
    return pipe

def truncate_prompt(prompt, max_length=77):
    """Truncate prompt to fit within CLIP's token limit"""
    words = prompt.split()
    truncated = []
    current_length = 0
    
    for word in words:
        # Rough estimation: each word is at least one token
        if current_length + 1 > max_length:
            break
        truncated.append(word)
        current_length += 1
    
    return " ".join(truncated)

def generate_image(model_name, scheduler_name, lora_name, lora_alpha, prompt, negative_prompt, 
                  steps, width, height, cfg_scale, seed, denoising_strength):
    logger.info(f"Starting image generation with parameters:")
    logger.info(f"Model: {model_name}")
    logger.info(f"Scheduler: {scheduler_name}")
    logger.info(f"LoRA: {lora_name} (alpha: {lora_alpha})")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Steps: {steps}, Width: {width}, Height: {height}, CFG: {cfg_scale}, Seed: {seed}")
    logger.info(f"Denoising Strength: {denoising_strength}")

    try:
        # Truncate prompts to avoid CLIP warning
        prompt = truncate_prompt(prompt)
        if negative_prompt:
            negative_prompt = truncate_prompt(negative_prompt)
        
        model_path = os.path.join('models', 'stable-diffusion', model_name)
        if not os.path.exists(model_path):
            error_msg = f"No model found at {model_path}. Please place your Stable Diffusion model files in the models/stable-diffusion directory."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info("Initializing StableDiffusionPipeline...")
        scheduler = SCHEDULERS[scheduler_name]()
        
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            variant="fp16" if torch.cuda.is_available() else None,
            scheduler=scheduler
        )
        
        if torch.cuda.is_available():
            # Clear CUDA cache before loading model
            torch.cuda.empty_cache()
            pipe = pipe.to("cuda")
            pipe = optimize_pipeline(pipe)
            logger.info("Using CUDA for generation")
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        else:
            logger.info("Using CPU for generation")
        
        # Load LoRA if selected
        if lora_name and lora_name != "None":
            lora_path = os.path.join('models', 'lora', lora_name)
            pipe = load_lora(pipe, lora_path, lora_alpha)
        
        generator = None
        if seed != -1:
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
            logger.info(f"Using seed: {seed}")
        
        logger.info("Generating image...")
        with torch.inference_mode(), torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
                guidance_scale=cfg_scale,
                generator=generator,
                strength=denoising_strength
            ).images[0]
        
        # Clear CUDA cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Final CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"Final CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        os.makedirs('output_images', exist_ok=True)
        output_path = os.path.join('output_images', f'generated_{seed}.png')
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
        
        return image, None  # Return None as error message when successful
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache on error
        return None, error_msg  # Return error message to display in UI

def create_ui():
    create_directories()
    logger.info("Creating UI...")

    with gr.Blocks(title="Simple Stable Diffusion", theme=gr.themes.Default()) as app:
        gr.Markdown("# Simple Stable Diffusion")
        
        with gr.Row():
            # Left column for inputs
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Model Settings")
                    model_dropdown = gr.Dropdown(
                        choices=get_available_models(),
                        value=get_available_models()[0],
                        label="Model",
                        container=True
                    )
                    scheduler_dropdown = gr.Dropdown(
                        choices=list(SCHEDULERS.keys()),
                        value="DPM++ 2M",
                        label="Scheduler",
                        container=True
                    )
                
                with gr.Group():
                    gr.Markdown("### LoRA Settings")
                    lora_dropdown = gr.Dropdown(
                        choices=get_available_loras(),
                        value="None",
                        label="LoRA Model",
                        container=True
                    )
                    lora_alpha = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.75,
                        step=0.05,
                        label="LoRA Weight"
                    )
                
                with gr.Group():
                    gr.Markdown("### Prompt Settings")
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3,
                        container=True
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Enter negative prompt here...",
                        lines=3,
                        container=True
                    )
                
                with gr.Group():
                    gr.Markdown("### Generation Settings")
                    with gr.Row():
                        steps = gr.Slider(minimum=1, maximum=150, value=20, step=1, label="Sampling Steps")
                        cfg_scale = gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label="CFG Scale")
                    
                    with gr.Row():
                        width = gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Width")
                        height = gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Height")
                    
                    with gr.Row():
                        seed = gr.Number(value=-1, label="Seed", precision=0)
                        randomize_seed = gr.Button("🎲")
                    
                    denoising_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.01,
                        label="Denoising Strength"
                    )
                
                # Generate button directly under controls
                generate_btn = gr.Button(
                    "Generate Image",
                    variant="primary",
                    size="lg",
                    scale=2,
                    min_width=300,
                    interactive=True
                )
            
            # Right column for output
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", height=512)
                error_output = gr.Textbox(label="Error Messages", visible=True)

        def generate_wrapper(*args):
            image, error = generate_image(*args)
            return [image, error if error else ""]

        generate_btn.click(
            fn=generate_wrapper,
            inputs=[
                model_dropdown,
                scheduler_dropdown,
                lora_dropdown,
                lora_alpha,
                prompt,
                negative_prompt,
                steps,
                width,
                height,
                cfg_scale,
                seed,
                denoising_strength
            ],
            outputs=[output_image, error_output]
        )
        
        def randomize_seed_fn():
            return np.random.randint(0, 1000000)
        
        randomize_seed.click(fn=randomize_seed_fn, outputs=[seed])

    return app

if __name__ == "__main__":
    logger.info("Starting Simple Stable Diffusion application...")
    
    # Log CUDA information
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    
    app = create_ui()
    
    # Try different ports if 7860 is occupied
    for port in range(7860, 7870):
        try:
            logger.info(f"Attempting to launch on port {port}")
            app.launch(
                server_name="127.0.0.1",  # Only allow local connections
                server_port=port,         # Try ports 7860-7869
                share=False,              # Disable sharing
                show_error=True,          # Show errors in UI
                inbrowser=True           # Open in browser automatically
            )
            break
        except OSError as e:
            logger.warning(f"Port {port} is in use, trying next port")
            if port == 7869:  # Last port in range
                logger.error("No available ports found in range 7860-7869")
                raise e
