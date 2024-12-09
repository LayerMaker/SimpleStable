import gradio as gr
import numpy as np
from PIL import Image
import os
import torch
from diffusers import StableDiffusionPipeline
import logging
import sys

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

def create_directories():
    directories = [
        'models/stable-diffusion',
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

def generate_image(model_name, prompt, negative_prompt, steps, width, height, cfg_scale, seed):
    logger.info(f"Starting image generation with parameters:")
    logger.info(f"Model: {model_name}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Steps: {steps}, Width: {width}, Height: {height}, CFG: {cfg_scale}, Seed: {seed}")

    try:
        model_path = os.path.join('models', 'stable-diffusion', model_name)
        if not os.path.exists(model_path):
            error_msg = f"No model found at {model_path}. Please place your Stable Diffusion model files in the models/stable-diffusion directory."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info("Initializing StableDiffusionPipeline...")
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            logger.info("Using CUDA for generation")
        else:
            logger.info("Using CPU for generation")
        
        generator = None
        if seed != -1:
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
            logger.info(f"Using seed: {seed}")
        
        logger.info("Generating image...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]
        
        os.makedirs('output_images', exist_ok=True)
        output_path = os.path.join('output_images', f'generated_{seed}.png')
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
        
        return image, None  # Return None as error message when successful
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg  # Return error message to display in UI

def create_ui():
    create_directories()
    logger.info("Creating UI...")

    with gr.Blocks(title="Simple Stable Diffusion", theme=gr.themes.Default()) as app:
        gr.Markdown("# Simple Stable Diffusion")
        
        with gr.Row():
            # Left column for inputs
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    value=get_available_models()[0],
                    label="Model",
                    container=True
                )
                
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
                    with gr.Row():
                        steps = gr.Slider(minimum=1, maximum=150, value=20, step=1, label="Sampling Steps")
                        cfg_scale = gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label="CFG Scale")
                    
                    with gr.Row():
                        width = gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Width")
                        height = gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Height")
                    
                    with gr.Row():
                        seed = gr.Number(value=-1, label="Seed", precision=0)
                        randomize_seed = gr.Button("🎲")
                
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
                prompt,
                negative_prompt,
                steps,
                width,
                height,
                cfg_scale,
                seed
            ],
            outputs=[output_image, error_output]
        )
        
        def randomize_seed_fn():
            return np.random.randint(0, 1000000)
        
        randomize_seed.click(fn=randomize_seed_fn, outputs=[seed])

    return app

if __name__ == "__main__":
    logger.info("Starting Simple Stable Diffusion application...")
    app = create_ui()
    # Launch with fixed port 7860
    app.launch(
        server_name="127.0.0.1",  # Only allow local connections
        server_port=7860,         # Fixed port
        share=False,              # Disable sharing
        show_error=True,          # Show errors in UI
        inbrowser=True            # Open in browser automatically
    )
