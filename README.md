# Text-to-Image Generation Project

This project allows you to generate images from text prompts using the Stable Diffusion model. It provides both a Command Line Interface (CLI) and a Web User Interface (Streamlit).

## Features

- **Text-to-Image Generation**: Convert natural language prompts into high-quality images.
- **Customizable Parameters**:
    - **Guidance Scale**: Controls how closely the image follows the prompt.
    - **Inference Steps**: Higher steps generally mean better quality but slower generation.
    - **Image Size**: Adjustable width and height.
    - **Number of Images**: Generate multiple images at once.
    - **Seed**: Use a specific seed for reproducible results.
    - **Negative Prompts**: Specify what you *don't* want in the image.
- **Automatic Device Detection**: Automatically uses GPU (CUDA/MPS) if available, otherwise falls back to CPU with memory optimizations.
- **Two Interfaces**: CLI for quick usage and Streamlit for an interactive experience.

## Prerequisites

- Python 3.8 or higher.
- A GPU with at least 4GB VRAM is recommended for reasonable performance.
- CPU generation is supported but will be significantly slower.

## Installation

1.  **Clone the repository** (or download the files).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Command Line Interface (CLI)

Run `main.py` to generate images from the terminal.

**Basic Usage:**
```bash
python main.py "A futuristic cylinder city in the clouds"
```

**Advanced Usage:**
```bash
python main.py "A futuristic city" --negative_prompt "blurry, low quality" --steps 30 --guidance_scale 8.0 --width 512 --height 512 --num_images 2
```

### 2. Streamlit Web Interface

Run the Streamlit app for an interactive UI.

```bash
streamlit run streamlit_app.py
```
This will open a new tab in your web browser where you can enter prompts and adjust parameters using sliders.

## Project Structure

- `main.py`: Entry point for the CLI.
- `streamlit_app.py`: Entry point for the Streamlit web app.
- `model_loader.py`: Handles loading the Stable Diffusion model and device optimization.
- `image_generator.py`: Contains the logic for generating and saving images.
- `outputs/`: Generated images are saved here automatically.
- `requirements.txt`: List of Python dependencies.

## How it Works

This project uses **Stable Diffusion**, a latent diffusion model.
1.  **Text Encoding**: Your text prompt is converted into a numerical representation (embeddings) using a text encoder (CLIP).
2.  **Diffusion Process**: The model starts with random noise and iteratively "denoises" it, guided by the text embeddings, to form a coherent image.
3.  **Decoding**: The final latent representation is decoded into a visible image.

## Troubleshooting

- **Out of Memory (OOM)**: If you run out of GPU memory, the code automatically tries to enable attention slicing. If that's not enough, try generating smaller images (e.g., 256x256) or reducing the batch size.
- **Slow Generation**: On CPU, generation is expected to be slow (minutes per image). Use a GPU for faster results.
