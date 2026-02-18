🎨 Text2Image Diffusion Engine
A Text-to-Image generation system powered by Stable Diffusion that converts natural language prompts into high-quality images.
The project provides both a Command Line Interface (CLI) and an interactive Streamlit Web UI with automatic hardware optimization and fast generation support.

🚀 Features

- 🧠 Text-to-Image generation using Stable Diffusion
- ⚡ Fast Mode (LCM) for faster CPU inference
- 💻 Command Line Interface (CLI)
- 🌐 Streamlit Web Interface
- 🎛️ Fully customizable generation parameters
- 🔍 Automatic device detection (CUDA / MPS / CPU)
- 🎯 Seed support for reproducible results
- 🚫 Negative prompt support
- 💾 Automatic image saving

📁 Project Structure

├── main.py                # CLI entry point
├── streamlit_app.py       # Streamlit Web UI
├── model_loader.py        # Model loading & device optimization
├── image_generator.py     # Image generation logic
├── download_model.py      # Model downloader
├── test_installation.py   # Environment verification
├── test_lcm.py            # Fast mode testing
├── requirements.txt       # Dependencies
└── outputs/               # Generated images

⚙️ Installation

📥 Clone Repository

```
git clone https://github.com/your-username/text2image-diffusion-engine.git
cd text2image-diffusion-engine
```
📦 Install Dependencies

```
pip install -r requirements.txt
```

---

✅ Verify Installation (Optional)

```
python test_installation.py
```

---

🖥️ Usage

▶️ Command Line Interface (CLI)

Basic usage:

```
python main.py "A futuristic city in the clouds"
```

Advanced usage:

```
python main.py "Cyberpunk city at night" --negative_prompt "blurry, low quality" --steps 30 --guidance_scale 8.0 --width 512 --height 512 --num_images 2
```

---

🌐 Streamlit Web Interface

Run the web application:

```
streamlit run streamlit_app.py
```

Then open the browser and generate images interactively.

---

⚡ Fast Mode (LCM)

Fast Mode significantly reduces generation time, especially on CPU devices.

Enable Fast Mode:

```
python main.py "A fantasy landscape" --fast
```

Typical settings:
- ⚡ Steps: 4 to 8
- 🎯 Guidance Scale: 1.0

---
🎛️ Parameters

- 📝 Prompt — Text description of the image
- 🚫 Negative Prompt — Elements to avoid
- ⏱️ Inference Steps — Controls quality vs speed
- 🎯 Guidance Scale — Prompt adherence strength
- 🖼️ Image Size — Width and height
- 🔁 Seed — Reproducible outputs
- 🧩 Number of Images — Batch generation

---

🧠 How It Works

1. Text prompt is converted into embeddings using a text encoder.
2. Stable Diffusion starts from random noise.
3. The diffusion process iteratively denoises guided by the prompt.
4. The final latent representation is decoded into an image.

---

💻 Requirements

- Python 3.8 or higher
- Recommended GPU with at least 4GB VRAM
- CPU generation supported (slower but optimized)

---

📦 Dependencies

- torch
- diffusers
- transformers
- accelerate
- streamlit
- Pillow
- peft

Install automatically:

```
pip install -r requirements.txt
```

---

📸 Output

Generated images are automatically saved inside:

```
outputs/
```

---

🛠️ Troubleshooting

Out of Memory:
- Reduce image size (256x256)
- Generate fewer images
- Use Fast Mode

Slow Generation:
- CPU inference is slower
- Use GPU if available

