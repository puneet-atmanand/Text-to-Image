import streamlit as st
from image_generator import ImageGenerator
import os
from PIL import Image

# Initialize the generator once (cache resource)
@st.cache_resource
def get_generator(use_lcm):
    return ImageGenerator(use_lcm=use_lcm)

st.set_page_config(page_title="Text-to-Image Generator", page_icon="🎨")

st.title("🎨 Text-to-Image Generator")
st.markdown("Generate images from text prompts using Stable Diffusion.")

# Sidebar for parameters
# Sidebar for parameters
st.sidebar.header("Parameters")

# Fast Mode (LCM) Toggle
use_lcm = st.sidebar.checkbox("⚡ Fast Mode (CPU Optimized)", value=True, help="Uses LCM LoRA for much faster generation (4-8 steps). Recommended for CPU.")

# Dynamic defaults based on mode
default_steps = 4 if use_lcm else 50
default_guidance = 1.0 if use_lcm else 7.5

steps = st.sidebar.slider("Number of Inference Steps", min_value=1, max_value=100, value=default_steps)
guidance_scale = st.sidebar.slider("Guidance Scale", min_value=0.0, max_value=20.0, value=default_guidance)
width = st.sidebar.select_slider("Width", options=[256, 512, 768, 1024], value=512)
height = st.sidebar.select_slider("Height", options=[256, 512, 768, 1024], value=512)
num_images = st.sidebar.number_input("Number of Images", min_value=1, max_value=4, value=1)
seed_input = st.sidebar.text_input("Seed (Optional)", value="")

# Main area
prompt = st.text_area("Enter your prompt:", height=100, placeholder="A futuristic city with flying cars...")
negative_prompt = st.text_input("Negative Prompt (Optional):", placeholder="blurry, bad quality")

if st.button("Generate Image"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image... This may take a while depending on your hardware."):
            try:
                generator = get_generator(use_lcm)
                
                seed = int(seed_input) if seed_input.isdigit() else None
                
                saved_paths = generator.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images,
                    seed=seed
                )
                
                if saved_paths:
                    st.success(f"Generated {len(saved_paths)} image(s)!")
                    
                    # Display images
                    cols = st.columns(len(saved_paths))
                    for idx, path in enumerate(saved_paths):
                        image = Image.open(path)
                        with cols[idx]:
                            st.image(image, caption=f"Image {idx+1}", use_column_width=True)
                            
                            with open(path, "rb") as file:
                                btn = st.download_button(
                                    label="Download Image",
                                    data=file,
                                    file_name=os.path.basename(path),
                                    mime="image/png"
                                )
                else:
                    st.error("Failed to generate image. Please check the logs.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("Created with ❤️ by Puneet Atmanand Iti.")
