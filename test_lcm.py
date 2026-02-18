import time
from image_generator import ImageGenerator
import os

def test_lcm_generation():
    print("Initializing Generator with Fast Mode (LCM)...")
    start_setup = time.time()
    generator = ImageGenerator(use_lcm=True)
    setup_time = time.time() - start_setup
    print(f"Setup time: {setup_time:.2f}s")

    prompt = "A futuristic city with flying cars, cinematic lighting, high detail"
    print(f"Generating image for prompt: '{prompt}'")
    
    start_gen = time.time()
    # Use typical LCM settings: 4 steps, guidance 1.0
    saved_files = generator.generate(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=1.0,
        num_images_per_prompt=1
    )
    gen_time = time.time() - start_gen
    
    print(f"Generation time: {gen_time:.2f}s")
    
    if saved_files and os.path.exists(saved_files[0]):
        print(f"Success! Image saved to {saved_files[0]}")
    else:
        print("Failed to generate image.")

if __name__ == "__main__":
    test_lcm_generation()
