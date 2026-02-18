import torch
from model_loader import ModelLoader
from PIL import Image
import os
from datetime import datetime
import logging

class ImageGenerator:
    def __init__(self, output_dir="outputs", use_lcm=False):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.loader = ModelLoader()
        self.pipeline = self.loader.load_model(use_lcm=use_lcm)
    
    def generate(self, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5, height=512, width=512, num_images_per_prompt=1, seed=None):
        """
        Generates images based on the provided prompt and parameters.
        """
        logging.info(f"Generating image for prompt: '{prompt}'")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(self.loader.device).manual_seed(seed)
        
        try:
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            ).images
            
            saved_paths = []
            for i, image in enumerate(images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{i}.png"
                filepath = os.path.join(self.output_dir, filename)
                image.save(filepath)
                logging.info(f"Image saved to: {filepath}")
                saved_paths.append(filepath)
            
            return saved_paths

        except Exception as e:
            logging.error(f"Error generating image: {e}")
            return []
