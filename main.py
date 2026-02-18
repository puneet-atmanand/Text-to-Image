import argparse
from image_generator import ImageGenerator
import logging

def main():
    parser = argparse.ArgumentParser(description="Text-to-Image Generation using Stable Diffusion")
    parser.add_argument("prompt", type=str, nargs="?", help="The text prompt to generate an image from.")
    parser.add_argument("--negative_prompt", type=str, default="", help="The negative text prompt.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--fast", action="store_true", help="Enable Fast Mode (LCM) for CPU optimization.")

    args = parser.parse_args()

    # Apply defaults for Fast Mode if enabled and not properly overridden (ignoring complex override logic for now, just checking flag)
    if args.fast:
        if args.steps == 50: args.steps = 4
        if args.guidance_scale == 7.5: args.guidance_scale = 1.0

    if not args.prompt:
        print("Welcome to Text-to-Image Generator!")
        args.prompt = input("Enter your text prompt: ")
        
        # Optional interactive mode for other parameters could be added here
        # For now, we'll stick to defaults or flags for simplicity in this hygiene check

    print(f"Initializing Generator...")
    try:
        generator = ImageGenerator(use_lcm=args.fast)
        print(f"Generating image(s) for: '{args.prompt}'")
        
        saved_files = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            num_images_per_prompt=args.num_images,
            seed=args.seed
        )

        if saved_files:
            print(f"Successfully generated {len(saved_files)} images:")
            for path in saved_files:
                print(f" - {path}")
        else:
            print("Failed to generate images. Check logs for details.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
