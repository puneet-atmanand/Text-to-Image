from diffusers import StableDiffusionPipeline
import logging
import sys

# Configure logging to show up in terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def download_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    logging.info(f"Starting download for model: {model_id}")
    logging.info("This involves downloading approx 4-6 GB of data. Please be patient.")
    
    try:
        # This triggers the specific download/caching of the model files
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, resume_download=True)
        logging.info("Download complete! The model is now cached.")
        return True
    except Exception as e:
        logging.error(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    download_model()
