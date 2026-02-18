import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_installation():
    logging.info("Checking installation...")
    
    try:
        import torch
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logging.error("PyTorch not installed. Please run: pip install -r requirements.txt")
        return False

    try:
        import diffusers
        logging.info(f"Diffusers version: {diffusers.__version__}")
    except ImportError:
        logging.error("Diffusers not installed. Please run: pip install -r requirements.txt")
        return False

    try:
        from model_loader import ModelLoader
        logging.info("ModelLoader import successful.")
    except ImportError as e:
        logging.error(f"Error importing ModelLoader: {e}")
        return False

    logging.info("Installation check passed!")
    return True

if __name__ == "__main__":
    if check_installation():
        logging.info("You are ready to run the project!")
    else:
        logging.error("Installation check failed. Please fix the errors above.")
