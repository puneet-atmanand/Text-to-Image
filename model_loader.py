import torch
from diffusers import StableDiffusionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelLoader:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.device = self._get_device()
        self.pipeline = None

    def _get_device(self):
        """
        Detects the available device for computation.
        Prioritizes CUDA (NVIDIA GPU), then MPS (Apple Silicon), and finally CPU.
        """
        if torch.cuda.is_available():
            logging.info("CUDA (NVIDIA GPU) detected.")
            return "cuda"
        elif torch.backends.mps.is_available():
            logging.info("MPS (Apple Silicon) detected.")
            return "mps"
        else:
            logging.info("No GPU detected. Using CPU.")
            return "cpu"

    def load_model(self, use_lcm=False):
        """
        Loads the Stable Diffusion pipeline.
        Optimizes for memory if running on CPU or low-VRAM GPU.
        """
        logging.info(f"Loading model: {self.model_id}...")
        try:
            # Load pipeline
            # Use float16 for GPU to save memory, float32 for CPU
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=torch_dtype
            )

            if use_lcm:
                logging.info("Loading LCM LoRA adapter for fast CPU generation...")
                self.pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
                self.pipeline.fuse_lora()
                from diffusers import LCMScheduler
                self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)
                logging.info("LCM LoRA adapter loaded and scheduler updated.")

            # Move to device
            self.pipeline = self.pipeline.to(self.device)

            # Optimizations
            if self.device == "cpu":
                logging.info("Enabling CPU offload for memory efficiency.")
                # self.pipeline.enable_model_cpu_offload() # Requires accelerate
                # Standard CPU optimization
                self.pipeline.enable_attention_slicing()
            elif self.device == "cuda":
                logging.info("Enabling attention slicing for GPU memory efficiency.")
                self.pipeline.enable_attention_slicing()
            
            # MPS optimization (limited)
            if self.device == "mps":
                self.pipeline.enable_attention_slicing()

            logging.info("Model loaded successfully.")
            return self.pipeline

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e
