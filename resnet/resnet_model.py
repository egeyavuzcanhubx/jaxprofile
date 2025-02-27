import jax
import jax.numpy as jnp
from PIL import Image
import flaxmodels as fm

def get_compute_device():

    devices = jax.devices()
    for d in devices:
        if d.platform == 'tpu':
            return d
    for d in devices:
        if d.platform == 'gpu':
            return d
    return jax.devices("cpu")[0]

class ResNetModel:
    def __init__(self, image_path: str):
        """
        :param image_path: Path of the image to be loaded.
        """
        self.image_path = image_path
        self.key = jax.random.PRNGKey(0)
        self.model = fm.ResNet18(output='logits', pretrained='imagenet')
        self.x = None
        self.params = None

    def load_image(self):
        """
        Loads the image, normalizes it, adds batch dimension, and moves it to the appropriate device.
        """
        # Load the image in RGB format.
        img = Image.open(self.image_path).convert('RGB')
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.expand_dims(x, axis=0)  # (1, H, W, C)

        # Move to the appropriate device (TPU > GPU > CPU).
        device = get_compute_device()
        x = jax.device_put(x, device=device)
        self.x = x
        print("[load_image] Image loaded. Shape:", self.x.shape, "Device:", self.x.device)


    def initialize_model(self):
        """
        Initializes model parameters using a sample input.
        """
        if self.x is None:
            raise ValueError("Image not loaded. Call load_image() first.")
        self.params = self.model.init(self.key, self.x)
        print("[initialize_model] Model parameters initialized.")

    def warmup(self, steps: int = 2):
        """
        Runs a few inference passes for GPU/TPU warmup.
        """
        if self.params is None or self.x is None:
            raise ValueError("Model not initialized or image not loaded.")
        print(f"[warmup] Running {steps} warmup steps...")
        for i in range(steps):
            out = self.model.apply(self.params, self.x, train=False)
            out.block_until_ready()  # Wait for all operations to complete
            print(f"[warmup] {i+1}. warmup step completed.")

    def inference(self):
        """
        Performs model inference.
        """
        if self.params is None or self.x is None:
            raise ValueError("Model not initialized or image not loaded.")
        out = self.model.apply(self.params, self.x, train=False)
        out.block_until_ready()
        return out
