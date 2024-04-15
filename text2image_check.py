"""
This program compare the diffrent between stable diffusion and image binds to combina the techniques
"""
from diffusers import StableDiffusionPipeline
import torch

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", 
torch_dtype=torch.float16).to(device)
pipe.enable_attention_slicing()
# Generate image 
# Text prompt
text_prompt = "a DSLR photo of a dog, front view"
# Generate image
image = pipe.generate_image(text_prompt, device=device)
# Save results
image[0].save("text2img2.png")