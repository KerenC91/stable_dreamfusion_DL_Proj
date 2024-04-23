import anything2image.imagebind as ib
import torch
from diffusers import StableUnCLIPImg2ImgPipeline


# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
).to(device)
model = ib.imagebind_huge(pretrained=True).eval().to(device)


with torch.no_grad():
    audio_paths=["audio_files/wave.wav"]
    # Audio embedding 
    embeddings = model.forward({ib.ModalityType.AUDIO: ib.load_and_transform_audio_data(audio_paths, device),})
    audio_embeddings = embeddings[ib.ModalityType.AUDIO]
    # Combina embedding and generate image
    w = 0.15
    # embeddings = (1 - w) * audio_embeddings + w * img_embeddings
    embeddings = w * audio_embeddings
    images = pipe(prompt='a DSLR photo of a flamingo, front view', 
                  height=512,
                  width=512,
                  guidance_scale=13,
                  num_inference_steps=50,
                  generator=torch.Generator(device=device).manual_seed(42),
                  image_embeds=embeddings.half()).images
    # Save results
    images[0].save("audiotext2img2_withPipe.png")