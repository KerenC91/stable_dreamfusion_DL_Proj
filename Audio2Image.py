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
    # # Text embedding
    # embeddings = model.forward({ib.ModalityType.TEXT: ib.load_and_transform_text(['a photo of a bird, fron_view'], device),}, normalize=False)
    # text_embeddings = embeddings[ib.ModalityType.TEXT]
    # Image embedding 
    embeddings = model.forward({ib.ModalityType.VISION: ib.load_and_transform_vision_data(["image_files/flamingo.jpg"], device),})
    img_embeddings = embeddings[ib.ModalityType.VISION]
    # Combina embedding and generate image
    w = 0.85
    # embeddings = (1 - w) * audio_embeddings + w * img_embeddings
    embeddings = (1 - w) * audio_embeddings
    images = pipe(prompt='a DSLR photo of a flamingo, front view', image_embeds=embeddings.half()).images
    # Save results
    images[0].save("audiotext2img2_(front0).png")