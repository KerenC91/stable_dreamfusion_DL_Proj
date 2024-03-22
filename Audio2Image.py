import anything2image.imagebind as ib
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
)
pipe = pipe.to(device)

model = ib.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# generate image
with torch.no_grad():
    audio_paths=["audio_files/bird_audio.wav"]
    embeddings = model.forward({
        ib.ModalityType.AUDIO: ib.load_and_transform_audio_data(audio_paths, device),
    })
    #embeddings['audio'].shape = torch.Size([1, 1024])
    #embeddings.keys() = dict_keys(['audio'])
    embeddings = embeddings[ib.ModalityType.AUDIO]
    ##embeddings.shape = torch.Size([1, 1024])
    images = pipe(prompt='a painting', image_embeds=embeddings.half()).images
    images[0].save("audiotext2img.png")#print(images[0].size) = (768, 768)