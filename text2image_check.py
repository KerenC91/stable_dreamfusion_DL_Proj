import anything2image.imagebind as ib
import torch
from diffusers import StableUnCLIPImg2ImgPipeline, DDIMScheduler
import matplotlib.pyplot as plt


@torch.no_grad()
def get_text_embedding(prompt, pipe):
    # Tokenize
    inputs = pipe.tokenizer(text_prompt, padding='max_length', max_length=pipe.tokenizer.model_max_length, return_tensors='pt')
    # Encode tokenizer to text embedding
    text_embeddings = pipe.text_encoder(inputs.input_ids.to(device))[0]
    return text_embeddings

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
).to(device)

# Define imagebinds and guidnace parameters
model = ib.imagebind_huge(pretrained=True).eval().to(device)
audio_paths=["audio_files/wave.wav"]
text_prompt = "a DSLR photo of a flamingo, front view"
guidance_scale = 13
num_inference_steps = 30
generator=torch.Generator(device=device).manual_seed(42)
w_audio = 0.15

# Embedding audio 
with torch.no_grad():
    # Audio embedding 
    embeddings = model.forward({ib.ModalityType.AUDIO: ib.load_and_transform_audio_data(audio_paths, device),})
    audio_embeddings = w_audio * embeddings[ib.ModalityType.AUDIO]
    audio_embeddings = pipe._encode_image(image=None,
                                          device=device,
                                          batch_size=1,
                                          num_images_per_prompt=1,
                                          do_classifier_free_guidance=True,
                                          noise_level=0,
                                          generator=generator,
                                          image_embeds=audio_embeddings.half())
# Define diffusion schudler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, torch_dtype=torch.float16)
# Encode the prompt
pos_embed = get_text_embedding(text_prompt, pipe)
neg_embed = get_text_embedding("", pipe)
text_embeddings = torch.cat([neg_embed, pos_embed], dim=0)
# Create our random starting point
latents = torch.randn((1, 4, 64, 64), device=device, generator=generator, dtype=torch.float16)
latents *= pipe.scheduler.init_noise_sigma
# Prepare the scheduler
pipe.scheduler.set_timesteps(num_inference_steps, device=device)

# Loop through the sampling timesteps
for i, t in enumerate(pipe.scheduler.timesteps):
    # Expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2)
    # Expand time
    tt = torch.cat([t.unsqueeze(0)] * 2)
    # Apply any scaling required by the scheduler
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # Predict the noise residual with the UNet
    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings, class_labels=audio_embeddings).sample

    # Perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Compute the previous noisy sample x_t -> x_t-1
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

# Decode the resulting latents into an image
with torch.no_grad():
    image = pipe.decode_latents(latents.detach())

# Save results
image = pipe.numpy_to_pil(image)[0]
image.save("audiotext2img2_(front0).png")