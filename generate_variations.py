import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image
import os

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(r"C:\Users\mobil\Desktop\25spring\stylePalm\diffusers\exp_prompts_none\cup_pix2pix", torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

image = load_image(r"C:\Users\mobil\Desktop\24fall\palmvein2024\palmvein\data\data_v3\ROI\OpenMV_ROI\10\001\850\L\Clean\0312.png")
# prompts = ["add texture of water", "add texture of dirt", "add subtle branches"]
prompts = [""]
num_inference_steps = [20]
image_guidance_scale = [0, 1, 1.5]
guidance_scale = [0, 15, 30]
dst_root = r"C:\Users\mobil\Desktop\25spring\stylePalm\diffusers\exp_prompts_none\stablediffusion"

for num_step in num_inference_steps:
    for s_i in image_guidance_scale:
        for s_t in guidance_scale:
            for prompt in prompts:
                if not os.path.exists(rf"{dst_root}\{prompt}"):
                    os.mkdir(rf"{dst_root}\{prompt}")
                edited_image = pipeline(
                    prompt,
                    image=image,
                    num_inference_steps=num_step,
                    image_guidance_scale=s_i,
                    guidance_scale=s_t,
                    generator=generator,
                    ).images[0]
                edited_image.save(rf"{dst_root}\{prompt}\edited_image_256__si_{s_i}_st_{s_t}_step_{num_step}.png")