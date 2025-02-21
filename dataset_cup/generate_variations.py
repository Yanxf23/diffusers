import PIL.Image
import requests
import torch
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image


model_path = r"C:\Users\mobil\Desktop\25spring\stylePalm\diffusers\dataset_cup\experiments\exp_prompts\dirty"
input_dir = r"C:\Users\mobil\Desktop\25spring\stylePalm\diffusers\dataset_cup\moresamples"
dst_root = r"C:\Users\mobil\Desktop\25spring\stylePalm\diffusers\dataset_cup\experiments\exp_prompts\dirty\results"

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

# prompts = ["add texture of water", "add texture of dirt", "add subtle branches"]
prompts = ["make it Dirty"]
num_inference_steps = [20]
image_guidance_scale = [0, 1, 1.2, 1.5]
guidance_scale = [0, 5, 10]

image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

for image_path in image_files:
    image = load_image(image_path)
    image_name = os.path.basename(image_path).split(".")[0]  

    for num_step in num_inference_steps:
        for s_i in image_guidance_scale:
            for s_t in guidance_scale:
                for prompt in prompts:
                    output_dir = os.path.join(dst_root, prompt)
                    os.makedirs(output_dir, exist_ok=True)

                    edited_image = pipeline(
                        prompt,
                        image=image,
                        num_inference_steps=num_step,
                        image_guidance_scale=s_i,
                        guidance_scale=s_t,
                        generator=generator,
                    ).images[0]

                    combined_image = PIL.Image.new("RGB", (image.width * 2, image.height))
                    combined_image.paste(image, (0, 0))
                    combined_image.paste(edited_image, (image.width, 0))

                    output_path = os.path.join(output_dir, f"{image_name}_si_{s_i}_st_{s_t}_step_{num_step}.png")
                    combined_image.save(output_path)
                    print(f"Saved: {output_path}")
