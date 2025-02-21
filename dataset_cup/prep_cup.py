import shutil
import os
from pathlib import Path
from PIL import Image
import datasets

def get_image_with_smallest_number(directory):
    if not os.path.exists(directory):
        return None, None
    files = os.listdir(directory)
    png_files = [f for f in files if (f.endswith('.png') and "corrected" not in f)]
    png_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    profile_index = png_files[0].split(".")[0]

    return profile_index, os.path.join(directory, png_files[0]) if png_files else None

def prep_cup():
    # conditions = ["warm", "warm", "Warm"]
    conditions = ["warm"]
    users = [f"{i:03}" for i in range(1, 81)]  
    wavelengths = ["850", "890"]
    hands = ["L", "R"]
    i = 0
    src_root = r"C:\Users\mobil\Desktop\24fall\palmvein2024\palmvein\data\data_v3\ROI\See3_ROI\10"
    dst_root = r"C:\Users\mobil\Desktop\25spring\stylePalm\diffusers\dataset_cup"
    for user in users:
        for condition in conditions:
            for wavelength in wavelengths:
                for hand in hands:
                    index, original_src = get_image_with_smallest_number(rf"{src_root}\{user}\{wavelength}\{hand}\Clean")
                    os.makedirs(rf"{dst_root}\original_warm", exist_ok=True)
                    original_dst = rf"{dst_root}\original_warm\image_{i}.jpg"
                    index, edited_src = get_image_with_smallest_number(rf"{src_root}\{user}\{wavelength}\{hand}\{condition}")
                    os.makedirs(rf"{dst_root}\edited_warm", exist_ok=True)
                    edited_dst = rf"{dst_root}\edited_warm\image_{i}.jpg"
                    prompt = f"make it {condition}\n"
                    if original_src is None or edited_src is None:
                        continue
                    i += 1
                    shutil.copy(original_src, original_dst)
                    shutil.copy(edited_src, edited_dst)
                    with open(rf"{dst_root}\prompts_warm.txt", "a") as f:
                        f.write(prompt)

def validation():
    ORIGINAL_IMAGES = Path("original_warm")
    EDITED_IMAGES = Path("edited_warm")
    PROMPTS = Path("prompts_warm.txt")

    # check if directories exists
    if not ORIGINAL_IMAGES.exists():
        raise FileNotFoundError(f"Directory: {ORIGINAL_IMAGES.absolute()} not found")
    if not EDITED_IMAGES.exists():
        raise FileNotFoundError(f"Directory: {EDITED_IMAGES.absolute()} not found")
    if not PROMPTS.exists():
        raise FileNotFoundError(f"File: {PROMPTS.absolute()} not found")

    # check if directory contains images
    ORIGINAL_IMAGES_COUNT = len(list(ORIGINAL_IMAGES.iterdir()))
    EDITED_IMAGES_COUNT = len(list(EDITED_IMAGES.iterdir()))

    if ORIGINAL_IMAGES_COUNT == 0:
        raise FileNotFoundError(f"Directory: {ORIGINAL_IMAGES.absolute()} does not contain any images")
    else:
        print(f"original images: {ORIGINAL_IMAGES_COUNT}")
        
    if EDITED_IMAGES_COUNT == 0:
        raise FileNotFoundError(f"Directory: {ORIGINAL_IMAGES.absolute()} does not contain any images")
    else:
        print(f"edited images: {EDITED_IMAGES_COUNT}")
        
    if not (ORIGINAL_IMAGES_COUNT == EDITED_IMAGES_COUNT):
        raise ValueError("Mismatch in the number of images in original and edited images")
        
    # check if prompts.txt is empty
    with open(PROMPTS, "r") as fp:
        prompts = fp.readlines()

    if len(prompts) == 0:
        raise ValueError(f"File: {PROMPTS.absolute()} does not contain any prompts")
    elif not (len(prompts) == EDITED_IMAGES_COUNT):
        raise ValueError("The number of Images don't match with the number of prompts")
    else:
        print(f"Prompts: {len(prompts)}")

def load_samples(original_images_path: list[Path], edited_images_path: list[Path], prompts_list: list[str]):
    original_images: list[Image.Image] = []
    edited_images: list[Image.Image] = []
    
    for orig_img, edit_img in zip(original_images_path, edited_images_path):     
        # load images
        original_images.append(Image.open(orig_img.absolute()))
        edited_images.append(Image.open(edit_img.absolute()))
        
        # format the dataset
        dataset_json = {
            "before": original_images,
            "after": edited_images,
            "prompt": prompts_list
        }
        
        # build the dataset
        features = datasets.Features({
            "before": datasets.Image(),
            "after": datasets.Image(),
            "prompt": datasets.Value('string')
        })
        
    return datasets.Dataset.from_dict(dataset_json, features)

prep_cup()

ORIGINAL_IMAGES = Path("original_warm")
EDITED_IMAGES = Path("edited_warm")
PROMPTS = Path("prompts_warm.txt")
with open(PROMPTS, "r") as fp:
    prompts = fp.readlines()
ip2p_dataset = load_samples(ORIGINAL_IMAGES.iterdir(), EDITED_IMAGES.iterdir(), prompts)

ip2p_dataset.push_to_hub(
    "eve25yan/cup_openMV_warm",
    split = 'train',
    private = True
)