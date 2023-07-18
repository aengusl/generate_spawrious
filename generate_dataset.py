# # %%
from ast import parse
import os
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import itertools
from datetime import datetime
import argparse
from ml_collections import ConfigDict
from config_gen import get_config
import torch
import random
import numpy as np
from caption_model import caption_from_images

config = get_config()
# global_save_label = config.global_save_label
global_save_label = '/home/aengusl/Desktop/Projects/OOD_workshop/spawrious/data/ood_data/iteration-{}/'
# batch_size = config.batch_size
# minibatch_size = config.minibatch_size
device = config.device

animals_to_generate = config.animals_to_generate
locations_to_generate = config.locations_to_generate
locations_to_avoid = config.locations_to_avoid

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = config.seed
minibatch_size = 4
batch_size = 3
machine_name = 'bigboy'
num_iters = 1
now = datetime.now()
begin_exp_time = now.strftime("%d%b_%H%M%S")

# The model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", use_auth_token=True
)
pipe = pipe.to(device)

def dirty_image_keyword_filter(
    images: list, keywords: list,
) -> bool:
    """
    Filter images by whether the caption contains the keywords for the object and background
    """
    dirty_bool = False
    preds = caption_from_images(images)
    for caption in preds:
        caption_words = caption.strip().split(" ")
        if not set(keywords) & set(caption_words):
            dirty_bool = True
            break
    return dirty_bool

def generate_batch(
    prompt: str,
    save_label: str,
    keywords: list = ['dog'],
    negative_prompt: str = "human, blurry, painting, cartoon, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, two, multiple",
    num_inference_steps: int = 150,
    batch_size: int = 3, # number of possible prompts for a combination * batch size = size of combination
    minibatch_size: int = 4,
    additional_label: str = None,
):
    """
    Generate a batch of images from a prompt.
    Save the images to a folder specified by the save label, under the format
        <save_label>/<machine_name>_<idx>.png
    """
    print('Generating images with prompt: \n', prompt)
    os.makedirs(save_label, exist_ok=True)
    prompt_list = [prompt] * minibatch_size
    negative_prompt_list = [negative_prompt] * minibatch_size
    batch_count = 0
    cleaning_total = 0
    with tqdm(total=batch_size) as pbar:
        while batch_count < batch_size:
            # Generate a batch of images
            output = pipe(
                prompt_list,
                negative_prompt=negative_prompt_list,
                num_inference_steps=num_inference_steps,
            )
            images = output.images

            # Filter for bad images
            # dirty_bool = dirty_image_keyword_filter(images=images, keywords=keywords) # True if images are dirty
            dirty_bool = False
            nsfw_bool = sum(output.nsfw_content_detected) > 0 #True if any images are nsfw
            if dirty_bool or nsfw_bool:
                print('Bad images detected: \n', 'dirty_bool:', dirty_bool, ', nsfw_bool:', nsfw_bool)
                if dirty_bool:
                    cleaning_total += 1
                continue
            
            # Save the images
            for idx, image in enumerate(images):
                save_path = os.path.join(save_label, f"{machine_name}_{batch_count+idx}.png")
                if additional_label is not None:
                    save_path = os.path.join(save_label, f"{machine_name}_{additional_label}_{batch_count+idx}.png")
                image.save(save_path, format="png")
                pbar.update(1)
            batch_count += len(images)
    return cleaning_total
# Let's go
# %%
"""
tester
"""
tester = False
if tester == True:
    generate_batch(
            prompt=f"((( one black dachshund sitting ))) on a rocky mountain, sunset. Highly detailed, with cinematic lighting",
            save_label=f"./3Mar/",
            batch_size=8,
            minibatch_size=1,
            num_inference_steps=10,
        )

else:
    """
    Create prompt list dictionary of form:
        {'animal-background': [prompt1,..,prompD]}
    """

    animal_list = [
        "labrador",
        "welsh corgi dog",
        "bulldog",
        "dachshund",
    ]
    one_word_animal_list = [
        "labrador",
        "corgi",
        "bulldog",
        "dachshund",
    ]
    animal_dict = {
        "labrador": "labrador",
        "corgi": "welsh corgi dog",
        "bulldog": "bulldog",
        "dachshund": "dachshund"
    }
    location_list = [
        'in a jungle',
        'on a rocky mountain',
        'in a hot, dry desert with cactuses around',
        # 'in a dark alley with trash and graffitis',
        'in a park, with puddles, bushes and dirt in the background',
        'playing fetch on a beach with a pier and ocean in the background',
        'in a snowy landscape with a cabin and a snowball in the background',
    ]
    one_word_location_list = [
        'jungle',
        'mountain',
        'desert',
        'dirt',
        'beach',
        'snow',
        # 'alley',
    ]
    location_dict = {
        'jungle': 'in a jungle',
        'mountain': 'on a rocky mountain',
        'desert': 'in a hot, dry desert with cactuses around',
        # 'alley': 'in a dark alley with trash and graffitis',
        'dirt': 'in a park, with puddles, bushes and dirt in the background',
        'beach': 'playing fetch on a beach with a pier and ocean in the background',
        'snow': 'in a snowy landscape with a cabin and a snowball in the background',
    }
    fur_list = [
        "black",
        "brown",
        "white",
        "",
    ]
    pose_list = [
        "sitting",
        "",
        "running",
    ]
    tod_list = [
        "pale sunrise",
        "sunset",
        "rainy day",
        "foggy day",
        "bright sunny day",
        "bright sunny day",
    ]
    prompt_template = "(((one {fur} {animal} {pose}))) {location}, {tod}. highly detailed, with cinematic lighting, 4k resolution, beautiful composition, hyperrealistic, trending, cinematic, masterpiece, close up"    

    assert animals_to_generate in one_word_animal_list or animals_to_generate == 'all'
    if animals_to_generate == 'all':
        pass
    else:
        one_word_animal_list = [animals_to_generate]

    assert locations_to_generate in one_word_location_list or locations_to_generate == 'all'
    if locations_to_generate == 'all':
        pass
    else:
        one_word_location_list = [locations_to_generate]
    
    if locations_to_avoid != 'None':
        for loc in locations_to_avoid:
            one_word_location_list.remove(loc)

    prompt_list_dict = {}
    for animal_word in one_word_animal_list:
        for location_word in one_word_location_list:
            animal = animal_dict[animal_word]
            location = location_dict[location_word]
            prompt_list_dict[f'{animal_word}-{location_word}'] = []
            for fur in fur_list:
                for pose in pose_list:
                    for tod in tod_list:
                        prompt = prompt_template.format(fur=fur, animal=animal, pose=pose, location=location, tod=tod)
                        prompt_list_dict[f'{animal_word}-{location_word}'].append(prompt)

    # %%
    """
    Generate a mini dataset with samples from each prompt
    """
    for iteration in tqdm(range(num_iters)):
        print('\n\n\n\n\n\nIteration:', iteration, '\n\n\n\n\n\n')
        cleaning_total = 0
        for animal_loc in tqdm(prompt_list_dict.keys()):
            print('\n\n\nAnimal-Location:', animal_loc, '\n\n\n')
            prompt_count = 0
            for prompt in prompt_list_dict[animal_loc]:
                # print(prompt)
                # if prompt_count > 100:
                #     break
                animal_str = animal_loc.split('-')[0]
                location_str = animal_loc.split('-')[1]
                save_label = os.path.join(global_save_label.format(iteration), f"{location_str}", f"{animal_str}")
                os.makedirs(save_label, exist_ok=True)
                cleaning_total += generate_batch(
                    prompt=prompt,
                    save_label=save_label,
                    keywords = ['dog'],
                    # batch_size=batch_size,
                    minibatch_size=minibatch_size,
                    num_inference_steps=100,
                    additional_label = f"prompt_{prompt_count}"
                )
                prompt_count += 1

# %%

