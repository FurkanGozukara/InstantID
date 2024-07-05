import sys

from diffusers.models.unet_2d_condition import UNet2DConditionModel

sys.path.append("./")
from tqdm import tqdm
from typing import Tuple
from PIL import PngImagePlugin
import time
import json
from datetime import datetime
import os
import cv2
import math
import torch
import random
import numpy as np
import argparse

import traceback

from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis
from style_template import styles
from pipelines.pipeline_common import quantize_4bit
from pipelines.pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl, get_torch_device
from controlnet_util import load_controlnet, load_depth_estimator as load_depth, get_depth_map, get_depth_anything_map, get_canny_image
from common.util import clean_memory

import gradio as gr
from downloader import ensure_libraries, download_file

CONFIG_DIR = "configs"
LATEST_CONFIG_FILE = "latest_config.txt"

def save_config(config_name, *args):
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    
    config = {
        "timestamp": datetime.now().isoformat(),
        "prompt": args[0],
        "negative_prompt": args[1],
        "style": args[2],
        "num_steps": args[3],
        "identitynet_strength_ratio": args[4],
        "adapter_strength_ratio": args[5],
        "pose_strength": args[6],
        "canny_strength": args[7],
        "depth_strength": args[8],
        "controlnet_selection": args[9],
        "guidance_scale": args[10],
        "seed": args[11],
        "randomize_seed": args[12],
        "scheduler": args[13],
        "enable_LCM": args[14],
        "enhance_face_region": args[15],
        "model_input": args[16],
        "model_dropdown": args[17],
        "width": args[18],
        "height": args[19],
        "num_images": args[20],
        "guidance_threshold": args[21],
        "depth_type": args[22],
        "lora_model_dropdown": args[23],
        "lora_scale": args[24]
    }
    
    filename = f"{config_name}.json"
    with open(os.path.join(CONFIG_DIR, filename), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save the latest used config name
    with open(os.path.join(CONFIG_DIR, LATEST_CONFIG_FILE), 'w') as f:
        f.write(config_name)
    
    config_list = get_config_list()
    return config_list, config_name

def update_config_dropdown(config_list, selected_config):
        return gr.update(choices=config_list, value=selected_config)

def load_config(config_name):
    if not config_name:
        with open(os.path.join(CONFIG_DIR, LATEST_CONFIG_FILE), 'w') as f:
            f.write("")
        return [gr.update()] * 25  # Return no updates if no config is selected
    
    filename = f"{config_name}.json"
    filepath = os.path.join(CONFIG_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"Config file {filename} not found.")
        return [gr.update()] * 25
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    # Save the latest used config name
    with open(os.path.join(CONFIG_DIR, LATEST_CONFIG_FILE), 'w') as f:
        f.write(config_name)
    
    return [
        gr.update(value=config["prompt"]),
        gr.update(value=config["negative_prompt"]),
        gr.update(value=config["style"]),
        gr.update(value=config["num_steps"]),
        gr.update(value=config["identitynet_strength_ratio"]),
        gr.update(value=config["adapter_strength_ratio"]),
        gr.update(value=config["pose_strength"]),
        gr.update(value=config["canny_strength"]),
        gr.update(value=config["depth_strength"]),
        gr.update(value=config["controlnet_selection"]),
        gr.update(value=config["guidance_scale"]),
        gr.update(value=config["seed"]),
        gr.update(value=config["randomize_seed"]),
        gr.update(value=config["scheduler"]),
        gr.update(value=config["enable_LCM"]),
        gr.update(value=config["enhance_face_region"]),
        gr.update(value=config["model_input"]),
        gr.update(value=config["model_dropdown"]),
        gr.update(value=config["width"]),
        gr.update(value=config["height"]),
        gr.update(value=config["num_images"]),
        gr.update(value=config["guidance_threshold"]),
        gr.update(value=config["depth_type"]),
        gr.update(value=config["lora_model_dropdown"]),
        gr.update(value=config["lora_scale"])
    ]

def get_config_list():
    try:
        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR)
            print(f"Created config directory: {CONFIG_DIR}")
        config_list = [f.split('.')[0] for f in os.listdir(CONFIG_DIR) if f.endswith('.json') and f.strip()]
        print(f"Config list: {config_list}")
        return config_list
    except Exception as e:
        print(f"Error in get_config_list: {str(e)}")
        print(traceback.format_exc())
        return []

def refresh_config_list():
    config_list = get_config_list()
    latest_config = get_latest_config()
    selected_config = latest_config if latest_config in config_list else None
    print(f"Refreshed config list: {config_list}, Selected config: {selected_config}")
    return config_list


def get_latest_config():
    try:
        latest_config_path = os.path.join(CONFIG_DIR, LATEST_CONFIG_FILE)
        if not os.path.exists(latest_config_path):
            print(f"Latest config file not found: {latest_config_path}")
            return None
        with open(latest_config_path, 'r') as f:
            latest_config = f.read().strip()
        print(f"Latest config: {latest_config}")
        return latest_config
    except Exception as e:
        print(f"Error in get_latest_config: {str(e)}")
        print(traceback.format_exc())
        return None



# Define the pre-defined models
PREDEFINED_MODELS = {
    "RealVisXL v4.0": "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors",
    "SDXL Base 1.0": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors",
    "Juggernaut-X v10": "https://huggingface.co/RunDiffusion/Juggernaut-X-v10/resolve/main/Juggernaut-X-RunDiffusion-NSFW.safetensors",
    "Juggernaut-XL v9": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    "Animagine XL v3.1":"https://civitai.com/api/download/models/403131?type=Model&format=SafeTensor&size=full&fp=fp16",
    "DynaVision XL v0.6":"https://civitai.com/api/download/models/297740",
    "EpiCRealism XL v7": "https://civitai.com/api/download/models/489217",
    "RealCartoon-XL v6":"https://civitai.com/api/download/models/254091?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "AAM XL Anime Mix v1":"https://civitai.com/api/download/models/303526",
    "Anima_pencil-XL v5" : "https://civitai.com/api/download/models/597138",
    "HimawariMix XL v13": "https://civitai.com/api/download/models/558064",
    "Jib Mix Realistic XL v13": "https://civitai.com/api/download/models/610292",
    "SDXL Yamers Anime Stage Anima":"https://civitai.com/api/download/models/377674",
    "Deep Blue XL v4.0.1":"https://civitai.com/api/download/models/420370",
    "SDXL FaeTastic v24": "https://civitai.com/api/download/models/291443",
    "PixelWave v10":"https://civitai.com/api/download/models/542574",
    "Pixel Art Diffusion XL Sprite Shaper":"https://civitai.com/api/download/models/364043",
    "Halcyon SDXL v1.7":"https://civitai.com/api/download/models/610541",
    "WildCardX-XL-Fusion OG":"https://civitai.com/api/download/models/345685",
    "Raemu XL v4" : "https://civitai.com/api/download/models/613928"




}

def update_ip_adapter_scale(adapter_strength_ratio):
    global pipe
    if pipe is not None:
        pipe.set_ip_adapter_scale(adapter_strength_ratio)
        print(f"Updated IP adapter scale to {adapter_strength_ratio}")


def download_all_predefined_models():
    status_messages = []
    for model_name in PREDEFINED_MODELS.keys():
        status = download_predefined_model(model_name)
        status_messages.append(f"{model_name}: {status}")
    return "\n".join(status_messages)


def get_model_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(('.safetensors', '.ckpt'))]

def download_model(url, name, model_type):
    ensure_libraries()
    
    # Remove '?download=true' from the end of the URL if present
    url = url.split('?download=true')[0]
    
    # Append the token if the URL contains 'civitai'
    if 'civitai' in url.lower():
        token = "token=5577db242d28f46030f55164cdd2da5d"
        if '?' in url:
            url += f"&{token}"
        else:
            url += f"?{token}"
    
    # Determine the download path based on model type
    download_path = used_model_path if model_type == "Checkpoint" else used_lora_path
    
    # If name is not provided, use the last part of the URL as the filename
    if not name:
        name = url.split('/')[-1].split('?')[0]  # Remove query parameters from filename
    
    # Ensure the filename has the correct extension
    if not (name.endswith('.safetensors') or name.endswith('.ckpt')):
        name += '.safetensors'
    
    dest = os.path.join(download_path, name)
    
    try:
        download_file(url, dest)
        return f"Download completed: {name}"
    except Exception as e:
        return f"Download failed: {str(e)}"

def download_predefined_model(model_name):
    if model_name in PREDEFINED_MODELS:
        url = PREDEFINED_MODELS[model_name]
        return download_model(url, model_name + ".safetensors", "Checkpoint")
    else:
        return "Selected model not found in predefined list."

parser = argparse.ArgumentParser()

parser.add_argument(
"--pretrained_model_folder", type=str, default=None
)
parser.add_argument(
"--models_path", type=str, default=None
)
parser.add_argument(
"--enable_LCM", type=bool, default=os.environ.get("ENABLE_LCM", False)
)
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--fp16", action="store_true", help="fp16")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for optimization memory consumption")
parser.add_argument("--share", action="store_true", help="Enable Gradio app sharing")
parser.add_argument(
"--loras_path", type=str, default=None
)

args = parser.parse_args()

load_mode = args.load_mode

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

dtype = dtype if str(device).__contains__("cuda") else torch.float32

dtypeQuantize = dtype

if(load_mode in ('4bit','8bit')):
    dtypeQuantize = torch.float8_e4m3fn

ENABLE_CPU_OFFLOAD = True if args.lowvram else False
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"
DEPTH_ESTIMATOR = ["LiheYoung/depth_anything", "Intel/dpt-hybrid-midas"]
_pretrained_model_folder = None
default_model = "wangqixun/YamerMIX_v8"
last_loaded_model = None
last_loaded_scheduler = None
last_loaded_depth_estimator = None
last_LCM_status = None

pipe = None
controlnet = None

# Load face encoder
app = FaceAnalysis(
    name="antelopev2",
    root="./",
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = f"checkpoints/ip-adapter.bin"
controlnet = None
controlnet_map = {}

def set_metadata_settings(image_path):
    if image_path is None:
        return gr.update()
    
    # Open the image and extract metadata
    with Image.open(image_path) as img:
        metadata = img.info

        # Extract and set the relevant metadata settings
        prompt = metadata.get("Prompt", "")
        face_file = metadata.get("Upload a photo of your face full path", "")
        pose_file = metadata.get("Upload a reference pose image (Optional) full path", "")
        negative_prompt = metadata.get("Negative Prompt", "")
        enable_LCM = metadata.get("Enable Fast Inference with LCM", "False") == "True"
        depth_type = metadata.get("Depth Estimator", "LiheYoung/depth_anything")
        identitynet_strength_ratio = float(metadata.get("IdentityNet strength (for fidelity)", "0.80"))
        adapter_strength_ratio = float(metadata.get("Image Adapter Strength", "0.80"))
        pose_strength = float(metadata.get("Pose strength", "0.40"))
        canny_strength = float(metadata.get("Canny strength", "0.40"))
        depth_strength = float(metadata.get("Depth strength", "0.40"))
        controlnet_selection = metadata.get("used Controlnets", "").split(", ") if len(metadata.get("used Controlnets", "")) > 1 else []
        model_dropdown = metadata.get("Dropdown Selected Model", None)
        model_input = metadata.get("Full Model Path - Used If Set", "")
        lora_model_dropdown = metadata.get("Select LoRA models", "").split(", ") if len(metadata.get("Select LoRA models", "")) > 1 else []
        width_target = int(metadata.get("Target Image Width", "1280"))
        height_target = int(metadata.get("Target Image Height", "1280"))
        style_name = metadata.get("Style Template", DEFAULT_STYLE_NAME)
        num_steps = int(metadata.get("Number Of Sample Steps", "5"))
        guidance_scale = float(metadata.get("CFG Scale", "5.0"))
        guidance_threshold = float(metadata.get("CFG Threshold", "1.0"))
        seed = int(metadata.get("Used Seed", "42"))
        enhance_face_region = metadata.get("Enhance non-face region", "True") == "True"
        scheduler = metadata.get("Used Scheduler", "EulerDiscreteScheduler")
        

    updates = [gr.update(value=prompt), gr.update(value=negative_prompt), gr.update(value=enable_LCM), gr.update(value=depth_type), gr.update(value=identitynet_strength_ratio), gr.update(value=adapter_strength_ratio), gr.update(value=pose_strength), gr.update(value=canny_strength), gr.update(value=depth_strength), gr.update(value=controlnet_selection), gr.update(value=model_dropdown), gr.update(value=model_input), gr.update(value=lora_model_dropdown), gr.update(value=width_target), gr.update(value=height_target), gr.update(value=style_name), gr.update(value=num_steps), gr.update(value=guidance_scale), gr.update(value=guidance_threshold), gr.update(value=seed), gr.update(value=enhance_face_region), gr.update(value=scheduler)]

    # Update the source images only if the file paths are non-empty
    if len(face_file) > 5:
        updates.append(gr.update(value=face_file))
    else:
        updates.append(gr.update())  # Do not update if the path is empty

    if len(pose_file) > 5:
        updates.append(gr.update(value=pose_file))
    else:
        updates.append(gr.update())  # Do not update if the path is empty
    lora_scale = float(metadata.get("LoRA Scale", "1.0"))
    updates.append(gr.update(value=lora_scale))
    return tuple(updates)


def read_image_metadata(image_path):
    if image_path is None:
        return
    # Check if the file exists
    if not os.path.exists(image_path):
        return "File does not exist."

    # Get the last modified date and format it
    last_modified_timestamp = os.path.getmtime(image_path)
    last_modified_date = datetime.fromtimestamp(last_modified_timestamp).strftime('%d %B %Y, %H:%M %p - UTC')

    # Open the image and extract metadata
    with Image.open(image_path) as img:
        width, height = img.size
        megapixels = (width * height) / 1e6

        metadata_str = f"Last Modified Date: {last_modified_date}\nMegapixels: {megapixels:.2f}\n"

        # Extract metadata based on image format
        if img.format == 'JPEG':
            exif_data = img._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = Image.ExifTags.TAGS.get(tag, tag)
                    metadata_str += f"{tag_name}: {value}\n"
        else:
            metadata = img.info
            if metadata:
                for key, value in metadata.items():
                    metadata_str += f"{key}: {value}\n"
            else:
                metadata_str += "No additional metadata found."

    return metadata_str

def load_controlnet_open_pose(pretrained_model_folder):
    global controlnet, controlnet_map, controlnet_map_fn

    openpose, controlnet_pose, controlnet_canny, controlnet_depth, controlnet = load_controlnet(pretrained_model_folder, dtype)      

    controlnet_map = {
    "pose": controlnet_pose,
    "canny": controlnet_canny,
    "depth": controlnet_depth,
    }
    controlnet_map_fn = {
    "pose": openpose,
    "canny": get_canny_image,
    "depth": get_depth_map,
}
def load_depth_estimator(pretrained_model_folder,depth_type):
    load_depth(pretrained_model_folder, device,depth_type)
    if(depth_type == "LiheYoung/depth_anything"):
        controlnet_map_fn["depth"]=get_depth_anything_map
    else:
        controlnet_map_fn["depth"]=get_depth_map


import platform
used_model_path='models'
used_lora_path='loras'

def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

def get_lora_model_names():
    global used_lora_path
    if args.loras_path:
        if os.path.exists(args.loras_path):
            used_lora_path = args.loras_path
    if not os.path.exists(used_lora_path):
        os.makedirs(used_lora_path)
    lora_files = sorted([f for f in os.listdir(used_lora_path) if f.endswith('.safetensors')])
    return lora_files

def get_model_names():
    global used_model_path
    if args.models_path:
        if os.path.exists(args.models_path):
            used_model_path = args.models_path
    if not os.path.exists(used_model_path):
        os.makedirs(used_model_path)
    model_files = [default_model]
    model_files += sorted([f for f in os.listdir(used_model_path) if f.endswith('.safetensors')])
    model_files = sorted(model_files)
    return model_files

def load_model(pretrained_model_folder, model_name):
    global pipe    
    print(f"Loading model: {model_name}")
    # Properly discard the old pipe if it exists
    
    if model_name.endswith(
        ".ckpt"
    ) or model_name.endswith(".safetensors"):
        model_path = model_name      
        scheduler_kwargs = hf_hub_download(
            repo_id="wangqixun/YamerMIX_v8",
            subfolder="scheduler",
            filename="scheduler_config.json",            
        ) if not pretrained_model_folder else hf_hub_download(
            repo_id="wangqixun/YamerMIX_v8",
            subfolder="scheduler",
            filename="scheduler_config.json",            
            local_dir=fr"{pretrained_model_folder}/wangqixun/YamerMIX_v8",
            local_dir_use_symlinks=False
        )

        (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
        pretrained_model_name_or_path=model_path,
        scheduler_name=None,
        weight_dtype=dtype,
        weight_quantize_dtype=dtypeQuantize
        )
        
        scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)
        print(f"vae dtype {vae.dtype}")
        pipe = StableDiffusionXLInstantIDPipeline(            
            vae=vae,
            text_encoder=text_encoders[0],
            text_encoder_2=text_encoders[1],
            tokenizer=tokenizers[0],
            tokenizer_2=tokenizers[1],
            unet=unet,
            scheduler=scheduler,
            controlnet=[controlnet],           
        )
    else:    
        if pretrained_model_folder:
            model_path = fr"{pretrained_model_folder}/{model_name}"
        else:
            model_path = model_name
        
        unet = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder="unet",
            torch_dtype=dtypeQuantize,
        )
        # Load vae
        vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        print(f"vae dtype {vae.dtype}")
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(            
            model_path,
            vae = vae,
            unet = unet,
            controlnet=[controlnet],
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        )
        
    if load_mode == '4bit':
        quantize_4bit(pipe.unet)
        #quantize_4bit(pipe.controlnet)
        if pipe.text_encoder is not None:
            quantize_4bit(pipe.text_encoder)
        if pipe.text_encoder_2 is not None:
            quantize_4bit(pipe.text_encoder_2)     

    return pipe
	
def refresh_model_names():
    model_names = get_model_names()
    lora_model_names = get_lora_model_names()
    return gr.update(choices=model_names), gr.update(choices=lora_model_names)

def assign_last_params(adapter_strength_ratio, with_cpu_offload):
    global pipe
    
    set_ip_adapter(adapter_strength_ratio)
    
    # apply improvements    
    if with_cpu_offload:                 
        pipe.enable_model_cpu_offload()        
    else:
        pipe.to(device)

    clean_memory()  
    
    #if load_mode != '4bit' :
    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling() 

def set_ip_adapter(adapter_strength_ratio):    
    pipe.load_ip_adapter_instantid(face_adapter,scale=adapter_strength_ratio)   
    #if pipe.image_proj_model != None and  load_mode == '4bit':
    #    quantize_4bit(pipe.image_proj_model)

    pipe.set_ip_adapter_scale(adapter_strength_ratio)
 
def load_scheduler(pretrained_model_folder, scheduler, with_LCM):
     
    # load and disable LCM
    if with_LCM:
        lora_model = "latent-consistency/lcm-lora-sdxl"
        pipe.load_lora_weights(lora_model)  
        pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_lora()
    else:
        pipe.disable_lora()     
        scheduler_class_name = scheduler.split("-")[0]
        add_kwargs = {}
        if len(scheduler.split("-")) > 1:
            add_kwargs["use_karras_sigmas"] = True
        if len(scheduler.split("-")) > 2:
            add_kwargs["algorithm_type"] = "sde-dpmsolver++"
        scheduler = getattr(diffusers, scheduler_class_name)
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config, **add_kwargs)  
    
def main(pretrained_model_folder, enable_lcm_arg=False, share=False):

    global _pretrained_model_folder
    global used_model_path
    global used_lora_path
    _pretrained_model_folder = pretrained_model_folder
   
    def reload_pipe(model_input, model_dropdown, scheduler, adapter_strength_ratio, with_LCM, depth_type, lora_model_dropdown, lora_scale_variable):
        global pipe  # Declare pipe as a global variable thas_cpu_offload manage it when the model changes
        global last_loaded_model, last_loaded_scheduler, last_loaded_depth_estimator, last_LCM_status
 
        # Trim the model_input to remove any leading or trailing whitespace
        model_input = model_input.strip() if model_input else None

        # Determine the model to load
        model_to_load = model_input if model_input else os.path.join(used_model_path, model_dropdown) if (model_dropdown and model_dropdown != default_model) else default_model if model_dropdown == default_model else None

        # Return early if no model is selected or inputted
        if not model_to_load:
            print("No model selected or inputted. Default model will be used.")
            model_to_load = default_model
   
        # Reload CPU offload to fix bug for half mode
        if pipe and ENABLE_CPU_OFFLOAD:
            restart_cpu_offload(adapter_strength_ratio)

        if not pipe:
            pipe = None            
            #load controlnet
            load_controlnet_open_pose(pretrained_model_folder)
            load_depth_estimator(pretrained_model_folder, depth_type)
            clean_memory()

            pipe = load_model(_pretrained_model_folder, model_to_load)
            last_loaded_model = model_to_load
            last_loaded_scheduler = scheduler
            last_LCM_status = with_LCM
            last_loaded_depth_estimator = depth_type
            load_scheduler(pretrained_model_folder, scheduler, with_LCM)
            assign_last_params(adapter_strength_ratio, ENABLE_CPU_OFFLOAD)

        # Reload depth estimator if needed
        if (pipe and model_to_load == last_loaded_model
            and depth_type != last_loaded_depth_estimator):          
            load_depth_estimator(pretrained_model_folder, depth_type)  

        # Reload scheduler if needed
        if (pipe and model_to_load == last_loaded_model 
            and scheduler != last_loaded_scheduler):
            last_LCM_status = with_LCM
            load_scheduler(pretrained_model_folder, scheduler, with_LCM)
            last_loaded_scheduler = scheduler

        if (pipe and model_to_load == last_loaded_model 
            and last_LCM_status != with_LCM):
            last_LCM_status = with_LCM
            load_scheduler(pretrained_model_folder, scheduler, with_LCM)
            last_loaded_scheduler = scheduler
   
        # Reload model if needed
        if (pipe and model_to_load != last_loaded_model):                                              
            # Reload model        
            pipe = load_model(_pretrained_model_folder, model_to_load)
            last_loaded_model = model_to_load
            last_loaded_scheduler = scheduler
            last_LCM_status = with_LCM
            last_loaded_depth_estimator = depth_type
            load_scheduler(pretrained_model_folder, scheduler, with_LCM)
            assign_last_params(adapter_strength_ratio, ENABLE_CPU_OFFLOAD)
    
        pipe.unload_lora_weights()
        if lora_model_dropdown:  # Check if any LoRA models are selected
            for lora_model in lora_model_dropdown:
                lora_path = os.path.join(used_lora_path, lora_model)
                print(f"Loading LoRA: {lora_path}")
                pipe.load_lora_weights(lora_path)
        
            pipe.fuse_lora(lora_scale=lora_scale_variable)
            print("LoRA models loaded and fused successfully.")
        else:
            print("No LoRA models selected. Skipping LoRA fusion.")

    print("Model loaded successfully.")

    def restart_cpu_offload(adapter_strength_ratio):
        
        #if load_mode != '4bit' :
        pipe.disable_xformers_memory_efficient_attention()

        set_ip_adapter(adapter_strength_ratio)            
        from pipelines.pipeline_common import optionally_disable_offloading
        optionally_disable_offloading(pipe)
        clean_memory()
        pipe.enable_model_cpu_offload()
        #if load_mode != '4bit' :
        pipe.enable_xformers_memory_efficient_attention()

    def toggle_lcm_ui(value):
        if value:
            return (
                gr.update(minimum=0, maximum=100, step=1, value=5),
                gr.update(minimum=0.1, maximum=20.0, step=0.1, value=1.5),
            )
        else:
            return (
                gr.update(minimum=5, maximum=100, step=1, value=30),
                gr.update(minimum=0.1, maximum=20.0, step=0.1, value=5),
            )

    def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        return seed

    def remove_tips():
        return gr.update(visible=False)

    def convert_from_cv2_to_image(img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def convert_from_image_to_cv2(img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_kps(
        image_pil,
        kps,
        color_list=[
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
        ],
    ):
        stickwidth = 4
        limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
        kps = np.array(kps)

        w, h = image_pil.size
        out_img = np.zeros([h, w, 3])

        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[index[0]]

            x = kps[index][:, 0]
            y = kps[index][:, 1]
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly(
                (int(np.mean(x)), int(np.mean(y))),
                (int(length / 2), stickwidth),
                int(angle),
                0,
                360,
                1,
            )
            out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        out_img = (out_img * 0.6).astype(np.uint8)

        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

        out_img_pil = Image.fromarray(out_img.astype(np.uint8))
        return out_img_pil


    def resize_img(input_image, size=None, max_side=1280, min_side=1024,
                   pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
        w, h = input_image.size

        # Create the temp_faces folder if it does not exist
        if not os.path.exists('temp_faces'):
            os.makedirs('temp_faces')

        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

        target_aspect_ratio = size[0] / size[1] if size else max_side / min_side

        # Detect faces in the image using FaceAnalysis
        face_image_cv2 = convert_from_image_to_cv2(input_image)
        faces = app.get(face_image_cv2)
        if faces:
            face = faces[0]  # Assuming the first face is the primary face
            x, y, x2, y2 = face['bbox']
            face_left = x
            face_top = y
            face_right = x2
            face_bottom = y2

            # Expand the bounding box to include some margin
            margin = 0.2 * max(face_right - face_left, face_bottom - face_top)
            face_left = max(0, face_left - margin)
            face_top = max(0, face_top - margin)
            face_right = min(w, face_right + margin)
            face_bottom = min(h, face_bottom + margin)

            # Save the found face
            found_face = input_image.crop((face_left, face_top, face_right, face_bottom))
            found_face.save(f'temp_faces/found_face_{timestamp}.png', 'PNG')

            # Determine which sides to crop to keep the face within the cropped area
            if w / h > target_aspect_ratio:  # Crop horizontally
                new_width = h * target_aspect_ratio
                if face_left + new_width > w:  # Face is closer to the right edge
                    left = w - new_width
                else:  # Face is closer to the left edge or in the middle
                    left = face_left
                right = left + new_width
                top = 0
                bottom = h
            else:  # Crop vertically
                new_height = w / target_aspect_ratio
                if face_top + new_height > h:  # Face is closer to the bottom edge
                    top = h - new_height
                else:  # Face is closer to the top edge or in the middle
                    top = face_top
                bottom = top + new_height
                left = 0
                right = w

            # Crop the image to the calculated area
            input_image = input_image.crop((left, top, right, bottom))
            w, h = input_image.size  # Update dimensions after cropping

            # Resize the image to the specified size, if provided
            if size:
                input_image = input_image.resize(size, mode)
        else:
            # No faces detected, fall back to original resizing logic
            input_image = input_image.resize(size, mode) if size else input_image

        # Save the final cropped image
        input_image.save(f'temp_faces/final_cropped_input_face_{timestamp}.png', 'PNG')

        if pad_to_max_side:
            # Create a new image with a white background
            max_dimension = max(*size) if size else max_side
            res = np.ones([max_dimension, max_dimension, 3], dtype=np.uint8) * 255
            w, h = input_image.size
            offset_x = (max_dimension - w) // 2
            offset_y = (max_dimension - h) // 2
            res[offset_y:offset_y + h, offset_x:offset_x + w] = np.array(input_image)
            input_image = Image.fromarray(res)

        return input_image

    def convert_from_image_to_cv2(img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def apply_style(
        style_name: str, positive: str, negative: str = ""
    ) -> Tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + " " + negative

    def generate_image(
        generation_type,
        face_image_path,
        pose_image_path,
        prompt,
        negative_prompt,
        style_name,
        num_steps,
        identitynet_strength_ratio,
        adapter_strength_ratio,
        pose_strength,
        canny_strength,
        depth_strength,
        controlnet_selection,
        guidance_scale,
        seed,
        randomize_seed,
        scheduler,
        enable_LCM,        
        enhance_face_region,
        model_input,
        model_dropdown,
        width_target,
        height_target,
        num_images,
        guidance_threshold,
        depth_type,
        lora_model_dropdown,
        lora_scale,
        head_only_control,
        progress=gr.Progress(),
    ):
        global controlnet_map, controlnet_map_fn
   
        if face_image_path is None:
            raise gr.Error(f"Cannot find any input face image! Please upload the face image")

        if prompt is None:
            prompt = "a person"

        org_prompt = prompt
        org_negative = negative_prompt
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        face_image = load_image(face_image_path)
        face_image = resize_img(face_image,size=(width_target, height_target))
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        face_info = app.get(face_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(f"Unable to detect a face in the image. Please upload a different photo with a clear face.")

        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
        img_controlnet = face_image
        if pose_image_path is not None:
            pose_image = load_image(pose_image_path)
            pose_image = resize_img(pose_image,size=(width_target, height_target))
            img_controlnet = pose_image
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)

            face_info = app.get(pose_image_cv2)

            if len(face_info) == 0:
                raise gr.Error(f"Cannot find any face in the reference image! Please upload another person image")

            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info["kps"])

            width, height = face_kps.size

        if enhance_face_region:
            control_mask = np.zeros([height_target, width_target, 3])
            x1, y1, x2, y2 = face_info["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None

        reload_pipe(model_input, model_dropdown, scheduler, adapter_strength_ratio, enable_LCM, depth_type, lora_model_dropdown, lora_scale)      
        set_ip_adapter(adapter_strength_ratio)
        control_scales, control_images = set_pipe_controlnet(identitynet_strength_ratio, pose_strength, canny_strength, depth_strength, controlnet_selection, width_target, height_target, face_kps, img_controlnet)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        images_generated = []
        start_time = time.time()

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")
    
        with torch.no_grad():        
            with torch.cuda.amp.autocast(dtype=dtype):
                for i in range(num_images):
                    progress(i / num_images, desc=f"Generating image {i+1}/{num_images}")
                    if randomize_seed or num_images > 1:
                        seed = random.randint(0, MAX_SEED)
                
                    generator = torch.Generator(device=pipe.device).manual_seed(seed)
                
                    iteration_start_time = time.time()
                    result_images = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image_embeds=face_emb,
                        image=control_images,
                        control_mask=control_mask,
                        controlnet_conditioning_scale=control_scales,
                        num_inference_steps=num_steps,
                        head_only_control=head_only_control,
                        controlnet_selection=controlnet_selection,
                        guidance_scale=guidance_scale,
                        height=height_target,
                        width=width_target,                
                        generator=generator,
                        face_info=face_info,
                        end_cfg=guidance_threshold,
                        device=pipe.device,
                        dtype=dtype
                    ).images

                    image = result_images[0]
                    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                    output_path = f"outputs/{current_time}.png"
                    if not os.path.exists("outputs"):
                        os.makedirs("outputs")
                    meta = PngImagePlugin.PngInfo()
                    meta.add_text("Upload a photo of your face full path", str(face_image_path))
                    meta.add_text("Upload a photo of your face image file name", os.path.basename(face_image_path) if face_image_path else "")
                    meta.add_text("Upload a reference pose image (Optional) full path", str(pose_image_path))
                    meta.add_text("Upload a reference pose image (Optional) image file name", os.path.basename(pose_image_path) if pose_image_path else "")
                    meta.add_text("", "")
                    meta.add_text("Prompt", str(org_prompt))
                    meta.add_text("Final Prompt (Includes Style)", str(prompt))
                    meta.add_text("", "")
                    meta.add_text("Negative Prompt", str(org_negative))
                    meta.add_text("Final Negative Prompt (Includes Style)", str(negative_prompt))
                    meta.add_text("Enable Fast Inference with LCM", str(enable_LCM))
                    meta.add_text("Depth Estimator", str(depth_type))
                    meta.add_text("IdentityNet strength (for fidelity)", str(identitynet_strength_ratio))
                    meta.add_text("Pose strength", str(pose_strength))
                    meta.add_text("Canny strength", str(canny_strength))
                    meta.add_text("LoRA Scale", str(lora_scale))
                    meta.add_text("Depth strength", str(depth_strength))
                    meta.add_text("used Controlnets", ", ".join(controlnet_selection))
                    meta.add_text("Dropdown Selected Model", str(model_dropdown))
                    meta.add_text("Default Model - Used If None Selected", str(default_model))
                    meta.add_text("Full Model Path - Used If Set", str(model_input))
                    meta.add_text("Select LoRA models", ", ".join(lora_model_dropdown))
                    meta.add_text("Target Image Width", str(width_target))
                    meta.add_text("Target Image Height", str(height_target))
                    meta.add_text("Style Template", str(style_name))
                    meta.add_text("Image Adapter Strength", str(adapter_strength_ratio))
                    meta.add_text("Number Of Sample Steps", str(num_steps))
                    meta.add_text("CFG Scale", str(guidance_scale))
                    meta.add_text("CFG Threshold", str(guidance_threshold))
                    meta.add_text("Used Seed", str(seed))
                    meta.add_text("Enhance non-face region", str(enhance_face_region))
                    meta.add_text("Used Scheduler", str(scheduler))
                    image.save(output_path, "PNG", pnginfo=meta)
                    images_generated.append(image)

                    iteration_end_time = time.time()
                    iteration_time = iteration_end_time - iteration_start_time
                    print(f"Image {i + 1}/{num_images} generated in {iteration_time:.2f} seconds.")            
                    if num_images > 1 and  ENABLE_CPU_OFFLOAD:                 
                        restart_cpu_offload(adapter_strength_ratio)
        
        total_time = time.time() - start_time
        average_time_per_image = total_time / num_images if num_images else 0
        clean_memory()
        print(f"{len(images_generated)} images generated in {total_time:.2f} seconds, average {average_time_per_image:.2f} seconds per image.")
        return images_generated, gr.update(visible=True), seed

    def generate_all_variations(
        variation_type,
        face_file,
        pose_file,
        prompt,
        negative_prompt,
        style_name,
        num_steps,
        identitynet_strength_ratio,
        adapter_strength_ratio,
        pose_strength,
        canny_strength,
        depth_strength,
        controlnet_selection,
        head_only_control,
        guidance_scale,
        seed,
        randomize_seed,  # Add this line
        scheduler,
        enable_LCM,        
        enhance_face_region,
        model_input,
        model_dropdown,
        width,
        height,
        num_images,
        guidance_threshold,
        depth_type,
        lora_model_dropdown,
        lora_scale,
        progress=gr.Progress(track_tqdm=True)
    ):
        all_images = []

        if variation_type == "styles":
            variations = STYLE_NAMES
            total_variations = len(variations)
            variable_name = "style"
        elif variation_type == "models":
            variations = get_model_names()
            total_variations = len(variations)
            variable_name = "model"
        else:
            raise ValueError("Invalid variation_type. Must be 'styles' or 'models'.")

        progress_text = gr.Textbox(label="Generation Progress", interactive=False)
        yield [], gr.update(visible=True), gr.update(visible=True, value="Starting generation...")

        start_time = time.time()
        for index, variation in enumerate(variations, start=1):
            if variation_type == "styles":
                current_style = variation
                current_model = model_dropdown
            else:
                current_style = style_name
                current_model = variation

            images, _, new_seed = generate_image(
            variation_type,
            face_file,
            pose_file,
            prompt,
            negative_prompt,
            current_style,
            num_steps,
            identitynet_strength_ratio,
            adapter_strength_ratio,
            pose_strength,
            canny_strength,
            depth_strength,
            controlnet_selection,
            guidance_scale,
            seed,
            randomize_seed,  # Add this line
            scheduler,
            enable_LCM,        
            enhance_face_region,
            model_input,
            current_model,
            width,
            height,
            num_images,
            guidance_threshold,
            depth_type,
            lora_model_dropdown,
            lora_scale,
            head_only_control,
            progress
            )
            all_images.extend(images)

            elapsed_time = time.time() - start_time
            images_generated = index * num_images
            total_images = total_variations * num_images
            images_left = total_images - images_generated
            percent_complete = (images_generated / total_images) * 100

            eta = (elapsed_time / images_generated) * images_left if images_generated > 0 else 0

            progress_message = f"Generated {images_generated}/{total_images} images ({percent_complete:.2f}% complete)\n"
            progress_message += f"Current {variable_name}: {variation}\n"
            progress_message += f"Elapsed time: {elapsed_time:.2f} seconds\n"
            progress_message += f"Estimated time remaining: {eta:.2f} seconds"
            print(progress_message)

            yield all_images, gr.update(visible=True), gr.update(visible=True, value=progress_message)

        final_message = f"Generation complete! Total time: {time.time() - start_time:.2f} seconds"
        yield all_images, gr.update(visible=True), gr.update(visible=True, value=final_message)

    def set_pipe_controlnet(identitynet_strength_ratio, pose_strength, canny_strength, depth_strength, controlnet_selection, width_target, height_target, face_kps, img_controlnet):
        global pipe
        if len(controlnet_selection) > 0:
            controlnet_scales = {
                "pose": pose_strength,
                "canny": canny_strength,
                "depth": depth_strength,
            }
            pipe.controlnet = MultiControlNetModel(
                [controlnet]
                + [controlnet_map[s] for s in controlnet_selection]
            )
            control_scales = [float(identitynet_strength_ratio)] + [
                controlnet_scales[s] for s in controlnet_selection
            ]
            control_images = [face_kps] + [
                controlnet_map_fn[s](img_controlnet).resize((width_target, height_target))
                for s in controlnet_selection
            ]
        else:
            pipe.controlnet = controlnet
            control_scales = float(identitynet_strength_ratio)
            control_images = face_kps
        return control_scales,control_images

    # Description
    title = r"""
    <h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
    """

    description = r"""
    <b>Only available to Premium Members and latest version : https://www.patreon.com/posts/1-click-very-gui-97769887<br>

    How to use:<br>
    1. Upload an image with a face. For images with multiple faces, we will only detect the largest face. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
    2. (Optional) You can upload another image as a reference for the face pose. If you don't, we will use the first detected face image to extract facial landmarks. If you use a cropped face at step 1, it is recommended to upload it to define a new face pose.
    3. (Optional) You can select multiple ControlNet models to control the generation process. The default is to use the IdentityNet only. The ControlNet models include pose skeleton, canny, and depth. You can adjust the strength of each ControlNet model to control the generation process.
    4. Enter a text prompt, as done in normal text-to-image models.
    """

    article = r"""
    """

    tips = r"""
    ### Usage tips of InstantID
    1. If you're not satisfied with the similarity, try increasing the weight of "IdentityNet Strength" and "Adapter Strength."    
    2. If you feel that the saturation is too high, first decrease the Adapter strength. If it remains too high, then decrease the IdentityNet strength.
    3. If you find that text control is not as expected, decrease Adapter strength.
    """

    css = """
    .gradio-container {width: 85% !important}
    """
    with gr.Blocks(css=css) as demo:
        with gr.Tab("InstantId - V17"):
            gr.Markdown(title)
            gr.Markdown(description)
            
            with gr.Row():
                with gr.Column():
                    with gr.Row(equal_height=True):
                        # upload face image
                        face_file = gr.Image(
                            label="Upload a photo of your face", type="filepath"
                        )
                        # optional: upload a reference pose image
                        pose_file = gr.Image(
                            label="Upload a reference pose image (Optional)",
                            type="filepath",
                        )
                    with gr.Row(equal_height=True):
                        progress_status = gr.Label()
                        usage_tips = gr.Markdown(
                        label="InstantID Usage Tips", value=tips, visible=True
                    )
                with gr.Column(scale=1):
            
                    gallery = gr.Gallery(label="Generated Images", columns=1, rows=1, height=512, format="png", preview=True, allow_preview=True)
                    

            with gr.Row():
                with gr.Column():

                    # prompt
                    prompt = gr.Textbox(
                        label="Prompt",
                        info="Give simple prompt is enough to achieve good face fidelity",
                        placeholder="A photo of a person",
                        value="",
                    
                    )

                    submit = gr.Button("Submit", variant="primary")

                    negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="low quality",
                            value="(text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, monochrome",
                        )
                    with gr.Row():
                        with gr.Column():
                            config_name = gr.Textbox(label="Configuration Name")
                        with gr.Column():
                            save_config_btn = gr.Button("Save Configuration")
                    config_dropdown = gr.Dropdown(label="Saved Configurations", choices=get_config_list(), value=get_latest_config())
                    with gr.Row():
                        with gr.Column():
                            load_config_btn = gr.Button("Load Configuration")
                        with gr.Column():
                            refresh_config_btn = gr.Button("Refresh Configuration List")

                with gr.Column():
                    model_names = get_model_names()
                    with gr.Row():
                        with gr.Column():
                            refresh_models = gr.Button("Refresh Models (Only SDXL Works)")
                            lora_model_names = get_lora_model_names()
                            lora_model_dropdown = gr.Dropdown(label="Select LoRA models",multiselect=True, choices=lora_model_names, value=[])
                            lora_scale = gr.Slider(label="Change applied LoRA Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.05,
                        value=1.0,
                    )
                        with gr.Column():
                            model_dropdown = gr.Dropdown(label="Select model from models folder", choices=model_names, value=None)
                            btn_open_outputs = gr.Button("Open Outputs Folder")
                            btn_open_outputs.click(fn=open_folder)
                            generate_all_styles_button = gr.Button("Generate With All Styles (Loop)", variant="secondary")
                            generate_all_models_button = gr.Button("Generate With All Models (Loop)", variant="secondary")
                    with gr.Row():
                        with gr.Column():
                            model_input = gr.Textbox(label="Hugging Face model repo name or local file full path", value="", placeholder="Enter model name or path")						
                    with gr.Row():
                        with gr.Column():
                            width = gr.Slider(label="Width", value=1280,step=64, visible=True,minimum=512,maximum=2048)
                        with gr.Column():
                            height = gr.Slider(label="Height", value=1280,step=64, visible=True,minimum=512,maximum=2048)
                    with gr.Row():
                            num_images = gr.Number(label="How many Images to Generate", value=1, step=1, minimum=1, visible=True)
                            style = gr.Dropdown(
                        label="Style template",
                        choices=STYLE_NAMES,
                        value=DEFAULT_STYLE_NAME,
                    )  

            with gr.Row():       
                with gr.Column():       
                    depth_type = gr.Dropdown(
                    label="Depth Estimator",
                    choices=DEPTH_ESTIMATOR,
                    value="LiheYoung/depth_anything")

                with gr.Column():                
                    enable_LCM = gr.Checkbox(
                    label="Enable Fast Inference with LCM", value=enable_lcm_arg,
                    info="LCM speeds up the inference step, the trade-off is the quality of the generated image. It performs better with portrait face images rather than distant faces",
                )
   
         
            with gr.Row():         
                with gr.Column():  
                    # strength
                    identitynet_strength_ratio = gr.Slider(
                        label="IdentityNet strength (for fidelity)",
                        minimum=0,
                        maximum=3,
                        step=0.05,
                        value=0.80,
                    )
                with gr.Column():  
                    adapter_strength_ratio = gr.Slider(
                        label="Image adapter strength (for detail)",
                        minimum=0,
                        maximum=3,
                        step=0.05,
                        value=0.80,
                    )


            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        head_only_control = gr.CheckboxGroup(
                            label="Apply Head-Only Control to",
                            choices=["pose", "canny", "depth"],
                            value=[],  # Default to empty list
                        )
                    with gr.Row():

                        pose_strength = gr.Slider(
                            label="Pose strength",
                            minimum=0,
                            maximum=3,
                            step=0.05,
                            value=0.40,
                        )
                        canny_strength = gr.Slider(
                            label="Canny strength",
                            minimum=0,
                            maximum=3,
                            step=0.05,
                            value=0.40,
                        )
                        depth_strength = gr.Slider(
                            label="Depth strength",
                            minimum=0,
                            maximum=3,
                            step=0.05,
                            value=0.40,
                        )
                    with gr.Row():
                        controlnet_selection = gr.CheckboxGroup(
                            ["pose", "canny", "depth"], label="Controlnet", value=None,
                            info="Use pose for skeleton inference, canny for edge detection, and depth for depth map estimation. You can try all three to control the generation process"
                        )
                with gr.Column():

                    with gr.Row():

                        num_steps = gr.Slider(
                            label="Number of sample steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=5 if enable_lcm_arg else 30,
                        )
                        guidance_scale = gr.Slider(
                            label="CFG scale",
                            minimum=0.1,
                            maximum=20.0,
                            step=0.1,
                            value=0.0 if enable_lcm_arg else 5.0,
                        )
                        guidance_threshold = gr.Slider(
                            label="CFG threshold",
                            minimum=0.4,
                            maximum=1,
                            step=0.1,
                            value=1,
                        )
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=42,
                        )

                        
                        enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        schedulers = [
                            "DEISMultistepScheduler",
                            "HeunDiscreteScheduler",
                            "EulerAncestralDiscreteScheduler",
                            "EulerDiscreteScheduler",
                            "DPMSolverMultistepScheduler",
                            "DPMSolverMultistepScheduler-Karras",
                            "DPMSolverMultistepScheduler-Karras-SDE",
                            "UniPCMultistepScheduler"
                        ]
                        scheduler = gr.Dropdown(
                            label="Schedulers",
                            choices=schedulers,
                            value="EulerDiscreteScheduler",
                        )
                refresh_models.click(
                    fn=refresh_model_names,
                    outputs=[model_dropdown, lora_model_dropdown]
                )
                submit.click(
                    fn=generate_image,
                    inputs=[
                        gr.Textbox(value="single", visible=False),  # Hidden input to specify single image generation
                        face_file,
                        pose_file,
                        prompt,
                        negative_prompt,
                        style,
                        num_steps,
                        identitynet_strength_ratio,
                        adapter_strength_ratio,
                        pose_strength,
                        canny_strength,
                        depth_strength,
                        controlnet_selection,
                        guidance_scale,
                        seed,
                        randomize_seed,  # Add this line
                        scheduler,
                        enable_LCM,                    
                        enhance_face_region,
                        model_input,
                        model_dropdown,
                        width,
                        height,
                        num_images,
                        guidance_threshold,
                        depth_type,
                        lora_model_dropdown,
                        lora_scale,
                        head_only_control
                    ],
                    outputs=[gallery, progress_status, seed],
                )


                enable_LCM.input(
                    fn=toggle_lcm_ui,
                    inputs=[enable_LCM],
                    outputs=[num_steps, guidance_scale],
                )
        with gr.Tab("Image Metadata"):

            with gr.Row():
                set_metadata_button = gr.Button("Load & Set Metadata Settings")
            with gr.Row():
                metadata_image_input = gr.Image(type="filepath", label="Upload Image")                
                metadata_output = gr.Textbox(label="Image Metadata", lines=25, max_lines=50)
            metadata_image_input.change(fn=read_image_metadata, inputs=[metadata_image_input], outputs=[metadata_output])
        with gr.Tab("Model Downloader"):
            gr.Markdown("## Model Downloader")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Checkpoints")
                    checkpoint_files = gr.Textbox(label="Available Checkpoint Files", value="\n".join(get_model_files(used_model_path)))
        
                with gr.Column():
                    gr.Markdown("### LoRAs")
                    lora_files = gr.Textbox(label="Available LoRA Files", value="\n".join(get_model_files(used_lora_path)))

            with gr.Row():
                model_url = gr.Textbox(label="Model Download URL")
                model_name = gr.Textbox(label="Model Name (Optional)")
                model_type = gr.Dropdown(choices=["Checkpoint", "LoRA"], value="Checkpoint", label="Model Type")

            download_button = gr.Button("Download Model")
            download_status = gr.Textbox(label="Download Status")

            gr.Markdown("### Pre-defined Models")
            predefined_model = gr.Dropdown(choices=sorted(list(PREDEFINED_MODELS.keys())), label="Select Pre-defined Model")
            download_predefined_button = gr.Button("Download Selected Model")
    
            # New button for downloading all predefined models
            download_all_predefined_button = gr.Button("Download All Pre-defined Models")

            refresh_button = gr.Button("Refresh File Lists")

            download_button.click(
                fn=download_model,
                inputs=[model_url, model_name, model_type],
                outputs=download_status
            )

            download_predefined_button.click(
                fn=download_predefined_model,
                inputs=[predefined_model],
                outputs=download_status
            )

            # New click event for downloading all predefined models
            download_all_predefined_button.click(
                fn=download_all_predefined_models,
                outputs=download_status
            )

            def refresh_lists():
                return (
                "\n".join(sorted(get_model_files(used_model_path))),
                "\n".join(sorted(get_model_files(used_lora_path)))
            )

            refresh_button.click(
                fn=refresh_lists,
                outputs=[checkpoint_files, lora_files]
            )
        set_metadata_button.click(fn=set_metadata_settings, inputs=[metadata_image_input], outputs=[prompt, negative_prompt, enable_LCM, depth_type, identitynet_strength_ratio, adapter_strength_ratio, pose_strength, canny_strength, depth_strength, controlnet_selection, model_dropdown, model_input, lora_model_dropdown, width, height, style, num_steps, guidance_scale, guidance_threshold, seed, enhance_face_region, scheduler, face_file, pose_file,lora_scale])
        generate_all_styles_button.click(
            fn=generate_all_variations,
            inputs=[
                gr.Textbox(value="styles", visible=False)  # Hidden input to specify variation type
            ] + [
                face_file,
                pose_file,
                prompt,
                negative_prompt,
                style,
                num_steps,
                identitynet_strength_ratio,
                adapter_strength_ratio,
                pose_strength,
                canny_strength,
                depth_strength,
                controlnet_selection,
                head_only_control,
                guidance_scale,
                seed,
                randomize_seed,  # Add this line
                scheduler,
                enable_LCM,                    
                enhance_face_region,
                model_input,
                model_dropdown,
                width,
                height,
                num_images,
                guidance_threshold,
                depth_type,
                lora_model_dropdown,
                lora_scale
            ],
            outputs=[gallery, progress_status],
        )

        adapter_strength_ratio.change(
            fn=update_ip_adapter_scale,
            inputs=[adapter_strength_ratio],
        )

        generate_all_models_button.click(
            fn=generate_all_variations,
            inputs=[
                gr.Textbox(value="models", visible=False)  # Hidden input to specify variation type
            ] + [
                face_file,
                pose_file,
                prompt,
                negative_prompt,
                style,
                num_steps,
                identitynet_strength_ratio,
                adapter_strength_ratio,
                pose_strength,
                canny_strength,
                depth_strength,
                controlnet_selection,
                head_only_control,
                guidance_scale,
                seed,
                randomize_seed,  # Add this line
                scheduler,
                enable_LCM,                    
                enhance_face_region,
                model_input,
                model_dropdown,
                width,
                height,
                num_images,
                guidance_threshold,
                depth_type,
                lora_model_dropdown,
                lora_scale
            ],
            outputs=[gallery, progress_status],
        )

        save_config_btn.click(
            fn=save_config,
            inputs=[config_name, prompt, negative_prompt, style, num_steps, identitynet_strength_ratio,
                    adapter_strength_ratio, pose_strength, canny_strength, depth_strength,
                    controlnet_selection, guidance_scale, seed, randomize_seed, scheduler,
                    enable_LCM, enhance_face_region, model_input, model_dropdown, width,
                    height, num_images, guidance_threshold, depth_type, lora_model_dropdown, lora_scale],
            outputs=[config_dropdown, config_dropdown]
        )

        load_config_btn.click(
            fn=load_config,
            inputs=[config_dropdown],
            outputs=[prompt, negative_prompt, style, num_steps, identitynet_strength_ratio,
                     adapter_strength_ratio, pose_strength, canny_strength, depth_strength,
                     controlnet_selection, guidance_scale, seed, randomize_seed, scheduler,
                     enable_LCM, enhance_face_region, model_input, model_dropdown, width,
                     height, num_images, guidance_threshold, depth_type, lora_model_dropdown, lora_scale]
        )

        refresh_config_btn.click(
            fn=refresh_config_list,
            outputs=[config_dropdown]
        ).then(
            fn=update_config_dropdown,
            inputs=[config_dropdown],
            outputs=[config_dropdown]
        )

        # Load the latest config on startup
        demo.load(
            fn=load_config,
            inputs=[config_dropdown],
            outputs=[prompt, negative_prompt, style, num_steps, identitynet_strength_ratio,
                     adapter_strength_ratio, pose_strength, canny_strength, depth_strength,
                     controlnet_selection, guidance_scale, seed, randomize_seed, scheduler,
                     enable_LCM, enhance_face_region, model_input, model_dropdown, width,
                     height, num_images, guidance_threshold, depth_type, lora_model_dropdown, lora_scale]
        )

        gr.Markdown(article)
    demo.launch(inbrowser=True, share=share)



if __name__ == "__main__":
    main(args.pretrained_model_folder,args.enable_LCM,args.share)