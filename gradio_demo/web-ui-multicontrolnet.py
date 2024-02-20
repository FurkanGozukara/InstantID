import sys
sys.path.append("./")

from typing import Tuple

import time
from datetime import datetime
import os
import cv2
import math
import torch
import random
import numpy as np
import argparse

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from style_template import styles
from pipelines.pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl, get_torch_device
from controlnet_util import load_controlnet, load_depth_estimator as load_depth, get_depth_map, get_depth_anything_map, get_canny_image
from common.util import clean_memory

import gradio as gr

parser = argparse.ArgumentParser()
parser.add_argument(
"--pretrained_model_folder", type=str, default=None
)
parser.add_argument(
"--enable_LCM", type=bool, default=os.environ.get("ENABLE_LCM", False)
)
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--fp16", action="store_true", help="fp16")
parser.add_argument("--share", action="store_true", help="Enable Gradio app sharing")

args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True   

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()

dtype = torch.bfloat16
if(args.fp16):
   dtype = torch.float16

dtype = dtype if str(device).__contains__("cuda") else torch.float32
ENABLE_CPU_OFFLOAD = True if args.lowvram else False
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"
DEPTH_ESTIMATOR = ["LiheYoung/depth_anything", "Intel/dpt-hybrid-midas"]
_pretrained_model_folder = None
default_model = "wangqixun/YamerMIX_v8"
last_loaded_model = None
last_loaded_scheduler = None
last_loaded_depth_estimator = None

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

def get_model_names():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_files = []
    model_files.append(default_model)
    model_files = model_files + [f for f in os.listdir(models_dir) if f.endswith('.safetensors')]
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
        )
        scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)
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
  
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(            
            model_path,
            controlnet=[controlnet],
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        )       
    return pipe

def assign_last_params(adapter_strength_ratio, with_cpu_offload):
    global pipe
    
    set_ip_adapter(adapter_strength_ratio)
    
    # apply improvements    
    if with_cpu_offload:                 
        pipe.enable_model_cpu_offload()        
    else:
        pipe.to(device)
    
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling() 

def set_ip_adapter(adapter_strength_ratio):    
    pipe.load_ip_adapter_instantid(face_adapter)    
    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    

def load_scheduler(pretrained_model_folder, scheduler, with_LCM):
     
    # load and disable LCM
    lora_model = "latent-consistency/lcm-lora-sdxl" if not pretrained_model_folder else fr"{pretrained_model_folder}/latent-consistency/lcm-lora-sdxl"
    pipe.load_lora_weights(lora_model)  

    if with_LCM:
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
    _pretrained_model_folder = pretrained_model_folder
   
    def reload_pipe(model_input, model_dropdown, scheduler, adapter_strength_ratio, with_LCM, depth_type):
        global pipe  # Declare pipe as a global variable thas_cpu_offload manage it when the model changes
        global last_loaded_model, last_loaded_scheduler, last_loaded_depth_estimator
     
        # Trim the model_input to remove any leading or trailing whitespace
        model_input = model_input.strip() if model_input else None

        # Determine the model to load
        model_to_load = model_input if model_input else os.path.join('models', model_dropdown) if (model_dropdown and model_dropdown != default_model) else default_model if model_dropdown == default_model else None

        # Return early if no model is selected or inputted
        if not model_to_load:
            print("No model selected or inputted. Default model will be used.")
            model_to_load = default_model
       
        # Reload CPU offload to fix bug for half mode
        if pipe and ENABLE_CPU_OFFLOAD:
            pipe.disable_xformers_memory_efficient_attention()
            set_ip_adapter(adapter_strength_ratio)            
            from pipelines.pipeline_common import optionally_disable_offloading
            optionally_disable_offloading(pipe)
            pipe.enable_model_cpu_offload()
            pipe.enable_xformers_memory_efficient_attention()

        if not pipe:
            pipe = None            
            #load controlnet
            load_controlnet_open_pose(pretrained_model_folder)
            load_depth_estimator(pretrained_model_folder, depth_type)
            clean_memory()

            pipe = load_model(_pretrained_model_folder, model_to_load)
            last_loaded_model = model_to_load
            last_loaded_scheduler = scheduler
            last_loaded_depth_estimator = depth_type
            load_scheduler(pretrained_model_folder, scheduler, with_LCM)
            assign_last_params(adapter_strength_ratio, ENABLE_CPU_OFFLOAD)

        # Reload depth estimator if neeed
        if (pipe and model_to_load == last_loaded_model
            and depth_type != last_loaded_depth_estimator):          
            load_depth_estimator(pretrained_model_folder, depth_type)  

        # Reload scheduler if needed
        if (pipe and model_to_load == last_loaded_model 
            and scheduler != last_loaded_scheduler):            
            load_scheduler(pretrained_model_folder, scheduler, with_LCM)
            last_loaded_scheduler = scheduler
       
        # Reload model if needed
        if (pipe and model_to_load != last_loaded_model):                                              
            # Reload model        
            pipe = load_model(_pretrained_model_folder, model_to_load)
            last_loaded_model = model_to_load
            last_loaded_scheduler = scheduler
            last_loaded_depth_estimator = depth_type
            load_scheduler(pretrained_model_folder, scheduler, with_LCM)
            assign_last_params(adapter_strength_ratio, ENABLE_CPU_OFFLOAD)
        
        print("Model loaded successfully.")
        clean_memory()

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
                        pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):
        w, h = input_image.size
    
        if size is not None:
            #print("size is not none")
            target_width, target_height = size
            target_aspect_ratio = target_width / target_height
            image_aspect_ratio = w / h

            if image_aspect_ratio > target_aspect_ratio:
                # Image is wider than desired aspect ratio
                new_width = int(h * target_aspect_ratio)
                new_height = h
                left = (w - new_width) / 2
                top = 0
                right = (w + new_width) / 2
                bottom = h
            else:
                # Image is taller than desired aspect ratio
                new_height = int(w / target_aspect_ratio)
                new_width = w
                top = 0  # Changed from: top = (h - new_height) / 2
                left = 0
                bottom = new_height  # Changed from: bottom = (h + new_height) / 2
                right = w

            # Crop the image to the target aspect ratio
            input_image = input_image.crop((left, top, right, bottom))
            print("input image cropped according to target width and height")
            w, h = input_image.size  # Update dimensions after cropping
        
            # Resize the image to the specified size
            input_image = input_image.resize(size, mode)
            input_image.save('temp.png', 'PNG', overwrite=True)

        else:
            # Resize logic when size is not specified
            #print("size is none")
            ratio = min_side / min(h, w)
            w, h = round(ratio * w), round(ratio * h)
            ratio = max_side / max(h, w)
            input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
            w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
            h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
            input_image = input_image.resize([w_resize_new, h_resize_new], mode)
            input_image.save('temp2.png', 'PNG', overwrite=True)

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

    def apply_style(
        style_name: str, positive: str, negative: str = ""
    ) -> Tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + " " + negative

    def generate_image(
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
        progress=gr.Progress(track_tqdm=True),
    ):
        global controlnet_map, controlnet_map_fn
       
        if face_image_path is None:
            raise gr.Error(
                f"Cannot find any input face image! Please upload the face image"
            )

        if prompt is None:
            prompt = "a person"

        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        face_image = load_image(face_image_path)
        face_image = resize_img(face_image,size=(width_target, height_target))
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        face_info = app.get(face_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(
                f"Unable to detect a face in the image. Please upload a different photo with a clear face."
            )

        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
        )[-1]
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
                raise gr.Error(
                    f"Cannot find any face in the reference image! Please upload another person image"
                )

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

        reload_pipe(model_input, model_dropdown, scheduler, adapter_strength_ratio, enable_LCM, depth_type)        
        control_scales, control_images = set_pipe_controlnet(identitynet_strength_ratio, pose_strength, canny_strength, depth_strength, controlnet_selection, width_target, height_target, face_kps, img_controlnet)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        images_generated = []
        start_time = time.time()

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")
        
        for i in range(num_images):
            if num_images > 1:
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
                guidance_scale=guidance_scale,
                height=height_target,
                width=width_target,                
                generator=generator,
                end_cfg=guidance_threshold
            ).images

            image = result_images[0]
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            output_path = f"outputs/{current_time}.png"
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            image.save(output_path)
            images_generated.append(image)

            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            
            print(f"Image {i + 1}/{num_images} generated in {iteration_time:.2f} seconds.")            
            
        total_time = time.time() - start_time
        average_time_per_image = total_time / num_images if num_images else 0
        clean_memory()
        print(f"{len(images_generated)} images generated in {total_time:.2f} seconds, average {average_time_per_image:.2f} seconds per image.")
        return images_generated, gr.update(visible=True)

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
    <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

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
        # description
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
            with gr.Column(scale=1):
            
                gallery = gr.Gallery(label="Generated Images", columns=1, rows=1, height=512)
                usage_tips = gr.Markdown(
                    label="InstantID Usage Tips", value=tips, visible=False
                )
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

            with gr.Column():
                model_names = get_model_names()
                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(label="Select model from models folder", choices=model_names, value=None)
                    with gr.Column():
                        model_input = gr.Textbox(label="Hugging Face model repo name or local file full path", value="", placeholder="Enter model name or path")
                with gr.Row():
                    with gr.Column():
                        width = gr.Number(label="Width", value=1280, visible=True)
                    with gr.Column():
                        height = gr.Number(label="Height", value=1280, visible=True)
                    with gr.Column():
                        num_images = gr.Number(label="How many Images to Generate", value=1, step=1, minimum=1, visible=True)

        with gr.Row():       
            with gr.Column():                
                enable_LCM = gr.Checkbox(
                    label="Enable Fast Inference with LCM", value=enable_lcm_arg,
                    info="LCM speeds up the inference step, the trade-off is the quality of the generated image. It performs better with portrait face images rather than distant faces",
                )
        with gr.Row():       
            with gr.Column():       
                depth_type = gr.Dropdown(
                label="Depth Estimator",
                choices=DEPTH_ESTIMATOR,
                value="LiheYoung/depth_anything")

            with gr.Column():       
                style = gr.Dropdown(
                    label="Style template",
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                )           
        with gr.Row():         
            with gr.Column():  
                # strength
                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength (for fidelity)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
            with gr.Column():  
                adapter_strength_ratio = gr.Slider(
                    label="Image adapter strength (for detail)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )


        with gr.Row():
            with gr.Column():
                with gr.Row():

                    pose_strength = gr.Slider(
                        label="Pose strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.40,
                    )
                    canny_strength = gr.Slider(
                        label="Canny strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.40,
                    )
                    depth_strength = gr.Slider(
                        label="Depth strength",
                        minimum=0,
                        maximum=1.5,
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

                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)
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

            submit.click(
                fn=remove_tips,
                outputs=usage_tips,
            ).then(
                fn=randomize_seed_fn,
                inputs=[seed, randomize_seed],
                outputs=seed,
                queue=False,
                api_name=False,
            ).then(
                fn=generate_image,
                inputs=[
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
                    scheduler,
                    enable_LCM,                    
                    enhance_face_region,model_input,model_dropdown,width,height,num_images,guidance_threshold,depth_type
                ],
                outputs=[gallery, usage_tips],
            )

            enable_LCM.input(
                fn=toggle_lcm_ui,
                inputs=[enable_LCM],
                outputs=[num_steps, guidance_scale],
                queue=False,
            )

        gr.Markdown(article)
    demo.launch(inbrowser=True, share=share)

if __name__ == "__main__":
    main(args.pretrained_model_folder,args.enable_LCM,args.share)