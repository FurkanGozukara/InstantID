import sys
sys.path.append('./')

import os
import cv2
import math
import torch
from datetime import datetime
import random
import numpy as np
import argparse

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from style_template import styles
from pipelines.pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl, get_torch_device
from common.util import clean_memory

import gradio as gr

MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"
_pretrained_model_folder = None
default_model = "wangqixun/YamerMIX_v8"
last_loaded_model = None
pipe = None

# Load face encoder
app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

face_adapter = f'checkpoints/ip-adapter.bin'
controlnet_path = f'checkpoints/ControlNetModel'

# Load pipeline face ControlNetModel
controlnet = ControlNetModel.from_pretrained(
    controlnet_path, 
    torch_dtype=dtype)

def get_model_names():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.safetensors')]
    return model_files

def load_model(pretrained_model_folder, model_name):
    
    print(f"Loading model: {model_name}")

    if model_name.endswith(
        ".ckpt"
    ) or model_name.endswith(".safetensors"):
        model_path = model_name
    else:    
        if pretrained_model_folder:
            model_path = fr"{pretrained_model_folder}/{model_name}"
        else:
            model_path = model_name
    
    if model_name.endswith(
        ".ckpt"
    ) or model_name.endswith(".safetensors"):        
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
            controlnet=controlnet,
        )

    else:
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        )       
    return pipe

def assign_last_params():
    global pipe
    pipe.load_ip_adapter_instantid(face_adapter)    
    pipe.to(device)
    
    # apply improvements
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()    
    pipe.enable_xformers_memory_efficient_attention()

def main(pretrained_model_folder, share=False):
    global _pretrained_model_folder
    _pretrained_model_folder = pretrained_model_folder

    def reload_pipe(model_input, model_dropdown, adapter_strength_ratio, with_cpu_offload):
        global pipe  # Declare pipe as a global variable thas_cpu_offloado manage it when the model changes
        global last_loaded_model

        # Trim the model_input to remove any leading or trailing whitespace
        model_input = model_input.strip() if model_input else None

        # Determine the model to load
        model_to_load = model_input if model_input else os.path.join('models', model_dropdown) if model_dropdown else None

        # Return early if no model is selected or inputted
        if not model_to_load:
            print("No model selected or inputted. Default model will be used.")
            model_to_load = default_model

         # Proceed with reloading the model if it's different from the last loaded model
        if not with_cpu_offload:                
            # Load the new model
            pipe = load_model(_pretrained_model_folder, model_to_load)
            last_loaded_model = model_to_load
            assign_last_params()
        else:
            if (model_to_load != last_loaded_model):                      
                # Properly discard the old pipe if it exists
                if hasattr(pipe, 'scheduler'):
                    del pipe.scheduler
                
            # Load the new model
            pipe = load_model(_pretrained_model_folder, model_to_load)
            last_loaded_model = model_to_load
            assign_last_params()
        
        if with_cpu_offload:            
            pipe.enable_model_cpu_offload()     

        # load and disable LCM
        lora_model = "latent-consistency/lcm-lora-sdxl" if not _pretrained_model_folder else fr"{_pretrained_model_folder}/latent-consistency/lcm-lora-sdxl"
        pipe.load_lora_weights(lora_model)        
        pipe.disable_lora()        
        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe.set_ip_adapter_scale(adapter_strength_ratio)

        print("Model loaded successfully.")
        clean_memory()

    def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        return seed

    def swap_to_gallery(images):
        return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

    def upload_example_to_gallery(images, prompt, style, negative_prompt):
        return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

    def remove_back_to_files():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    def remove_tips():
        return gr.update(visible=False)

    def convert_from_cv2_to_image(img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def convert_from_image_to_cv2(img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
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
            polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
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

    def resize_img2(input_image, max_side=1280, min_side=1024, size=None, 
                pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):

            w, h = input_image.size
            if size is not None:
                w_resize_new, h_resize_new = size
            else:
                ratio = min_side / min(h, w)
                w, h = round(ratio*w), round(ratio*h)
                ratio = max_side / max(h, w)
                input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
                w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
                h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
            input_image = input_image.resize([w_resize_new, h_resize_new], mode)

            if pad_to_max_side:
                res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
                offset_x = (max_side - w_resize_new) // 2
                offset_y = (max_side - h_resize_new) // 2
                res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
                input_image = Image.fromarray(res)
            return input_image

    def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + ' ' + negative

    def generate_image(face_image, 
                       pose_image, 
                       prompt, 
                       negative_prompt, 
                       style_name, 
                       num_steps, 
                       identitynet_strength_ratio, 
                       adapter_strength_ratio, 
                       guidance_scale, 
                       seed, 
                       width, 
                       height, 
                       num_images, 
                       model_input, 
                       model_dropdown,
                       enable_CPUOffload,
                       progress=gr.Progress(track_tqdm=True)
    ):
      
        start_time = datetime.now()
        images = []

        if face_image is None:
            raise gr.Error("Cannot find any input face image! Please upload the face image")

        if prompt is None:
            prompt = "a person"

        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
    
        face_image = load_image(face_image)
        face_image = resize_img(face_image, size=(width, height))
        face_image_cv2 = convert_from_image_to_cv2(face_image)
    
        face_info = app.get(face_image_cv2)
    
        if len(face_info) == 0:
            raise gr.Error(f"Cannot find any face in the image! Please upload another person image")
    
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]  
        face_emb = face_info['embedding']
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info['kps'])
    
        if pose_image is not None:
            pose_image = load_image(pose_image)
            pose_image = resize_img(pose_image, size=(width, height))
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)
        
            face_info = app.get(pose_image_cv2)
        
            if len(face_info) == 0:
                raise gr.Error(f"Cannot find any face in the reference image! Please upload another person image")
        
            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info['kps'])
        
        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")
    
        reload_pipe(model_input, model_dropdown, adapter_strength_ratio, enable_CPUOffload)       
        
        for i in range(num_images):
            iteration_start = datetime.now()
            if num_images > 1:
                seed = random.randint(0, MAX_SEED)
            generator = torch.Generator(device=device).manual_seed(seed)
            
            result_images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_embeds=face_emb,
                image=face_kps,
                controlnet_conditioning_scale=float(identitynet_strength_ratio),
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator
            ).images

            for img in result_images:
                current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                output_path = f"outputs/{current_time}.png"
                if not os.path.exists("outputs"):
                    os.makedirs("outputs")
                img.save(output_path)
                images.append(img)

            iteration_end = datetime.now()
            iteration_duration = (iteration_end - iteration_start).total_seconds()
            print(f"Image {i+1}/{num_images} generated in {iteration_duration} seconds.")
            
            if not enable_CPUOffload and i < (num_images - 1):
                reload_pipe(model_input, model_dropdown, adapter_strength_ratio, enable_CPUOffload)       
            

        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"Total inference time: {total_duration} seconds for {num_images} images.")
        return images, gr.update(visible=True)



    title = r"""
    <h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
    """

    description = r"""
    How to use:<br>
    1. Upload a person image. For multiple person images, we will only detect the biggest face. Make sure face is not too small and not significantly blocked or blurred.
    2. (Optionally) upload another person image as reference pose. If not uploaded, we will use the first person image to extract landmarks. If you use a cropped face at step1, it is recommeneded to upload it to extract a new pose.
    3. Enter a text prompt as done in normal text-to-image models.
    """
    article = r"""
    """
    tips = r"""
    ### Usage tips of InstantID
    1. If you're not satisfied with the similarity, try to increase the weight of "IdentityNet Strength" and "Adapter Strength".
    2. If you feel that the saturation is too high, first decrease the Adapter strength. If it is still too high, then decrease the IdentityNet strength.
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
                face_files = gr.Image(label="Upload a photo of your face", type="filepath")
                # uploaded_faces = gr.Gallery(label="Your images", visible=False, columns=1, rows=1, height=512)
                # clear_button_face = gr.ClearButton(value="Remove and upload new ones", components=[face_files], size="sm", visible=True)
        
                pose_files = gr.Image(label="Upload a reference pose image (optional)", type="filepath")
                # uploaded_poses = gr.Gallery(label="Your images", visible=False, columns=1, rows=1, height=512)
                # clear_button_pose = gr.ClearButton(value="Remove and upload new ones", components=[pose_files], size="sm", visible=True)
            with gr.Column():
                gallery = gr.Gallery(label="Generated Images", columns=1, rows=1, height=512)                
                usage_tips = gr.Markdown(tips, visible=False)

        with gr.Row():       
            with gr.Column():                
                enable_CPUOffload = gr.Checkbox(
                    label="Enable CPU Offload", value=True,
                    info="Offloading the weights to the CPU and only loading them on the GPU when performing the forward pass can also save memory.",                    
                )
        with gr.Row():
            with gr.Column():
                submit = gr.Button("Submit", variant="primary")
                prompt = gr.Textbox(label="Prompt", info="Give simple prompt is enough to achieve good face fidelity", placeholder="A photo of a person", value="")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="low quality", value="(text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, monochrome")        
                style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
        
                identitynet_strength_ratio = gr.Slider(label="IdentityNet strength (for fidelity)", minimum=0, maximum=1.5, step=0.05, value=0.80)
                adapter_strength_ratio = gr.Slider(label="Image adapter strength (for detail)", minimum=0, maximum=1.5, step=0.05, value=0.80)
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

                    num_steps = gr.Slider(label="Number of sample steps", minimum=20, maximum=100, step=1, value=30)
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.1, maximum=10.0, step=0.1, value=5)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        
                    # face_files.upload(fn=swap_to_gallery, inputs=face_files, outputs=[uploaded_faces, clear_button_face, face_files])
                    # pose_files.upload(fn=swap_to_gallery, inputs=pose_files, outputs=[uploaded_poses, clear_button_pose, pose_files])

                    # clear_button_face.click(fn=remove_back_to_files, inputs=[], outputs=[uploaded_faces, clear_button_face, face_files])
                    # clear_button_pose.click(fn=remove_back_to_files, inputs=[], outputs=[uploaded_poses, clear_button_pose, pose_files])

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
                            face_files, pose_files, prompt, negative_prompt, style, num_steps, 
                            identitynet_strength_ratio, adapter_strength_ratio, guidance_scale, 
                            seed, width, height, num_images, model_input, model_dropdown, enable_CPUOffload
                        ],
                        outputs=[gallery, usage_tips]
                    )
        gr.Markdown(article)
    demo.launch(inbrowser=True, share=share)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_folder", type=str, default=None
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio app sharing")
    args = parser.parse_args()

    main(args.pretrained_model_folder,args.share)