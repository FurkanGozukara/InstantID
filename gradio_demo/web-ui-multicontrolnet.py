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
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from style_template import styles
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl, get_torch_device, torch_gc
from controlnet_util import openpose, get_depth_map, get_canny_image

import gradio as gr


# global variable
MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"

# Load face encoder
app = FaceAnalysis(
    name="antelopev2",
    root="./",
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = f"./checkpoints/ip-adapter.bin"
controlnet_path = f"./checkpoints/ControlNetModel"

# Load pipeline face ControlNetModel
controlnet_identitynet = ControlNetModel.from_pretrained(
    controlnet_path, torch_dtype=dtype
)

# controlnet-pose
controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

controlnet_pose = ControlNetModel.from_pretrained(
    controlnet_pose_model, torch_dtype=dtype
)
controlnet_canny = ControlNetModel.from_pretrained(
    controlnet_canny_model, torch_dtype=dtype
)
controlnet_depth = ControlNetModel.from_pretrained(
    controlnet_depth_model, torch_dtype=dtype
)

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

def get_model_names():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.safetensors')]
    return model_files

def assign_last_params():
    global pipe
    

    #pipe.to(device)
    
    pipe.load_ip_adapter_instantid(face_adapter)
    
    
    pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            # load and disable LCM
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    pipe.disable_lora()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main(pretrained_model_name_or_path="wangqixun/YamerMIX_v8", enable_lcm_arg=False, share=False):

    global pipe  # Declare pipe as a global variable to manage it when the model changes
    
    last_loaded_model_path = pretrained_model_name_or_path  # Track the last loaded model path

    def load_model(pretrained_model_name_or_path):
        if pretrained_model_name_or_path.endswith(
            ".ckpt"
        ) or pretrained_model_name_or_path.endswith(".safetensors"):
            scheduler_kwargs = hf_hub_download(
                repo_id="wangqixun/YamerMIX_v8",
                subfolder="scheduler",
                filename="scheduler_config.json",
            )

            (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
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
                controlnet=[controlnet_identitynet],
            )

        else:
            pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                pretrained_model_name_or_path,
                controlnet=[controlnet_identitynet],
                torch_dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
            )

            pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
        return pipe

    
    print(f"Loading model: {pretrained_model_name_or_path}")
    pipe = load_model(pretrained_model_name_or_path)

    assign_last_params()

    def reload_pipe_if_needed(model_input, model_dropdown):
        nonlocal last_loaded_model_path

        # Trim the model_input to remove any leading or trailing whitespace
        model_input = model_input.strip() if model_input else None

        # Determine the model to load
        model_to_load = model_input if model_input else os.path.join('models', model_dropdown) if model_dropdown else None

        # Return early if no model is selected or inputted
        if not model_to_load:
            print("No model selected or inputted. Please select or input a model. Default model will be used.")
            return

        # Proceed with reloading the model if it's different from the last loaded model
        if model_to_load != last_loaded_model_path:
            print(f"Reloading model: {model_to_load}")
            global pipe
            # Properly discard the old pipe if it exists
            if hasattr(pipe, 'scheduler'):
                del pipe.scheduler

            # Load the new model
            pipe = load_model(model_to_load)
            last_loaded_model_path = model_to_load
            assign_last_params()

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
        progress=gr.Progress(track_tqdm=True),
    ):
        reload_pipe_if_needed(model_input, model_dropdown)
        if enable_LCM:
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

        if len(controlnet_selection) > 0:
            controlnet_scales = {
                "pose": pose_strength,
                "canny": canny_strength,
                "depth": depth_strength,
            }
            pipe.controlnet = MultiControlNetModel(
                [controlnet_identitynet]
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
            pipe.controlnet = controlnet_identitynet
            control_scales = float(identitynet_strength_ratio)
            control_images = face_kps

        generator = torch.Generator(device=device).manual_seed(seed)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        images_generated = []
        start_time = time.time()
        for i in range(num_images):
            if num_images > 1:
                seed = random.randint(0, MAX_SEED)
            generator = torch.Generator(device=device).manual_seed(seed)
            pipe.enable_xformers_memory_efficient_attention()
            #pipe.enable_model_cpu_offload()
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
            ).images

            for img in result_images:
                current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                output_path = f"outputs/{current_time}.png"
                if not os.path.exists("outputs"):
                    os.makedirs("outputs")
                img.save(output_path)
                images_generated.append(img)

            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            print(f"Image {i + 1}/{num_images} generated in {iteration_time:.2f} seconds.")

        total_time = time.time() - start_time
        average_time_per_image = total_time / num_images if num_images else 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{len(images_generated)} images generated in {total_time:.2f} seconds, average {average_time_per_image:.2f} seconds per image.")
        return images_generated, gr.update(visible=True)

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
                        ["pose", "canny", "depth"], label="Controlnet", value=["pose"],
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
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=20.0,
                        step=0.1,
                        value=0.0 if enable_lcm_arg else 5.0,
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
                        "EulerDiscreteScheduler",
                        "DPMSolverMultistepScheduler",
                        "DPMSolverMultistepScheduler-Karras",
                        "DPMSolverMultistepScheduler-Karras-SDE",
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
                    enhance_face_region,model_input,model_dropdown,width,height,num_images
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="wangqixun/YamerMIX_v8"
    )
    parser.add_argument(
        "--enable_LCM", type=bool, default=os.environ.get("ENABLE_LCM", False)
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio app sharing")

    args = parser.parse_args()

    main(args.pretrained_model_name_or_path, args.enable_LCM,args.share)