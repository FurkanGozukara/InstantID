import torch
import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector
import cv2
from diffusers.models import ControlNetModel

from transformers import DPTImageProcessor, DPTForDepthEstimation

depth_estimator = None
feature_extractor = None
openpose = None
controlnet_pose = None
controlnet_canny = None
controlnet_depth = None
controlnet = None

def load_controlnet(pretrained_model_folder, controlnet_selection, device, dtype):
    global depth_estimator, feature_extractor, controlnet_pose, controlnet_canny, controlnet_depth

    # Load pipeline face ControlNetModel    
    controlnet_identity_model = f"checkpoints/ControlNetModel"
    depth_estimator_model = "Intel/dpt-hybrid-midas" if not pretrained_model_folder else fr"{pretrained_model_folder}/Intel/dpt-hybrid-midas"    
    openpose_model = "lllyasviel/ControlNet" if not pretrained_model_folder else fr"{pretrained_model_folder}/lllyasviel/Annotators"        
    controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0" if not pretrained_model_folder else fr"{pretrained_model_folder}/thibaud/controlnet-openpose-sdxl-1.0"
    controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0" if not pretrained_model_folder else fr"{pretrained_model_folder}/diffusers/controlnet-canny-sdxl-1.0"
    controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small" if not pretrained_model_folder else fr"{pretrained_model_folder}/diffusers/controlnet-depth-sdxl-1.0-small"

    depth_estimator = DPTForDepthEstimation.from_pretrained(depth_estimator_model).to(device)
    feature_extractor = DPTImageProcessor.from_pretrained(depth_estimator_model)
    openpose = OpenposeDetector.from_pretrained(openpose_model).to("cpu")

    controlnet_identity = ControlNetModel.from_pretrained(
        controlnet_identity_model, torch_dtype=dtype
        )
    
    if "pose" in controlnet_selection:
        controlnet_pose = ControlNetModel.from_pretrained(
        controlnet_pose_model, torch_dtype=dtype
        )
    if "canny" in controlnet_selection:
        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_canny_model, torch_dtype=dtype
        )
    if "depth" in controlnet_selection:
        controlnet_depth = ControlNetModel.from_pretrained(
            controlnet_depth_model, torch_dtype=dtype
        )
    return openpose, controlnet_pose, controlnet_canny, controlnet_depth, controlnet_identity

def get_depth_map(image):    
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def get_canny_image(image, t1=100, t2=200):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image, t1, t2)
    return Image.fromarray(edges, "L")