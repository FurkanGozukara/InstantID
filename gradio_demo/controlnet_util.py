import torch
import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector
import cv2
from diffusers.models import ControlNetModel
import os
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from transformers import DPTImageProcessor, DPTForDepthEstimation
from torchvision.transforms import Compose

depth_estimator = None
feature_extractor = None
openpose = None
controlnet_pose = None
controlnet_canny = None
controlnet_depth = None
controlnet = None
depth_anything = None 


def load_controlnet(pretrained_model_folder, dtype):
    global controlnet_pose, controlnet_canny, controlnet_depth

    # Load pipeline face ControlNetModel    
    controlnet_identity_model = f"checkpoints/ControlNetModel"
    
    openpose_model = "lllyasviel/ControlNet" if not pretrained_model_folder else fr"{pretrained_model_folder}/lllyasviel/Annotators"        
    #controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0" if not pretrained_model_folder else fr"{pretrained_model_folder}/thibaud/controlnet-openpose-sdxl-1.0"
    controlnet_pose_model = os.path.join("xinsir_controlnet", "controlnet-openpose-sdxl-1.0")

    #controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0" if not pretrained_model_folder else fr"{pretrained_model_folder}/diffusers/controlnet-canny-sdxl-1.0"
    controlnet_canny_model = os.path.join("xinsir_controlnet", "controlnet-canny-sdxl-1.0")

    #controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small" if not pretrained_model_folder else fr"{pretrained_model_folder}/diffusers/controlnet-depth-sdxl-1.0-small"
    controlnet_depth_model = os.path.join("xinsir_controlnet", "controlnet-depth-sdxl-1.0")

    openpose = OpenposeDetector.from_pretrained(openpose_model).to("cpu")

    controlnet_identity = ControlNetModel.from_pretrained(
        controlnet_identity_model, torch_dtype=dtype
        )
    
    controlnet_pose = ControlNetModel.from_pretrained(
    controlnet_pose_model, torch_dtype=dtype
    )

    controlnet_canny = ControlNetModel.from_pretrained(
        controlnet_canny_model, torch_dtype=dtype
    )

    controlnet_depth = ControlNetModel.from_pretrained(
        controlnet_depth_model, torch_dtype=dtype
    )
    
    # Check if we need to handle multi-GPU setup for Kaggle
    # Import args here to avoid circular imports
    import sys
    import argparse
    
    # Check if we're in Kaggle mode and have multiple GPUs
    # We need to access the args from the main module
    try:
        import __main__
        if hasattr(__main__, 'args') and __main__.args.kaggle and torch.cuda.device_count() >= 2:
            print("🔄 Placing ControlNet models on GPU 0 (Kaggle multi-GPU mode)")
            controlnet_identity = controlnet_identity.to('cuda:0')
            controlnet_pose = controlnet_pose.to('cuda:0')
            controlnet_canny = controlnet_canny.to('cuda:0')
            controlnet_depth = controlnet_depth.to('cuda:0')
            print("✅ All ControlNet models moved to GPU 0")
    except:
        # If we can't access args, just continue with default behavior
        pass
    
    return openpose, controlnet_pose, controlnet_canny, controlnet_depth, controlnet_identity

def load_depth_estimator(pretrained_model_folder, device, depth_type):
    global depth_estimator, feature_extractor
    
    # Check if we're in Kaggle mode and have multiple GPUs
    # For Kaggle multi-GPU setup, use GPU 0 for depth estimator
    target_device = device
    try:
        import __main__
        if hasattr(__main__, 'args') and __main__.args.kaggle and torch.cuda.device_count() >= 2:
            target_device = 'cuda:0'
            print("🔄 Using GPU 0 for depth estimator (Kaggle multi-GPU mode)")
    except:
        # If we can't access args, just use the default device
        pass
    
    if depth_type == 'LiheYoung/depth_anything':
        depth_estimator = None
        feature_extractor = None
        depth_anything_model = 'LiheYoung/depth_anything_vitl14' if not pretrained_model_folder else fr"{pretrained_model_folder}/LiheYoung/depth_anything_vitl14"
        depth_estimator = DepthAnything.from_pretrained(depth_anything_model).to(target_device).eval()
    else:
        depth_estimator_model = "Intel/dpt-hybrid-midas" if not pretrained_model_folder else fr"{pretrained_model_folder}/Intel/dpt-hybrid-midas"    
        feature_extractor = DPTImageProcessor.from_pretrained(depth_estimator_model)
        depth_estimator = DPTForDepthEstimation.from_pretrained(depth_estimator_model).to(target_device)    

    return depth_estimator

def get_depth_map(image):    
    # Determine the device based on where the depth estimator is located
    device_to_use = "cuda"
    try:
        import __main__
        if hasattr(__main__, 'args') and __main__.args.kaggle and torch.cuda.device_count() >= 2:
            device_to_use = 'cuda:0'  # Use GPU 0 for Kaggle multi-GPU mode
        elif depth_estimator is not None and hasattr(depth_estimator, 'device'):
            device_to_use = str(depth_estimator.device)
    except:
        # If we can't access args, just use the default device
        pass
    
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device_to_use)
    with torch.no_grad(), torch.autocast(device_to_use):
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

def get_depth_anything_map(image):    
    # Determine the device based on where the depth estimator is located
    device_to_use = "cuda"
    try:
        import __main__
        if hasattr(__main__, 'args') and __main__.args.kaggle and torch.cuda.device_count() >= 2:
            device_to_use = 'cuda:0'  # Use GPU 0 for Kaggle multi-GPU mode
        elif depth_estimator is not None and hasattr(depth_estimator, 'device'):
            device_to_use = str(depth_estimator.device)
    except:
        # If we can't access args, just use the default device
        pass
    
    transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])
    
    image = np.array(image) / 255.0

    h, w = image.shape[:2]

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device_to_use)

    with torch.no_grad():
        depth = depth_estimator(image)

    depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
    depth = depth.cpu().numpy().astype(np.uint8)
    depth_image = Image.fromarray(depth)
    return depth_image

def get_canny_image(image, t1=100, t2=200):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image, t1, t2)
    return Image.fromarray(edges, "L")