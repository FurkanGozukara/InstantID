import torch
from torch import nn
from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("Warning: bitsandbytes not available. Quantization will be disabled.")

def optionally_disable_offloading(_pipeline):
    """
    Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

    Args:
        _pipeline (`DiffusionPipeline`):
            The pipeline to disable offloading for.

    Returns:
        tuple:
            A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
    """
    is_model_cpu_offload = False
    is_sequential_cpu_offload = False
    print(
            "Restarting CPU Offloading..."
          )
    if _pipeline is not None:
        for _, component in _pipeline.components.items():
            if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                if not is_model_cpu_offload:
                    is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                if not is_sequential_cpu_offload:
                    is_sequential_cpu_offload = isinstance(component._hf_hook, AlignDevicesHook)

               
                remove_hook_from_module(component, recurse=True)

    return (is_model_cpu_offload, is_sequential_cpu_offload)

def quantize_4bit(module):
    if not BNB_AVAILABLE:
        print("Warning: bitsandbytes not available. Skipping 4-bit quantization.")
        return
    
    # Safety check: Don't quantize VAE components - be very aggressive about protection
    module_name = str(type(module).__name__).lower()
    module_path = str(type(module)).lower()
    
    # Check for VAE-related components
    vae_keywords = ['vae', 'encoder', 'decoder', 'autoencoder', 'variational']
    if any(keyword in module_name for keyword in vae_keywords) or any(keyword in module_path for keyword in vae_keywords):
        print(f"Skipping quantization for VAE component: {type(module).__name__}")
        return
    
    try:
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                in_features = child.in_features
                out_features = child.out_features
                device = child.weight.data.device

                # Create and configure the Linear layer
                has_bias = True if child.bias is not None else False
                
                # TODO: Make that configurable
                # fp16 for compute dtype leads to faster inference
                # and one should almost always use nf4 as a rule of thumb
                bnb_4bit_compute_dtype = torch.float16
                quant_type = "nf4"

                new_layer = bnb.nn.Linear4bit(
                    in_features,
                    out_features,
                    bias=has_bias,
                    compute_dtype=bnb_4bit_compute_dtype,
                    quant_type=quant_type,
                )

                new_layer.load_state_dict(child.state_dict())
                new_layer = new_layer.to(device)

                # Set the attribute
                setattr(module, name, new_layer)
            else:
                # Recursively apply to child modules
                quantize_4bit(child)
    except Exception as e:
        print(f"Error during 4-bit quantization: {e}")
        print("Continuing without quantization...")

def quantize_8bit(module):
    if not BNB_AVAILABLE:
        print("Warning: bitsandbytes not available. Skipping 8-bit quantization.")
        return
    
    # Safety check: Don't quantize VAE components - be very aggressive about protection
    module_name = str(type(module).__name__).lower()
    module_path = str(type(module)).lower()
    
    # Check for VAE-related components
    vae_keywords = ['vae', 'encoder', 'decoder', 'autoencoder', 'variational']
    if any(keyword in module_name for keyword in vae_keywords) or any(keyword in module_path for keyword in vae_keywords):
        print(f"Skipping quantization for VAE component: {type(module).__name__}")
        return
    
    try:
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                in_features = child.in_features
                out_features = child.out_features
                device = child.weight.data.device

                # Create and configure the Linear layer
                has_bias = True if child.bias is not None else False
                
                # Use fp16 for compute dtype
                bnb_8bit_compute_dtype = torch.float16

                new_layer = bnb.nn.Linear8bitLt(
                    in_features,
                    out_features,
                    bias=has_bias,
                    has_fp16_weights=False,
                    threshold=6.0,
                )

                new_layer.load_state_dict(child.state_dict())
                new_layer = new_layer.to(device)

                # Set the attribute
                setattr(module, name, new_layer)
            else:
                # Recursively apply to child modules
                quantize_8bit(child)
    except Exception as e:
        print(f"Error during 8-bit quantization: {e}")
        print("Continuing without quantization...")