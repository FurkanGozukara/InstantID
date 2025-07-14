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

def quantize_4bit(module, module_name="Module", progress_callback=None):
    """
    Quantize a PyTorch module to 4-bit using bitsandbytes.
    
    Args:
        module: PyTorch module to quantize
        module_name: Name of the module for progress display
        progress_callback: Optional callback function for progress updates
    """
    if not BNB_AVAILABLE:
        if progress_callback:
            progress_callback(f"⚠️ bitsandbytes not available. Skipping 4-bit quantization for {module_name}")
        else:
            print("Warning: bitsandbytes not available. Skipping 4-bit quantization.")
        return
    
    # Safety check: Don't quantize VAE components - be very aggressive about protection
    module_type = str(type(module).__name__).lower()
    module_path = str(type(module)).lower()
    
    # Check for VAE-related components
    vae_keywords = ['vae', 'encoder', 'decoder', 'autoencoder', 'variational']
    if any(keyword in module_type for keyword in vae_keywords) or any(keyword in module_path for keyword in vae_keywords):
        if progress_callback:
            progress_callback(f"🚫 Skipping quantization for VAE component: {module_name}")
        else:
            print(f"Skipping quantization for VAE component: {type(module).__name__}")
        return
    
    # Count total linear layers first for progress tracking
    total_linear_layers = 0
    def count_linear_layers(m):
        nonlocal total_linear_layers
        for child in m.children():
            if isinstance(child, torch.nn.Linear):
                total_linear_layers += 1
            else:
                count_linear_layers(child)
    
    count_linear_layers(module)
    
    if total_linear_layers == 0:
        if progress_callback:
            progress_callback(f"ℹ️ No Linear layers found in {module_name}")
        return
    
    if progress_callback:
        progress_callback(f"🔄 Quantizing {module_name}: 0/{total_linear_layers} layers")
    
    quantized_layers = 0
    try:
        def quantize_linear_layers(m, parent_name=""):
            nonlocal quantized_layers
            for name, child in m.named_children():
                current_name = f"{parent_name}.{name}" if parent_name else name
                
                if isinstance(child, torch.nn.Linear):
                    in_features = child.in_features
                    out_features = child.out_features
                    device = child.weight.data.device

                    # Create and configure the Linear layer
                    has_bias = True if child.bias is not None else False
                    
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
                    setattr(m, name, new_layer)
                    quantized_layers += 1
                    
                    # Update progress less frequently to speed up
                    if quantized_layers % max(1, total_linear_layers // 10) == 0 or quantized_layers == total_linear_layers:
                        if progress_callback:
                            progress_callback(f"🔄 Quantizing {module_name}: {quantized_layers}/{total_linear_layers} layers")
                else:
                    # Recursively apply to child modules
                    quantize_linear_layers(child, current_name)
        
        quantize_linear_layers(module)
        
        if quantized_layers > 0:
            if progress_callback:
                progress_callback(f"✅ {module_name}: {quantized_layers} layers quantized to 4-bit")
            else:
                print(f"Successfully quantized {quantized_layers} Linear layers in {type(module).__name__} to 4-bit")
            
    except Exception as e:
        error_msg = f"❌ Error quantizing {module_name}: {e}"
        if progress_callback:
            progress_callback(error_msg)
        else:
            print(f"Error during 4-bit quantization: {e}")
            print("Continuing without quantization...")

def quantize_8bit(module, module_name="Module", progress_callback=None):
    """
    Quantize a PyTorch module to 8-bit using bitsandbytes.
    
    Args:
        module: PyTorch module to quantize
        module_name: Name of the module for progress display
        progress_callback: Optional callback function for progress updates
    """
    if not BNB_AVAILABLE:
        if progress_callback:
            progress_callback(f"⚠️ bitsandbytes not available. Skipping 8-bit quantization for {module_name}")
        else:
            print("Warning: bitsandbytes not available. Skipping 8-bit quantization.")
        return
    
    # Safety check: Don't quantize VAE components - be very aggressive about protection
    module_type = str(type(module).__name__).lower()
    module_path = str(type(module)).lower()
    
    # Check for VAE-related components
    vae_keywords = ['vae', 'encoder', 'decoder', 'autoencoder', 'variational']
    if any(keyword in module_type for keyword in vae_keywords) or any(keyword in module_path for keyword in vae_keywords):
        if progress_callback:
            progress_callback(f"🚫 Skipping quantization for VAE component: {module_name}")
        else:
            print(f"Skipping quantization for VAE component: {type(module).__name__}")
        return
    
    # Count total linear layers first for progress tracking
    total_linear_layers = 0
    def count_linear_layers(m):
        nonlocal total_linear_layers
        for child in m.children():
            if isinstance(child, torch.nn.Linear):
                total_linear_layers += 1
            else:
                count_linear_layers(child)
    
    count_linear_layers(module)
    
    if total_linear_layers == 0:
        if progress_callback:
            progress_callback(f"ℹ️ No Linear layers found in {module_name}")
        return
    
    if progress_callback:
        progress_callback(f"🔄 Quantizing {module_name}: 0/{total_linear_layers} layers")
    
    quantized_layers = 0
    try:
        def quantize_linear_layers(m, parent_name=""):
            nonlocal quantized_layers
            for name, child in m.named_children():
                current_name = f"{parent_name}.{name}" if parent_name else name
                
                if isinstance(child, torch.nn.Linear):
                    in_features = child.in_features
                    out_features = child.out_features
                    device = child.weight.data.device

                    # Create and configure the Linear layer
                    has_bias = True if child.bias is not None else False
                    
                    # MEMORY FIX: Set has_fp16_weights=False for proper 8-bit quantization
                    # has_fp16_weights=True keeps full precision weights + quantized weights = MORE memory
                    # has_fp16_weights=False stores only quantized weights = LESS memory
                    new_layer = bnb.nn.Linear8bitLt(
                        in_features,
                        out_features,
                        bias=has_bias,
                        has_fp16_weights=False,  # This is the key fix for memory reduction
                        threshold=6.0,
                    )

                    # Use the standard load_state_dict approach (faster than manual copying)
                    new_layer.load_state_dict(child.state_dict())
                    new_layer = new_layer.to(device)

                    # Set the attribute
                    setattr(m, name, new_layer)
                    quantized_layers += 1
                    
                    # Update progress less frequently to speed up
                    if quantized_layers % max(1, total_linear_layers // 10) == 0 or quantized_layers == total_linear_layers:
                        if progress_callback:
                            progress_callback(f"🔄 Quantizing {module_name}: {quantized_layers}/{total_linear_layers} layers")
                else:
                    # Recursively apply to child modules
                    quantize_linear_layers(child, current_name)
        
        quantize_linear_layers(module)
        
        if quantized_layers > 0:
            if progress_callback:
                progress_callback(f"✅ {module_name}: {quantized_layers} layers quantized to 8-bit")
            else:
                print(f"Successfully quantized {quantized_layers} Linear layers in {type(module).__name__} to 8-bit")
            
    except Exception as e:
        error_msg = f"❌ Error quantizing {module_name}: {e}"
        if progress_callback:
            progress_callback(error_msg)
        else:
            print(f"Error during 8-bit quantization: {e}")
            print("Continuing without quantization...")