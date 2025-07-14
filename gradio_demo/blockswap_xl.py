"""
BlockSwap Module for SDXL UNet Models

This module implements dynamic block swapping between GPU and CPU memory
for SDXL UNet models to enable running on limited VRAM systems.

Key Features:
- Dynamic UNet block offloading during inference
- Non-blocking GPU transfers for optimal performance
- Support for down_blocks, mid_block, and up_blocks
- Minimal performance overhead with intelligent caching
- Memory usage tracking and debugging

Adapted from block_swap folder for SDXL UNet compatibility.
"""

import time
import types
import torch
import weakref
import psutil
import gc
from typing import Dict, Any, List, Tuple, Optional, Union
import torch.nn as nn


def get_module_memory_mb(module: torch.nn.Module) -> float:
    """
    Calculate memory usage of a module in MB.
    
    Args:
        module: PyTorch module to measure
        
    Returns:
        Memory usage in megabytes
    """
    total_bytes = sum(
        param.nelement() * param.element_size() 
        for param in module.parameters() 
        if param.data is not None
    )
    return total_bytes / (1024 * 1024)


class BlockSwapDebugger:
    """
    Debug logger for BlockSwap operations.
    
    Tracks memory usage, swap timings, and provides detailed logging
    for debugging and performance analysis of block swapping operations.
    """

    def __init__(self, enabled: bool = False):
        """
        Initialize the debugger.
        
        Args:
            enabled: Whether debug logging is enabled
        """
        self.enabled = enabled
        self.swap_times: List[Dict[str, Any]] = []
        self.vram_history: List[float] = []

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message if debugging is enabled."""
        if self.enabled:
            print(f"[BlockSwap {level}] {message}")

    def log_swap_time(self, component_id, duration: float, component_type: str = "block", direction: str = "compute") -> None:
        """
        Log swap timing information for any component (blocks or modules).
        
        Args:
            component_id: Block index (int) or module name (str)
            duration: Time taken for the swap operation
            component_type: Type of component ("down_block", "mid_block", "up_block", "module")
            direction: Direction of swap ("compute" or "offload")
        """
        if self.enabled:
            # Store timing data with component info
            self.swap_times.append({
                'component_id': component_id,
                'component_type': component_type,
                'duration': duration,
                'direction': direction
            })
            
            # Format message based on component type
            if component_type in ["down_block", "mid_block", "up_block"]:
                message = f"{component_type}[{component_id}] swap {direction}: {duration*1000:.1f}ms"
            else:
                message = f"{component_type} {component_id} swap {direction}: {duration*1000:.1f}ms"
            
            self.log(message, "SWAP")

    def log_memory_state(self, stage: str, show_tensors: bool = False) -> None:
        """Log current memory state for debugging."""
        if self.enabled:
            # GPU Memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self.log(f"{stage} - GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                self.vram_history.append(allocated)
            
            # CPU Memory
            process = psutil.Process()
            cpu_mem = process.memory_info().rss / 1024**3
            self.log(f"{stage} - CPU: {cpu_mem:.2f}GB")

    def clear_history(self) -> None:
        """Clear swap timing and memory history."""
        self.swap_times.clear()
        self.vram_history.clear()


def apply_block_swap_to_unet(pipe, block_swap_config: Dict[str, Any]) -> None:
    """
    Apply block swapping configuration to a SDXL UNet model.
    
    This is the main entry point for configuring block swapping on a UNet.
    It handles block selection and device placement for SDXL models.
    
    Args:
        pipe: Pipeline containing the UNet model
        block_swap_config: Configuration dictionary with keys:
            - blocks_to_swap: Number of blocks to swap from each section
            - swap_down_blocks: Whether to swap down blocks
            - swap_mid_block: Whether to swap mid block  
            - swap_up_blocks: Whether to swap up blocks
            - use_non_blocking: Whether to use non-blocking transfers
            - enable_debug: Whether to enable debug logging
    """
    if not block_swap_config:
        return

    blocks_to_swap = block_swap_config.get("blocks_to_swap", 0)
    if blocks_to_swap <= 0:
        return
    
    # Create debugger
    enable_debug = block_swap_config.get("enable_debug", False)
    debugger = BlockSwapDebugger(enabled=enable_debug)
    
    # Store debugger on pipeline for cleanup
    pipe._blockswap_debugger = debugger

    # Get the UNet model
    unet = pipe.unet
    if not hasattr(unet, 'down_blocks'):
        debugger.log("UNet doesn't have expected structure for BlockSwap", "WARN")
        return

    # Determine devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    offload_device = "cpu"
    use_non_blocking = block_swap_config.get("use_non_blocking", True)

    # Configure UNet with blockswap attributes
    unet.blocks_to_swap = blocks_to_swap
    unet.main_device = device
    unet.offload_device = offload_device
    unet.use_non_blocking = use_non_blocking

    debugger.log(f"Configuring BlockSwap for UNet: {blocks_to_swap} blocks per section")
    debugger.log_memory_state("Before BlockSwap", show_tensors=False)

    # Configure which block types to swap
    swap_down_blocks = block_swap_config.get("swap_down_blocks", True)
    swap_mid_block = block_swap_config.get("swap_mid_block", True)
    swap_up_blocks = block_swap_config.get("swap_up_blocks", True)

    memory_stats = _configure_unet_blocks(unet, device, offload_device, 
                                         use_non_blocking, debugger, 
                                         blocks_to_swap, swap_down_blocks, 
                                         swap_mid_block, swap_up_blocks)

    # Log memory summary
    _log_memory_summary(memory_stats, offload_device, device, debugger)
    
    # Wrap block forward methods for dynamic swapping
    if swap_down_blocks and hasattr(unet, 'down_blocks'):
        for i, block in enumerate(unet.down_blocks[:blocks_to_swap]):
            _wrap_unet_block_forward(block, f"down_{i}", "down_block", unet, debugger)

    if swap_mid_block and hasattr(unet, 'mid_block') and unet.mid_block is not None:
        _wrap_unet_block_forward(unet.mid_block, 0, "mid_block", unet, debugger)

    if swap_up_blocks and hasattr(unet, 'up_blocks'):
        for i, block in enumerate(unet.up_blocks[:blocks_to_swap]):
            _wrap_unet_block_forward(block, f"up_{i}", "up_block", unet, debugger)

    # Mark BlockSwap as active
    pipe._blockswap_active = True

    # Store configuration for debugging and cleanup
    pipe._block_swap_config = {
        "blocks_swapped": blocks_to_swap,
        "swap_down_blocks": swap_down_blocks,
        "swap_mid_block": swap_mid_block,
        "swap_up_blocks": swap_up_blocks,
        "use_non_blocking": use_non_blocking,
        "offload_device": offload_device,
        "main_device": device,
        "enable_debug": enable_debug,
        "offload_memory": memory_stats.get('offload_memory', 0),
        "main_memory": memory_stats.get('main_memory', 0)
    }

    # Protect UNet from being moved entirely
    _protect_unet_from_move(unet, pipe, debugger)

    debugger.log_memory_state("After BlockSwap", show_tensors=False)
    debugger.log("✅ BlockSwap configuration complete for UNet")


def _configure_unet_blocks(unet, device: str, offload_device: str, 
                          use_non_blocking: bool, debugger: BlockSwapDebugger,
                          blocks_to_swap: int, swap_down_blocks: bool,
                          swap_mid_block: bool, swap_up_blocks: bool) -> Dict[str, float]:
    """Configure UNet block placement and memory tracking."""
    
    memory_stats = {"offload_memory": 0.0, "main_memory": 0.0}
    
    # Configure down blocks
    if swap_down_blocks and hasattr(unet, 'down_blocks'):
        total_down_blocks = len(unet.down_blocks)
        blocks_to_move = min(blocks_to_swap, total_down_blocks)
        
        for i, block in enumerate(unet.down_blocks):
            if i < blocks_to_move:
                block_memory = get_module_memory_mb(block)
                block.to(offload_device, non_blocking=use_non_blocking)
                memory_stats["offload_memory"] += block_memory
                debugger.log(f"Moved down_block[{i}] to {offload_device} ({block_memory:.1f}MB)")
            else:
                block_memory = get_module_memory_mb(block)
                memory_stats["main_memory"] += block_memory

    # Configure mid block
    if swap_mid_block and hasattr(unet, 'mid_block') and unet.mid_block is not None:
        block_memory = get_module_memory_mb(unet.mid_block)
        unet.mid_block.to(offload_device, non_blocking=use_non_blocking)
        memory_stats["offload_memory"] += block_memory
        debugger.log(f"Moved mid_block to {offload_device} ({block_memory:.1f}MB)")

    # Configure up blocks
    if swap_up_blocks and hasattr(unet, 'up_blocks'):
        total_up_blocks = len(unet.up_blocks)
        blocks_to_move = min(blocks_to_swap, total_up_blocks)
        
        for i, block in enumerate(unet.up_blocks):
            if i < blocks_to_move:
                block_memory = get_module_memory_mb(block)
                block.to(offload_device, non_blocking=use_non_blocking)
                memory_stats["offload_memory"] += block_memory
                debugger.log(f"Moved up_block[{i}] to {offload_device} ({block_memory:.1f}MB)")
            else:
                block_memory = get_module_memory_mb(block)
                memory_stats["main_memory"] += block_memory

    return memory_stats


def _log_memory_summary(memory_stats: Dict[str, float], offload_device: str, 
                       device: str, debugger: BlockSwapDebugger) -> None:
    """Log memory allocation summary."""
    if debugger.enabled:
        debugger.log("=" * 50)
        debugger.log("BlockSwap Memory Summary:")
        debugger.log(f"  Main device ({device}): {memory_stats.get('main_memory', 0):.1f}MB")
        debugger.log(f"  Offload device ({offload_device}): {memory_stats.get('offload_memory', 0):.1f}MB")
        debugger.log(f"  Total managed: {memory_stats.get('main_memory', 0) + memory_stats.get('offload_memory', 0):.1f}MB")
        debugger.log("=" * 50)


def _wrap_unet_block_forward(block: torch.nn.Module, block_id: str, block_type: str, 
                            unet: torch.nn.Module, debugger: BlockSwapDebugger) -> None:
    """Wrap UNet block forward to handle device movement using weak references."""
    
    if hasattr(block, '_original_forward'):
        return  # Already wrapped

    # Store original forward method
    original_forward = block.forward
    
    # Create weak references
    unet_ref = weakref.ref(unet)
    debugger_ref = weakref.ref(debugger)
    
    # Store block info on the block itself
    block._block_id = block_id
    block._block_type = block_type
    
    def wrapped_forward(self, *args, **kwargs):
        # Retrieve weak references
        unet = unet_ref()
        debugger = debugger_ref()
        
        if not unet:
            # UNet has been garbage collected, fall back to original
            return original_forward(*args, **kwargs)

        # Check if block swap is active
        if hasattr(unet, 'blocks_to_swap'):
            t_start = time.time() if debugger and debugger.enabled else None

            # Only move to GPU if necessary
            current_device = next(self.parameters()).device
            target_device = torch.device(unet.main_device)
            
            if current_device != target_device:
                self.to(unet.main_device, non_blocking=unet.use_non_blocking)
                
            # Synchronize if needed
            if hasattr(unet, 'use_non_blocking') and not unet.use_non_blocking:
                torch.cuda.synchronize()

            # Execute forward pass
            output = original_forward(*args, **kwargs)

            # Move back to offload device
            self.to(unet.offload_device, non_blocking=unet.use_non_blocking)
            
            # Log timing if debugger is available
            if debugger and t_start is not None:
                debugger.log_swap_time(
                    component_id=self._block_id,
                    duration=time.time() - t_start,
                    component_type=self._block_type,
                    direction="compute"
                )

            # Clear cache under memory pressure
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > torch.cuda.get_device_properties(0).total_memory * 0.9:
                torch.cuda.empty_cache()
        else:
            output = original_forward(*args, **kwargs)

        return output

    # Bind the wrapped function as a method to the block
    block.forward = types.MethodType(wrapped_forward, block)
    
    # Store reference to original forward for cleanup
    block._original_forward = original_forward


def _protect_unet_from_move(unet, pipe, debugger: BlockSwapDebugger) -> None:
    """Protect UNet from being moved entirely to a different device."""
    
    if hasattr(unet, '_original_to'):
        return  # Already protected

    original_to = unet.to
    pipe_ref = weakref.ref(pipe)
    debugger_ref = weakref.ref(debugger)

    def protected_unet_to(self, device, *args, **kwargs):
        # Check blockswap status using weak reference
        pipe = pipe_ref()
        debugger = debugger_ref()
        
        if pipe and hasattr(pipe, '_blockswap_active') and pipe._blockswap_active:
            if debugger:
                debugger.log(f"Prevented UNet move to {device} - BlockSwap active", "PROTECT")
            return self  # Don't move when BlockSwap is active
        
        return original_to(device, *args, **kwargs)

    unet.to = types.MethodType(protected_unet_to, unet)
    unet._original_to = original_to


def cleanup_blockswap(pipe, keep_state_for_cache: bool = False) -> None:
    """
    Clean up block swap configuration and restore original methods.
    
    Args:
        pipe: Pipeline with BlockSwap applied
        keep_state_for_cache: Whether to keep state for potential reuse
    """
    if not hasattr(pipe, '_blockswap_active') or not pipe._blockswap_active:
        return

    debugger = getattr(pipe, '_blockswap_debugger', None)
    if debugger:
        debugger.log("🧹 Cleaning up BlockSwap...")

    unet = pipe.unet

    # Restore original UNet to method
    if hasattr(unet, '_original_to'):
        unet.to = unet._original_to
        delattr(unet, '_original_to')

    # Restore original forward methods and move blocks back to main device
    def restore_blocks(blocks, block_type):
        if not blocks:
            return
        for i, block in enumerate(blocks):
            if hasattr(block, '_original_forward'):
                block.forward = block._original_forward
                delattr(block, '_original_forward')
                
                # Move block back to main device if needed
                if hasattr(unet, 'main_device'):
                    block.to(unet.main_device)
                    
                if hasattr(block, '_block_id'):
                    delattr(block, '_block_id')
                if hasattr(block, '_block_type'):
                    delattr(block, '_block_type')

    # Restore down blocks
    if hasattr(unet, 'down_blocks'):
        restore_blocks(unet.down_blocks, "down_block")

    # Restore mid block
    if hasattr(unet, 'mid_block') and unet.mid_block is not None:
        if hasattr(unet.mid_block, '_original_forward'):
            unet.mid_block.forward = unet.mid_block._original_forward
            delattr(unet.mid_block, '_original_forward')
            if hasattr(unet, 'main_device'):
                unet.mid_block.to(unet.main_device)

    # Restore up blocks
    if hasattr(unet, 'up_blocks'):
        restore_blocks(unet.up_blocks, "up_block")

    # Clean up UNet attributes
    for attr in ['blocks_to_swap', 'main_device', 'offload_device', 'use_non_blocking']:
        if hasattr(unet, attr):
            delattr(unet, attr)

    # Clean up pipeline attributes
    if not keep_state_for_cache:
        for attr in ['_blockswap_active', '_blockswap_debugger', '_block_swap_config']:
            if hasattr(pipe, attr):
                delattr(pipe, attr)
    else:
        pipe._blockswap_active = False

    if debugger:
        debugger.log("✅ BlockSwap cleanup complete")


def get_block_swap_memory_info(pipe) -> Dict[str, Any]:
    """
    Get current BlockSwap memory information.
    
    Args:
        pipe: Pipeline with BlockSwap applied
        
    Returns:
        Dictionary with memory information
    """
    if not hasattr(pipe, '_block_swap_config'):
        return {"active": False}

    config = pipe._block_swap_config
    info = {
        "active": True,
        "blocks_swapped": config.get("blocks_swapped", 0),
        "offload_memory_mb": config.get("offload_memory", 0),
        "main_memory_mb": config.get("main_memory", 0),
        "swap_down_blocks": config.get("swap_down_blocks", True),
        "swap_mid_block": config.get("swap_mid_block", True),
        "swap_up_blocks": config.get("swap_up_blocks", True),
    }

    # Add current GPU memory if available
    if torch.cuda.is_available():
        info["current_gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        info["current_gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["gpu_utilization_percent"] = (info["current_gpu_allocated_gb"] / info["gpu_total_gb"]) * 100

    # Add CPU memory info
    try:
        process = psutil.Process()
        info["cpu_memory_gb"] = process.memory_info().rss / 1024**3
    except ImportError:
        pass  # psutil not available

    return info


def print_memory_summary(pipe):
    """
    Print a formatted memory summary for BlockSwap.
    
    Args:
        pipe: Pipeline with BlockSwap applied
    """
    info = get_block_swap_memory_info(pipe)
    
    if not info.get("active"):
        print("📊 BlockSwap: Not active")
        return
        
    print("📊 BlockSwap Memory Summary:")
    print(f"  🔄 Blocks swapped: {info['blocks_swapped']}")
    print(f"  💾 Offloaded memory: {info['offload_memory_mb']:.1f} MB")
    print(f"  🎮 Main memory: {info['main_memory_mb']:.1f} MB")
    
    if 'current_gpu_allocated_gb' in info:
        print(f"  🎯 GPU allocated: {info['current_gpu_allocated_gb']:.2f} GB")
        print(f"  📦 GPU reserved: {info['current_gpu_reserved_gb']:.2f} GB")
        if 'gpu_utilization_percent' in info:
            print(f"  📈 GPU utilization: {info['gpu_utilization_percent']:.1f}%")
    
    if 'cpu_memory_gb' in info:
        print(f"  💻 CPU memory: {info['cpu_memory_gb']:.2f} GB")


def optimize_memory_for_blockswap():
    """
    Optimize memory settings for BlockSwap usage.
    """
    try:
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print("🧹 Memory optimized for BlockSwap")
    except Exception as e:
        print(f"⚠️ Memory optimization failed: {e}") 