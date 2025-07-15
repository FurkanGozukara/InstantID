"""
VRAM Optimizer Module

This module provides comprehensive VRAM optimization for InstantID by combining
multiple memory optimization techniques including:
- CPU offloading
- Quantization
- BlockSwap
- Sequential loading
- Smart tensor management
- Dynamic memory monitoring
"""

import torch
import gc
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from common.util import aggressive_cleanup, get_memory_info, smart_cleanup


class VRAMOptimizer:
    """
    Comprehensive VRAM optimization manager that combines multiple techniques
    to minimize memory usage while maintaining performance.
    """
    
    def __init__(self, 
                 enable_cpu_offload: bool = True,
                 enable_quantization: bool = True,
                 enable_blockswap: bool = True,
                 enable_sequential_loading: bool = True,
                 enable_tensor_offloading: bool = True,
                 quantization_mode: str = "8bit",
                 blockswap_blocks: int = 2,
                 debug: bool = False):
        
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_quantization = enable_quantization
        self.enable_blockswap = enable_blockswap
        self.enable_sequential_loading = enable_sequential_loading
        self.enable_tensor_offloading = enable_tensor_offloading
        self.quantization_mode = quantization_mode
        self.blockswap_blocks = blockswap_blocks
        self.debug = debug
        
        # Memory monitoring
        self.memory_history = []
        self.optimization_active = False
        
        # Tensor cache for offloading
        self.tensor_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize optimization settings
        self._initialize_optimization()
        
        if self.debug:
            print("🔧 VRAMOptimizer initialized with:")
            print(f"  - CPU Offload: {self.enable_cpu_offload}")
            print(f"  - Quantization: {self.enable_quantization} ({self.quantization_mode})")
            print(f"  - BlockSwap: {self.enable_blockswap} ({self.blockswap_blocks} blocks)")
            print(f"  - Sequential Loading: {self.enable_sequential_loading}")
            print(f"  - Tensor Offloading: {self.enable_tensor_offloading}")
    
    def _initialize_optimization(self):
        """Initialize optimization settings"""
        # Optimize PyTorch settings for memory efficiency
        if torch.cuda.is_available():
            # Reduce memory fragmentation
            torch.cuda.empty_cache()
            
            # Optimize memory allocation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Disable tf32 for better memory usage and determinism
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.allow_tf32 = False
            
            # Improve determinism (helps minimize output differences)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Enable deterministic algorithms where possible
            try:
                torch.use_deterministic_algorithms(True)
            except:
                if self.debug:
                    print("⚠️ Deterministic algorithms not fully supported on this PyTorch version")
            
            if self.debug:
                print("🚀 CUDA optimization settings applied")
                print("🎯 Deterministic settings enabled for consistent outputs")
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get configuration for various optimization components"""
        config = {
            'cpu_offload': self.enable_cpu_offload,
            'quantization': {
                'enabled': self.enable_quantization,
                'mode': self.quantization_mode
            },
            'blockswap': {
                'enabled': self.enable_blockswap,
                'blocks_to_swap': self.blockswap_blocks,
                'swap_down_blocks': True,
                'swap_mid_block': True,
                'swap_up_blocks': True,
                'use_non_blocking': True
            },
            'sequential_loading': self.enable_sequential_loading,
            'tensor_offloading': self.enable_tensor_offloading
        }
        return config
    
    def optimize_model_loading(self, model_path: str, model_type: str = "auto") -> Dict[str, Any]:
        """Optimize model loading with sequential loading and memory management"""
        if not self.enable_sequential_loading:
            return {}
        
        # Get memory info before loading
        memory_before = get_memory_info()
        
        loading_config = {
            'torch_dtype': torch.float16 if self.enable_quantization else torch.float32,
            'low_cpu_mem_usage': True,
            'device_map': 'auto' if self.enable_cpu_offload else None,
            'variant': 'fp16' if self.enable_quantization else None
        }
        
        # Force CPU loading for large models
        if memory_before.get('gpu_free_gb', 0) < 4.0:
            loading_config['device_map'] = 'cpu'
            if self.debug:
                print(f"🚀 Forcing CPU loading due to low VRAM: {memory_before.get('gpu_free_gb', 0):.2f}GB")
        
        return loading_config
    
    def optimize_pipeline_creation(self, pipe_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize pipeline creation with memory-efficient settings"""
        optimized_kwargs = pipe_kwargs.copy()
        
        # Add memory optimization settings
        if self.enable_cpu_offload:
            optimized_kwargs['enable_cpu_offload'] = True
        
        # Force fp16 for memory savings
        if self.enable_quantization:
            optimized_kwargs['torch_dtype'] = torch.float16
        
        # Enable memory efficient attention
        optimized_kwargs['enable_attention_slicing'] = True
        optimized_kwargs['enable_vae_slicing'] = True
        optimized_kwargs['enable_vae_tiling'] = True
        
        return optimized_kwargs
    
    def cache_tensor(self, tensor: torch.Tensor, key: str, device: str = "cpu"):
        """Cache tensor to reduce memory usage"""
        if not self.enable_tensor_offloading:
            return
        
        if key in self.tensor_cache:
            # Clear previous cached tensor
            del self.tensor_cache[key]
        
        # Move tensor to specified device (usually CPU)
        cached_tensor = tensor.to(device)
        self.tensor_cache[key] = cached_tensor
        
        if self.debug:
            print(f"📦 Cached tensor '{key}' to {device}")
    
    def retrieve_tensor(self, key: str, target_device: Optional[str] = None) -> Optional[torch.Tensor]:
        """Retrieve cached tensor and move to target device"""
        if key not in self.tensor_cache:
            return None
        
        tensor = self.tensor_cache[key]
        
        if target_device:
            tensor = tensor.to(target_device)
        
        if self.debug:
            print(f"📥 Retrieved tensor '{key}' to {target_device or 'current device'}")
        
        return tensor
    
    def clear_cache(self):
        """Clear tensor cache"""
        for key in list(self.tensor_cache.keys()):
            del self.tensor_cache[key]
        self.tensor_cache.clear()
        
        if self.debug:
            print("🧹 Tensor cache cleared")
    
    def monitor_memory(self) -> Dict[str, float]:
        """Monitor memory usage and return current stats"""
        memory_info = get_memory_info()
        
        # Add to history
        self.memory_history.append({
            'timestamp': time.time(),
            'memory_info': memory_info
        })
        
        # Keep only last 100 entries
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
        
        return memory_info
    
    def should_optimize(self, threshold_gb: float = 1.0) -> bool:
        """Check if optimization should be triggered based on memory usage"""
        memory_info = self.monitor_memory()
        free_vram = memory_info.get('gpu_free_gb', 0)
        
        return free_vram < threshold_gb
    
    def optimize_inference_step(self, step_callback=None):
        """Optimize during inference step"""
        if not self.optimization_active:
            return
        
        # Smart cleanup during inference
        if self.should_optimize():
            if self.debug:
                print("🔄 Running inference optimization")
            
            # Clear unnecessary caches
            aggressive_cleanup()
            
            if step_callback:
                step_callback()
    
    def start_optimization(self):
        """Start optimization mode"""
        self.optimization_active = True
        
        if self.debug:
            print("🚀 VRAM optimization started")
        
        # Initial memory cleanup
        aggressive_cleanup()
        
        # Log initial memory state
        memory_info = self.monitor_memory()
        if self.debug:
            print(f"📊 Initial VRAM: {memory_info.get('gpu_allocated_gb', 0):.2f}GB allocated, "
                  f"{memory_info.get('gpu_free_gb', 0):.2f}GB free")
    
    def stop_optimization(self):
        """Stop optimization mode"""
        self.optimization_active = False
        
        # Final cleanup
        aggressive_cleanup()
        self.clear_cache()
        
        if self.debug:
            print("🛑 VRAM optimization stopped")
            
            # Log final memory state
            memory_info = self.monitor_memory()
            print(f"📊 Final VRAM: {memory_info.get('gpu_allocated_gb', 0):.2f}GB allocated, "
                  f"{memory_info.get('gpu_free_gb', 0):.2f}GB free")
    
    def get_memory_report(self) -> str:
        """Get detailed memory usage report"""
        memory_info = self.monitor_memory()
        
        report = "📊 VRAM Optimization Report\n"
        report += "=" * 40 + "\n"
        
        # GPU Memory
        if torch.cuda.is_available():
            report += f"GPU Memory:\n"
            report += f"  Allocated: {memory_info.get('gpu_allocated_gb', 0):.2f}GB\n"
            report += f"  Cached: {memory_info.get('gpu_cached_gb', 0):.2f}GB\n"
            report += f"  Peak: {memory_info.get('gpu_peak_gb', 0):.2f}GB\n"
            report += f"  Total: {memory_info.get('gpu_total_gb', 0):.2f}GB\n"
            report += f"  Free: {memory_info.get('gpu_free_gb', 0):.2f}GB\n"
            report += f"  Utilization: {(memory_info.get('gpu_allocated_gb', 0) / memory_info.get('gpu_total_gb', 1)) * 100:.1f}%\n"
        
        # System Memory
        report += f"System Memory:\n"
        report += f"  Usage: {memory_info.get('system_ram_gb', 0):.2f}GB\n"
        report += f"  Percent: {memory_info.get('system_ram_percent', 0):.1f}%\n"
        
        # Optimization Status
        report += f"Optimization Status:\n"
        report += f"  Active: {self.optimization_active}\n"
        report += f"  Cached Tensors: {len(self.tensor_cache)}\n"
        report += f"  History Length: {len(self.memory_history)}\n"
        
        return report
    
    def __enter__(self):
        """Context manager entry"""
        self.start_optimization()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_optimization()


def create_vram_optimizer(args) -> VRAMOptimizer:
    """Create VRAMOptimizer instance based on command line arguments"""
    if not args.vram_optimize:
        return None
    
    # Determine optimization settings
    enable_cpu_offload = args.lowvram or args.vram_optimize
    enable_quantization = bool(args.load_mode) or args.vram_optimize
    quantization_mode = args.load_mode or "8bit"
    
    # Create optimizer
    optimizer = VRAMOptimizer(
        enable_cpu_offload=enable_cpu_offload,
        enable_quantization=enable_quantization,
        enable_blockswap=args.vram_optimize,
        enable_sequential_loading=args.vram_optimize,
        enable_tensor_offloading=args.vram_optimize,
        quantization_mode=quantization_mode,
        blockswap_blocks=3 if args.vram_optimize else 2,  # More aggressive with --vram_optimize
        debug=False  # Set to True for detailed logging
    )
    
    return optimizer 